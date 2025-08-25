import ray
import atexit
import json
import os
import numpy as np
from collections import Counter
import filelock
from vllm import SamplingParams, LLM
from transformers import AutoProcessor  
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import copy
from io import BytesIO
import base64
import random

from pixel_grounding.misc import generate_id, get_date, load_and_resize_image
import re

# ref: screenspot-pro/models/qwen2_5vl.py
def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ref: screenspot-pro/models/qwen2_5vl.py
def get_qwen2_5vl_prompt_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI reasoning agent."
                },
                {
                    "type": "text",
                    "text": """
You are given an  **instruction** and a **screenshot** of a user interface.  
Your task is to **think step by step** about which point to click in the screenshot to complish the instruction.
Then, output the result in the following format:
<think>your step-by-step reasoning here</think>  
<action>click(x,y)</action>

The <think> section should include the reason why you click on a certain point, such as an icon or text's label, and how it relates to the instruction.
"""
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Instruction: " + instruction + "\n"
                },
                {
                    "type": "text",
                    "text": "The screenshot below has resolution: {{screen_width}}x{{screen_height}}\n".replace("{{screen_width}}", str(screen_width)).replace("{{screen_height}}", str(screen_height))
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
            ]
        }
    ]


def setup_ray():
    if not ray.is_initialized():
        print(f"ray is not initialized, initializing...")
        ray.init(num_gpus= WORKER_NODE, num_cpus= 16)
    def cleanup_ray():
        if ray.is_initialized():
            ray.shutdown()
            print("closing ray...")
    atexit.register(cleanup_ray)


# setting for vlm
from pixel_grounding.misc import TENSOR_PARALLEL_SIZE
from pixel_grounding.models.qwen2_5vl import CustomQwen2_5VL_VLLM_Model
LIMIT_MM_PER_PROMPT={
    "image": 1
}
MAX_NUM_SEQS= 16
MAX_MODEL_LEN= (1024*16) # reduce it when OOM!


# from pixel_grounding.build_prompt import build_qwen_focus_cot_with_example

@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class InferenceActor:
    def __init__(self, model_name, actor_id:int):
        self.model_name = model_name
        self.actor_id = actor_id
        
        # processor llm
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"#####model_name: [{self.model_name}]#########")
        print(f"#####TENSOR_PARALLEL_SIZE: [{TENSOR_PARALLEL_SIZE}]#########")
        print(f"#####LIMIT_MM_PER_PROMPT: [{LIMIT_MM_PER_PROMPT}]#########")
        print(f"#####MAX_NUM_SEQS: [{MAX_NUM_SEQS}]#########")
        print(f"#####MAX_MODEL_LEN: [{MAX_MODEL_LEN}]#########")
        self.llm = LLM(
            model=self.model_name, 
            dtype="auto", 
            tensor_parallel_size= TENSOR_PARALLEL_SIZE,
            limit_mm_per_prompt= LIMIT_MM_PER_PROMPT,
            max_num_seqs= MAX_NUM_SEQS,
            max_model_len= MAX_MODEL_LEN, # max_model_len should be ????
            max_num_batched_tokens= MAX_MODEL_LEN
            # gpu_memory_utilization= 0.95,
        )

    def generate_one_by_one(self, llm_inputs: list, sampling_params: SamplingParams) -> list[str]:
        responses = []
        finish_reasons = []
        for idx, llm_input in enumerate(llm_inputs):
            try:
                print(f"[{self.actor_id}]->sampling_params: {sampling_params}")
                outs = self.llm.generate([llm_input], sampling_params=sampling_params)
                responses.append(outs[0].outputs[0].text)
                finish_reasons.append(outs[0].outputs[0].finish_reason)
            except Exception as e:
                responses.append("")
                finish_reasons.append("Error_in_Generate")
        return responses, finish_reasons
    
    def set_response_file(self, response_file: str):
        self.response_file = response_file
    def write_file(self, items: list[dict]):
        if not hasattr(self, "response_file"):
            return
        if not hasattr(self, "file_lock"):
            self.file_lock = filelock.FileLock("tmp/example.lock")
        with self.file_lock:
            with open(self.response_file, 'a') as fa:
                for item in items:
                    fa.write(json.dumps(item)+'\n')

    def write_one_by_one(self, item: dict):
        with open("temp.jsonl", 'a') as fa:
            fa.write(json.dumps(item)+'\n')

    def infer(self, samples: list, sampling_params: SamplingParams, batch_size: int= 16):

        # conversation -> get image -> replace with and height
        llm_inputs = []
        raw_images = []
        for sample in samples:
            # 注意设置好 min-pixex 和 max-pixel
            image_url = sample['crop_image_url']
            instruction = sample['instruction']
            raw_image, resize_image = load_and_resize_image(
                image_url= image_url,
                factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                min_pixels= self.processor.image_processor.min_pixels,
                max_pixels=MAX_PIXELS,
            )
            # organize data: ref: screenspot-pro/models/qwen2_5vl.py
            messages = get_qwen2_5vl_prompt_msg(
                image= resize_image,
                instruction= instruction,
                screen_width= resize_image.width,
                screen_height= resize_image.height,
            )
            # messages = build_qwen_focus_cot_with_example(
            #     image_example= RESIZE_IMAGE_EXAMPLE,
            #     instruction_example= INSTRUCTION_EXAMPLE,
            #     screen_width_example= RESIZE_IMAGE_EXAMPLE.width,
            #     screen_height_example= RESIZE_IMAGE_EXAMPLE.height,
            #     cot_example= COT_EXAMPLE,
            #     focus_box_example= FOCUS_BOX_EXAMPLE,
            #     instruction= instruction,
            #     image= resized_image,
            #     screen_width= resized_width,
            #     screen_height= resized_height,
            # )
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": [resize_image]}
            }
            raw_images.append(raw_image)
            llm_inputs.append(llm_input)
        for batch_start in tqdm(range(0, len(llm_inputs), batch_size), desc= "generation"):
            batch_inputs = llm_inputs[batch_start:batch_start+batch_size]
            batch_samples = samples[batch_start:batch_start+batch_size]
            batch_raw_images = raw_images[batch_start:batch_start+batch_size]
            # sampling parameter 设置采样的次数
            responses = []
            # batch infer
            batch_outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
            responses = [[o.text for o in output.outputs] for output in batch_outputs]
            finish_reasons = [[o.finish_reason for o in output.outputs] for output in batch_outputs]
            log_response_answers = []
            for idx_in_batch, (llm_input, response, finish_reason, sample, raw_image) in enumerate(zip(batch_inputs, responses, finish_reasons, batch_samples, batch_raw_images)):
                # just write into file now.
                # 注意image的大小信息
                # assert len(response) == len(finish_reason) == VOTING_TIMES
                run_image = llm_input['multi_modal_data']['image'][0]
                item = copy.deepcopy(sample)
                item['running'] = {}
                item['running']['prompt'] = llm_input['prompt']
                item['running']['response'] = response
                item['running']['finish_reason'] = finish_reason
                item['running']['run_image_info'] = {
                    "raw_width": raw_image.width,
                    "raw_height": raw_image.height,
                    "width": run_image.width,
                    "height": run_image.height,
                }
                log_response_answers.append(item)
                # self.write_one_by_one(item)
            # write to file
            print(f"Actor@{self.actor_id} is writting into file ...")
            self.write_file(log_response_answers)

        metric = Counter()
        return metric
    


def train_data_reader(max_samples: int= 20_000):
    while True:
        existed_ids = set()
        if os.path.exists(RESPONSE_FILE):
            existed_ids = set([json.loads(line)['id'] for line in open(RESPONSE_FILE)])
        
        json_items = []
        # show-ui desktop
        FILTER_DATA_FILE = "/public/home/ljt/hqp/GUI-grounding/generate/cot/grounding-R1-cot_focus_expand_limit_box_post.jsonl"
        # IMAGE_DIR = "/public/home/ljt/hqp/GUI-grounding/train_data/grounding-r1"
        print(f"[INFO] load from [{FILTER_DATA_FILE}]")
        with open(FILTER_DATA_FILE, 'r') as f:     ###showui 
            for line in f:
                item = json.loads(line)
                if item['id'] in existed_ids: continue
                # make sure crop image url
                try:
                    img_file = item['crop_image_url']
                    with Image.open(img_file) as img:
                        img.verify()  # 快速验证图像是否损坏
                        if img.width < 30 or img.height < 30:
                            raise Exception("Image too small")
                except Exception as e:
                    print("[ERROR] Image verification failed for item id:", item['id'], "Error:", e, "skipping this item ...")
                    continue
                json_items.append(item)
                if len(json_items) >= max_samples:
                    break
        
        
        print(f"load : [{len(json_items)}]")
        print(f"skip existed: [{len(existed_ids)}]")

        if json_items:
            yield json_items
        else:
            break

def batch_inf():
    setup_ray()

    infer_actors = [InferenceActor.remote(
        model_name= MODEL_NAME,
        actor_id= i
    ) for i in range(WORKER_NODE)]
    for infer_actor in infer_actors:
        infer_actor.set_response_file.remote(RESPONSE_FILE)
    
    for train_dataset in train_data_reader(MAX_SAMPLES):
        # train_dataset = train_data_reader()[:20_000]   ###jedi 
        print(f"[INFO] load train dataset: {len(train_dataset)} samples")
        train_chunks = np.array_split(train_dataset, WORKER_NODE)

        sampling_params = SamplingParams(
            n=VOTING_TIMES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )

        futures = []

        for chunk, infer_actor in zip(train_chunks, infer_actors):
            ####
            print(f"[DEBUG] chunk size: {len(chunk)}")
            future = infer_actor.infer.remote(
                samples= chunk,
                sampling_params= sampling_params,
                batch_size=BATCH_SIZE,
            )
            futures.append(future)
    
        # get results
        infer_results = []
        for future in futures:
            try:
                infer_results.append(ray.get(future))
            except Exception as e:
                print(f"ray get error: {e}")
                infer_results.append(None)
    return 

import os
if __name__ == "__main__":
    MAX_SAMPLES = 10_000

    BATCH_SIZE = 16
    VOTING_TIMES = 3
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_TOKENS = (1024*3) # 只用输出坐标
    RESPONSE_FILE = f"generate/0712cot-generate-without-example/grounding-R1-cot_click.jsonl"    
    os.makedirs(os.path.dirname(RESPONSE_FILE), exist_ok=True)

    MODEL_NAME = "/public/home/ljt/hqp/models/qwen2.5_vl_7b"
    # load WORKER_NODE from env
    WORKER_NODE = int(os.environ.get("WORKER_NODE", 1))
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 2408448))
    print(f"[INFO] WORKER_NODE: {WORKER_NODE} MAX_PIXELS: {MAX_PIXELS}")

    batch_inf()
####输入，输出文件夹，reader
