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
import argparse

from laser.utils.misc import generate_id, get_date, load_and_resize_image
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
Your task is to **think step by step** about which area in the screenshot is most relevant to the instruction.  
Then, output the result in the following format:

<think>your step-by-step reasoning here</think>  
<box>(x1,y1,x2,y2)</box>

The <think> section should explain why you focus on that area in relation to the instruction.  
The <box> should define a bounding box that covers the **focus region**: a sufficiently large area likely to contain the target interface element(s) for completing the instruction. It doesn't need to be pixel-precise, but it should tightly bound the visually relevant region.

- Make sure the box includes enough visual context for reasoning, typically greater 400x400 pixels in size.
- Center the box on the relevant region but include sufficient margin to avoid cropping important contextual cues (e.g., labels, icons, nearby buttons).
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
LIMIT_MM_PER_PROMPT={
    "image": 3
}
MAX_NUM_SEQS= 16
MAX_MODEL_LEN= (1024*12) # reduce it when OOM!

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

    def infer(self, samples: list, sampling_params: SamplingParams, batch_size: int= 16):

        # conversation -> get image -> replace with and height
        llm_inputs = []
        raw_images = []
        for sample in samples:
            # 注意设置好 min-pixex 和 max-pixel
            image_url = sample['image_url']
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
                assert len(response) == len(finish_reason) == VOTING_TIMES
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

                ### box、cot and cot_correct
                cot_list = []
                box_list = []
                cot_correct = []
                model_box_list = []

                grounded_bbox = item["coordinate"]


                for res in response:
                    think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
                    think_content = think_match.group(1) if think_match else None

                    if think_content == None:
                        continue

                    box_match = re.search(r"<box>(.*?)</box>", res, re.DOTALL)
                    box_content = box_match.group(1) if box_match else None
                    if box_content== None:
                        continue

                    box = list(map(float, re.findall(r"\d+\.?\d*", box_content)))  

                     ### 框超出屏幕   ##TODO: 修改框超出屏幕的逻辑， 有些框只是超出一点而已，但是他的cot是有用的，并且也能狂刀焦点区域(1)缩小到去掉超出屏幕的部分（2）也要排除过大的>75%
                    if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3] or box[2] > run_image.width or box[3] > run_image.height:  
                        continue

                    model_box = box.copy() 

                    box = [round(box[0] / run_image.width * raw_image.width, 4), round(box[1] / run_image.height * raw_image.height, 4), round(box[2] / run_image.width * raw_image.width, 4), round(box[3] / run_image.height * raw_image.height, 4)]

                    if len(box) != 4:
                        continue

                    cot_list.append(think_content)
                    box_list.append(box)
                    model_box_list.append(model_box)

                    if box[0] <= grounded_bbox[0] and box[1] <= grounded_bbox[1] and box[2] >= grounded_bbox[2] and box[3] >= grounded_bbox[3]:
                        cot_correct.append(True)
                    else:
                        cot_correct.append(False)

                ###应该把模型生成的框也放进去，方便处理
                item['running']['model_boxes'] = model_box_list
                item['running']['cots'] =  cot_list
                item['running']['boxes'] = box_list
                item['running']['cot_correct'] = cot_correct

                log_response_answers.append(item)
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
        FILTER_DATA_FILE = "/public/home/ljt/hqp/GUI-grounding/train_data/grounding-r1/grounding_r1_aria.jsonl"
        IMAGE_DIR = "/public/home/ljt/hqp/GUI-grounding/train_data/grounding-r1"
        print(f"[INFO] load from [{FILTER_DATA_FILE}]")
        with open(FILTER_DATA_FILE, 'r') as f:     ###showui 
            for line in f:
                item = json.loads(line)
                # if item['type'] == 'original': continue
                if item['id'] in existed_ids: continue
                if len(item['coordinate']) != 4: continue
                item['image_url'] = os.path.join(IMAGE_DIR ,item['image_url'])
                if not os.path.exists(item['image_url']):
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
    
    return None

import os
if __name__ == "__main__":
    MAX_SAMPLES = 10_000

    BATCH_SIZE = 16
    VOTING_TIMES = 5
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_TOKENS = (1024*3) # 只用输出坐标
    RESPONSE_FILE = f"generate/0720_generate_cot_focus_with_low_maxpixels/grounding-R1-cot_focus.jsonl"    ####showui
    os.makedirs(os.path.dirname(RESPONSE_FILE), exist_ok=True)

    MODEL_NAME = "/public/home/ljt/hqp/models/qwen2.5_vl_7b"
    # load WORKER_NODE from env
    WORKER_NODE = int(os.environ.get("WORKER_NODE", 1))
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 313600))
    print(f"[INFO] WORKER_NODE: {WORKER_NODE} MAX_PIXELS: {MAX_PIXELS}")

    batch_inf()
####输入，输出文件夹，reader
