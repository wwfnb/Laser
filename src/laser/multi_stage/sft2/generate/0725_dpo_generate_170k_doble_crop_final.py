import ray
import numpy as np
from vllm import SamplingParams, LLM
import atexit
import copy
import itertools
import torch
import json
import re
import argparse
import os
import filelock
from PIL import Image
import logging
from tqdm import tqdm
import random
from io import BytesIO
import base64
from transformers import AutoProcessor  
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from pixel_grounding.misc import generate_id, get_date, load_and_resize_image, crop_image_by_box, resize_image, extend_box, smart_extend_box
from pixel_grounding.build_prompt import build_sys_msg_crop_cot_click_cot, bulid_assistant_msg, build_user_msg_1, bulid_user_msg_2

# ref: screenspot-pro/models/qwen2_5vl.py
build_sys_msg= build_sys_msg_crop_cot_click_cot

SYS_MSG = build_sys_msg()


logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']




# def eval_sample_positive_gt(raw_box_lst, raw_point_lst, bbox):
#     result_lst = []
#     n_focus_correctness = []
#     for raw_point in raw_point_lst:
#         if raw_point == None:
#             result_lst.append("wrong_format")
#         elif (bbox[0] <= raw_point[0] <= bbox[2]) and (bbox[1] <= raw_point[1] <= bbox[3]):
#             result_lst.append("correct")
#         else:
#             result_lst.append("wrong")
#     if "correct" in result_lst:
#         correctness = "correct"
#     else:
#         if "wrong" in result_lst:
#             correctness = "wrong"
#         else:
#             correctness = "wrong_format"
#     for pre_box in raw_box_lst:
#         if pre_box == None:
#             n_focus_correctness.append("wrong_format")
#         elif pre_box[0] <= bbox[0] and pre_box[1] <= bbox[1] and pre_box[2] >= bbox[2] and pre_box[3] >= bbox[3]:
#             n_focus_correctness.append("correct")
#         else:
#             n_focus_correctness.append("wrong")
    
#     return n_focus_correctness, result_lst, correctness

def eval_sample_positive_gt(trajs, bbox):
    result_lst = []
    n_focus_correctness = []

    for traj in trajs:
        raw_point = traj["raw_point"]
        if raw_point == None:
            result_lst.append("wrong_format")
        elif (bbox[0] <= raw_point[0] <= bbox[2]) and (bbox[1] <= raw_point[1] <= bbox[3]):
            result_lst.append("correct")
        else:
            result_lst.append("wrong")

        focus_lst = []
        raw_box_lst = traj["raw_box"]
        for pre_box in raw_box_lst:
            if pre_box == None:
                focus_lst.append("wrong_format")
            elif pre_box[0] <= bbox[0] and pre_box[1] <= bbox[1] and pre_box[2] >= bbox[2] and pre_box[3] >= bbox[3]:
                focus_lst.append("correct")
            else:
                focus_lst.append("wrong")
        n_focus_correctness.append(focus_lst)

    if "correct" in result_lst:
        correctness = "correct"
    else:
        if "wrong" in result_lst:
            correctness = "wrong"
        else:
            correctness = "wrong_format"
    return n_focus_correctness, result_lst, correctness


def setup_ray():
    if not ray.is_initialized():
        print(f"ray is not initialized, initializing...")
        ray.init(
        num_cpus=16,
        num_gpus=WORKER_NODE,
        address="local"      # 禁止自动连接外部
        )
    def cleanup_ray():
        if ray.is_initialized():
            ray.shutdown()
            print("closing ray...")
    atexit.register(cleanup_ray)

####TODO: 需要设置参数
# from pixel_grounding.misc import TENSOR_PARALLEL_SIZE
# from pixel_grounding.models.qwen2_5vl import CustomQwen2_5VL_VLLM_Model
# LIMIT_MM_PER_PROMPT={
#     "image": 3
# }
# MAX_NUM_SEQS= 16
MAX_MODEL_LEN= 8192*2 # reduce it when OOM!

from pixel_grounding.misc import TENSOR_PARALLEL_SIZE  ##根据显存控制并行卡的数量

@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class InferenceActor:
    def __init__(self, model_name, actor_id:int):
        self.model_name = model_name
        self.actor_id = actor_id
        self.max_pixels = 99999999
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # processor llm
        # self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"#####model_name: [{self.model_name}]#########")
        print(f"#####TENSOR_PARALLEL_SIZE: [{TENSOR_PARALLEL_SIZE}]#########")
        print(f"#####MAX_MODEL_LEN: [{MAX_MODEL_LEN}]#########")

        self.llm = LLM(
            model=self.model_name, 
            dtype="auto", 
            max_num_seqs=MAX_NUM_SEQS,  ###并行推理数量
            limit_mm_per_prompt={"image": 2},
            tensor_parallel_size= TENSOR_PARALLEL_SIZE,
            max_model_len= MAX_MODEL_LEN, # max_model_len should be ????
            max_num_batched_tokens= MAX_MODEL_LEN
            # mm_processor_kwargs={   ###这个会让显存炸掉
            #     "min_pixels": 28 * 28,
            #     "max_pixels": self.max_pixels,
            # },
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


    def trans_msg_to_inputs(self, batch_history_msg: list, batch_resize_img:list):
        batch_input = []
        for h_msg, resize_img_lst in zip(batch_history_msg, batch_resize_img):
            # print("len_hmsg", len(h_msg))
            # print("len resize_img_lst", len(resize_img_lst))
            assert len(resize_img_lst)*2 == len(h_msg)

            prompt = self.processor.apply_chat_template(
                h_msg, tokenize=False, add_generation_prompt=True
            )
            # print("prompt:", prompt)
            inputs = {
                "prompt": prompt, 
                "multi_modal_data": {"image": resize_img_lst}
            }
            batch_input.append(inputs)
        return batch_input
    
    ####TODO: 这个解析可能需要改， 因为crop这个输出的box很多长度非法， 看一下最终的response来改（目前发挥你高度）
    def parser_res(self, res: str):
        # print("paser_res.........")
        def get_para(action_str: str):
            box_match = re.findall(r"\d+(?:\.\d+)?", action_str)
            box = [float(x) for x in box_match]
            return box

        res = res.strip()
        think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        else:
            think= None

        if not res.endswith("</action>"):
            res += "</action>"
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", res, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
            if "crop" in action_text:
                action = "crop"
                para = get_para(action_text)
            elif "click" in action_text:
                action = "click"
                para = get_para(action_text)
            else:
                action = None
                para = None
        else:
            action = None
            para = None
        # print(f"action: {action}, para: {para}")
        return action, para
    
    def trans_box_to_box(self, box, src_image, ob_image):
        return [box[0] / src_image.size[0] * ob_image.size[0], 
                box[1] / src_image.size[1] * ob_image.size[1], 
                box[2] / src_image.size[0] * ob_image.size[0], 
                box[3] / src_image.size[1] * ob_image.size[1]]
    
    def trans_point_to_point(self, point, src_image, ob_image):
        return [point[0] / src_image.size[0] * ob_image.size[0], 
                point[1] / src_image.size[1] * ob_image.size[1]]

    
    def process_responses(self, sample_ids, responses, sample_id2traj):
        for (sample_id, response) in zip(sample_ids, responses):
            action, para = self.parser_res(response[0])
            current_traj = sample_id2traj[sample_id]   ### debug
            debugg = 0
            if not(action is not None and para is not None and para != []):
                continue
            try:
                if action == "click":
                    model_point = para[:2]
                    # crop_point = self.trans_point_to_point(model_point, item["resize_crop_image"], item["crop_image"])
                    debugg = 1
                    click_point_on_crop = self.trans_point_to_point(model_point, current_traj["resize_images"][-1], current_traj["images"][-1])
                    debugg= 2
                    if current_traj['raw_box']:
                        pre_raw_box = current_traj['raw_box'][-1]
                        click_point_base_raw = [click_point_on_crop[0] + pre_raw_box[0], click_point_on_crop[1] + pre_raw_box[1]]
                    else:
                        # 直接完成的点击操作
                        click_point_base_raw = click_point_on_crop
                    # finish this traj
                    debugg= 3
                    current_traj['raw_response'].append(response[0])
                    debugg= 4
                    current_traj['raw_point'] = click_point_base_raw
                    debugg= 5
                    current_traj['is_end'] = True
                elif action == "crop":
                    print("")
                    assert current_traj['resize_images'] and current_traj['images']
                    debugg= 6
                    model_box = para[:4]
                    crop_box_on_crop = self.trans_box_to_box(model_box,  current_traj["resize_images"][-1], current_traj["images"][-1])
                    if current_traj['raw_box']:
                        pre_raw_box = current_traj['raw_box'][-1]
                        crop_box_base_raw = [crop_box_on_crop[0] + pre_raw_box[0], crop_box_on_crop[1] + pre_raw_box[1], crop_box_on_crop[2] + pre_raw_box[0], crop_box_on_crop[3] + pre_raw_box[1]]
                    else:
                        crop_box_base_raw = crop_box_on_crop
                    # extend crop box if it is too small !
                    if (crop_box_base_raw[3] - crop_box_base_raw[1] <= 28 or crop_box_base_raw[2] - crop_box_base_raw[0] <= 28):
                        crop_box_base_raw = smart_extend_box(box= crop_box_base_raw, image= current_traj["images"][0])
                        # assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] <= current_traj["images"][0].size[0] and crop_box_base_raw[2] >= 0 and crop_box_base_raw[3] <= current_traj["images"][0].size[1], f"error crop box, failed by assert."
                        assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] >= 0 and crop_box_base_raw[2] <= current_traj["images"][0].size[0] and crop_box_base_raw[3] <= current_traj["images"][0].size[1], f"error crop box, failed by assert. crop_box_base_raw(after extend): [{crop_box_base_raw}] but raw_image_size: [{current_traj['images'][0].size}]"
                    # crop image
                    debugg= 7
                    crop_image = crop_image_by_box(current_traj["images"][0], crop_box_base_raw)
                    resize_crop_image = resize_image(
                        image = crop_image, 
                        factor= PROCESSOR.image_processor.patch_size * PROCESSOR.image_processor.merge_size,
                        min_pixels= PROCESSOR.image_processor.min_pixels,
                        max_pixels=MAX_PIXELS,                 
                    )
                    assistant_msg = bulid_assistant_msg(response[0])
                    user_msg_2 = bulid_user_msg_2(resize_crop_image)
                    
                    current_traj['raw_box'].append(crop_box_base_raw)
                    current_traj['raw_response'].append(response[0])
                    # add crop images
                    current_traj['images'].append(crop_image)
                    current_traj['resize_images'].append(resize_crop_image)
                    # add new messages
                    current_traj['history_msgs'].append(assistant_msg)
                    current_traj['history_msgs'].append(user_msg_2)
                    # just one step ; set is_end to True
                    # current_traj['is_end'] = True

            except Exception as e:
                print(f"Error processing image [{sample_id}] with response [{response}]: {e}")
                print(debugg)
                continue

    def open_image(self, image_url: str):
        if isinstance(image_url, str):
            assert os.path.exists(image_url) and os.path.isfile(image_url), "Invalid input image path."
            open_image = Image.open(image_url).convert('RGB')
        assert isinstance(open_image, Image.Image), "Invalid input image."
        return open_image

    def initial_batch(self, samples: list) -> dict:
        sample_id2traj = {}
        for sample in samples:
            instruction = sample['prompt_to_evaluate']
            raw_image = self.open_image(sample["img_filename"])
            resize_raw_image = self.open_image(sample["resize_image_url"])
            
            usr_msg_1 = build_user_msg_1(instruction, resize_raw_image)
            history_msgs = [SYS_MSG, usr_msg_1]
            # dict to record traj
            initial_traj = {
                "sample_id": sample['id'],
                "raw_response": [],
                "raw_box": [],
                "images": [raw_image],
                "resize_images": [resize_raw_image],
                "raw_point": None,
                # pop when record
                "history_msgs": history_msgs,
                "is_end": False,
            }
            sample_id2traj[sample['id']] = initial_traj
        return sample_id2traj
    
    def not_end_trajs(self, sample_id2traj: dict):
        sample_ids = []
        batch_history_msgs = []
        batch_history_imgs = []
        for sample_id, traj in sample_id2traj.items():
            if traj['is_end']: continue
            sample_ids.append(sample_id)
            batch_history_msgs.append(traj['history_msgs'])
            batch_history_imgs.append(traj['resize_images'])
        return sample_ids, batch_history_msgs, batch_history_imgs

    def eval_trajs(self, record_item):
        new_trajs = copy.deepcopy(record_item["trajs"])
        old_traj_crop_box = record_item["old_traj"]["raw_box"][0]
        ####转换成原图的raw_box
        for t in new_trajs:
            raw_box = t["raw_box"][0]
            t["raw_box"] = [[raw_box[0] + old_traj_crop_box[0], raw_box[1] + old_traj_crop_box[1],  raw_box[2] + old_traj_crop_box[0], raw_box[3] + old_traj_crop_box[1]]]
            raw_point = t["raw_point"]
            t["raw_point"] = [raw_point[0] + old_traj_crop_box[0], raw_point[1] + old_traj_crop_box[1]]
            n_focus_correctness, result_lst, correctness = eval_sample_positive_gt([t], record_item["bbox"])

            if len(n_focus_correctness[0]) == 0:
                t["n_focus_correctness"] = "wrong_format"
            else:
                t["n_focus_correctness"] = n_focus_correctness[0][0]
            t["correctness"] =  correctness
        return new_trajs


    def infer(self, samples: list, sampling_params: SamplingParams, batch_size: int= 16, max_roll: int= 5):

        
        sample_id2sample = {}
        for sample in samples:
            sample_id2sample[sample['id']] = sample

        # results_list = []
        for batch_start in range(0, len(samples), batch_size):
            try:
                log_items = []
                batch_samples = samples[batch_start:batch_start + batch_size]
                sample_id2trajs = {}
                for sample in batch_samples:
                    sample_id2trajs.setdefault(sample['id'], [])
                # traj: raw_response: list, raw_box: list, row_point: Point, Image_size: list[Size], resize_Image_size: list[Size]
                for sampling_idx in range(N_TIMES):
                    # {sampling_idx} Time: infer this batch
                    sample_id2traj = self.initial_batch(samples= batch_samples)
                    for step in range(max_roll):
                        sample_ids, batch_history_msgs, batch_history_imgs = self.not_end_trajs(sample_id2traj)
                        if not sample_ids:
                            break
                        batch_inputs = self.trans_msg_to_inputs(batch_history_msgs, batch_history_imgs)
                        batch_outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
                        responses = [[o.text for o in output.outputs] for output in batch_outputs]
                        self.process_responses(sample_ids, responses, sample_id2traj)
                        assert len(responses) == len(sample_ids) == len(batch_history_msgs) == len(batch_history_imgs)
                    for sample in batch_samples:
                        sample_id = sample['id']
                        traj = sample_id2traj[sample_id]
                        # clean traj
                        if "images" in traj: traj.pop("images")
                        if "resize_images" in traj: traj.pop("resize_images")
                        if "history_msgs" in traj: traj.pop("history_msgs")
                        sample_id2trajs[sample_id].append(traj)
                # record this batch
                for sample_id, trajs in sample_id2trajs.items():
                    record_item = copy.deepcopy(sample_id2sample[sample_id])
                    record_item['trajs'] = trajs
                    # record_item['new_trajs_base_raw'] = self.eval_trajs(record_item)
                    log_items.append(record_item)
                self.write_file(log_items)
            except Exception as e:
                pass
        return None
    


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--worker_node', type=int, required=True)
    # 添加一个 max-pixels 参数
    parser.add_argument('--max_pixels', type=int, required=True, default=2408448, help="Maximum number of pixels for resizing images.")
    parser.add_argument('--n_times', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--response_file', type=str, required=True)
    parser.add_argument('--all_smart_image_dir', type=str, required=True)
    args = parser.parse_args()
    return args


####得debug一下[debug]   , 最终要的是要保持图像的dir 和 后缀一致，不然就容易bug
def smart_size_all_raw(tasks_to_run:list[dict], max_pixels:int):

    ###图像用id命名
    print("[INFO] smart size all raw images")
    resize_image_dir = os.path.join(ALL_SMART_IMAGE_DIR, str(max_pixels))
    os.makedirs(resize_image_dir, exist_ok=True)
    print(f"[INFO] resize_image_dir: {resize_image_dir}")

    for item in tasks_to_run:
        # all item should set these.
        crop_image_url = os.path.join(resize_image_dir, f"{item['id']}_raw.png")
        item["img_filename"] = crop_image_url
        item["img_path"] = crop_image_url
        item["resize_image_url"] = os.path.join(resize_image_dir, f"{item['id']}_resize.png")

    ###并行保存没有smartsize的图片
    smart_size_tasks = []
    for item in tasks_to_run:
        if not os.path.exists(item["resize_image_url"]) or not os.path.exists(item["img_filename"]):
            # generate resize tasks
            resize_task = {
                "raw_image_url": item['raw_image_url'],
                "crop_box_base_raw": item["old_traj"]["raw_box"][0],
                "crop_image_save_url": item["img_filename"],
                "resize_crop_image_save_url": item["resize_image_url"]
            }
            smart_size_tasks.append(resize_task)
    parallel_smart_size(smart_size_tasks, MAX_PIXELS, max_workers=40)
    
    new_tasks_to_run = []
    for item in tasks_to_run:
        if not os.path.exists(item["resize_image_url"]) or not os.path.exists(item["img_filename"]):
            continue
        new_tasks_to_run.append(item)
    return new_tasks_to_run

###修改成crop图像后resized,, crop_image 也要保存， 因为infer 代码基于crop后的
def smart_size_trunk(chunk:list[dict], max_pixels:int):
    for resize_task in chunk:
        # load raw
        raw_image = Image.open(resize_task['raw_image_url']).convert('RGB')
        # get crop image
        crop_image = crop_image_by_box(raw_image, resize_task["crop_box_base_raw"])
        crop_image.save(resize_task['crop_image_save_url'])
        # get resize crop image
        resize_crop_image = resize_image(
            image = crop_image,
            factor= PROCESSOR.image_processor.patch_size * PROCESSOR.image_processor.merge_size,
            min_pixels= PROCESSOR.image_processor.min_pixels,
            max_pixels=max_pixels,
        )
        resize_crop_image.save(resize_task['resize_crop_image_save_url'])
        # done

def parallel_smart_size(tasks_to_run:list[dict], max_pixels:int, max_workers:int=30):
    if len(tasks_to_run) == 0:
        print("No tasks to resize")
        return
    chunk_size = ceil(len(tasks_to_run) / max_workers)
    chunks = [
        tasks_to_run[i:i+chunk_size]
        for i in range(0, len(tasks_to_run), chunk_size)
    ]
    # result = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(smart_size_trunk, chunk, max_pixels) for chunk in chunks]
        for future in futures:
            future.result()
    return None



####保存old resize_image_url 在old_trajs里面
def train_data_reader(max_samples: int= 20_000):
    while True:
        existed_ids = set()
        if os.path.exists(RESPONSE_FILE):
            existed_ids = set([json.loads(line)['id'] for line in open(RESPONSE_FILE)])
        
        json_items = []
        # show-ui desktop
        print(f"[INFO] load from [{DATA_PATH}]")
        with open(DATA_PATH, 'r') as f:     
            for line in f:
                item = json.loads(line)
                if item['id'] in existed_ids: continue
                if not os.path.exists(item["img_path"]): continue 
                ###crop 后的图像小于28 * 28， 跳过
                item["raw_image_url"] = item["img_path"]
                old_traj = item["old_traj"]
                old_traj["resize_image_url"] = item.pop("resize_image_url")
                raw_box = old_traj["raw_box"][0]
                if (raw_box[2] - raw_box[0]) < 50 or (raw_box[3] - raw_box[1]) < 50:
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
        train_dataset = smart_size_all_raw(train_dataset, MAX_PIXELS) # bug ?
        # train_dataset = train_data_reader()[:20_000]   ###jedi 
        print(f"[INFO] load train dataset: {len(train_dataset)} samples")
        train_chunks = np.array_split(train_dataset, WORKER_NODE)

        sampling_params = SamplingParams(
            n=1,
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


if __name__ == "__main__":
    args = parse_args()

    MAX_SAMPLES = 5_000
    # VOTING_TIMES = 5
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_TOKENS = (1024*4)

    RESPONSE_FILE = args.response_file
    os.makedirs(os.path.dirname(RESPONSE_FILE), exist_ok=True)

    ALL_SMART_IMAGE_DIR= args.all_smart_image_dir
    # os.makedirs(os.path.dirname(ALL_SMART_IMAGE_DIR), exist_ok=True)

    DATA_PATH = args.data_path
    N_TIMES = args.n_times
    MODEL_NAME = args.model_name_or_path
    WORKER_NODE = args.worker_node
    BATCH_SIZE = 16 
    print("[debug]: batch size: ", BATCH_SIZE)

    PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)

    MAX_PIXELS = args.max_pixels # 这个要和训练保持一致
    MAX_NUM_SEQS = 8
    print(f"[INFO] MAX_PIXELS: {MAX_PIXELS}")
    print(f"[INFO] MODEL_NAME: {MODEL_NAME}")

    batch_inf() 