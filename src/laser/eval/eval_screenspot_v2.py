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


# ref: screenspot-pro/models/qwen2_5vl.py
def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
############################################################
# set system prompt here
from pixel_grounding.build_prompt import build_sys_msg_crop_cot_click_no_cot, build_sys_msg_crop_cot_click_cot
from pixel_grounding.build_prompt import build_user_msg_1, bulid_user_msg_2, bulid_assistant_msg

print(f"[INFO] click_cot")
build_sys_msg = build_sys_msg_crop_cot_click_cot
SYS_MSG = build_sys_msg()


logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

###TODO：refusal 怎么处理？  目前当作bbox处理
def eval_point(box_type:str, point, bbox):
    def _is_point_in_rectangle(point, rect):
        return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

    def _is_point_in_polygon(point, polygon):
        x, y = point
        n = len(polygon) // 2
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i * 2], polygon[i * 2 + 1]
            xj, yj = polygon[j * 2], polygon[j * 2 + 1]

            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i

        return inside

    if box_type == "bbox" or box_type == "refusal":
        return _is_point_in_rectangle(point, bbox)
    elif box_type == "polygon":
        return _is_point_in_polygon(point, bbox)
    else:
        raise ValueError(f"[ERROR] box_type {box_type} not supported")



def eval_sample_positive_gt(sample, bbox):
    result_lst = []
    n_focus_correctness = []

    trajs = sample["trajs"]

    for traj in trajs:
        raw_point = traj["raw_point"]
        if raw_point == None:
            result_lst.append("wrong_format")
        else:
            if eval_point(sample["box_type"], raw_point, bbox):
                result_lst.append("correct")
            else:
                result_lst.append("wrong")

        ##TODO:polygan逻辑没写
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

# def eval_sample_positive_gt(sample, response):
#     bbox = sample["bbox"]
#     bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
#     # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
#     img_size = sample["img_size"]
#     bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
#     click_point = point  # may be none
#     # print(click_point)
#     # print(bbox)
#     # import pdb;pdb.set_trace()
#     if click_point is None or len(click_point) != 2:
#         return "wrong_format"
#     # Check if the predicted point falls in the ground truth box
#     if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
#         return "correct"
#     else:
#         return "wrong"
    
def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
    }
    return metrics


def evaluate_all_cla(result_report, results):
    """Calculates the metrics for a simple result list."""
    def collect_results_to_eval(results, cla):
        filtered_results = []
        for sample in results:
            if cla.lower() in sample["class"].lower():
                filtered_results.append(sample)
        return filtered_results


    # cla_lst = ["text_matching", "element_recognition", "layout_understanding", "fine_grained_manipulation", "refusal"]
    cla_lst = ['Mobile,Text','Mobile,Icon','Desktop,Text','Desktop,Icon','Web,Text','Web,Icon']
    result_dict = dict()
    for cla in cla_lst:
        filtered_results = collect_results_to_eval(results, cla)
        result_dict[cla] = calc_metric_for_result_list(filtered_results)
    for cla, result in result_dict.items():
        result_report["metrics"][cla] = result
    return result_report

def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }
    result_report["metrics"]["overall"] = calc_metric_for_result_list(results)
    ### eval_cla
    evaluate_all_cla(result_report, results)
    # Save detailed results
    result_report["details"] = results
    return result_report


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
MAX_MODEL_LEN= 8192*4 # reduce it when OOM!

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
            limit_mm_per_prompt={"image": 5},
            tensor_parallel_size= TENSOR_PARALLEL_SIZE,
            max_model_len= MAX_MODEL_LEN, # max_model_len should be ????
            max_num_batched_tokens= MAX_MODEL_LEN
            # mm_processor_kwargs={   ###这个会让显存炸掉
            #     "min_pixels": 28 * 28,
            #     "max_pixels": self.max_pixels,
            # },
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
                        # crop_box_base_raw = extend_box(crop_box_base_raw)
                        crop_box_base_raw = smart_extend_box(box= crop_box_base_raw, image= current_traj["images"][0])
                        # assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] <= current_traj["images"][0].size[0] and crop_box_base_raw[2] >= 0 and crop_box_base_raw[3] <= current_traj["images"][0].size[1]
                        assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] >= 0 and crop_box_base_raw[2] <= current_traj["images"][0].size[0] and crop_box_base_raw[3] <= current_traj["images"][0].size[1], f"error crop box, failed by assert. crop_box_base_raw(after extend): [{crop_box_base_raw}] but raw_image_size: [{current_traj['images'][0].size}]"
                    # crop image
                    debugg= 7
                    crop_image = crop_image_by_box(current_traj["images"][0], crop_box_base_raw)
                    resize_crop_image = resize_image(
                        image = crop_image, 
                        factor= PROCESSOR.image_processor.patch_size * PROCESSOR.image_processor.merge_size,
                        # min_pixels= PROCESSOR.image_processor.min_pixels,
                        min_pixels= MIN_PIXELS,
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

    def infer(self, samples: list, sampling_params: SamplingParams, batch_size: int= 16, max_roll: int= 5):
        
        sample_id2sample = {}
        for sample in samples:
            sample_id2sample[sample['id']] = sample

        results_list = []
        for batch_start in range(0, len(samples), batch_size):
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
                results_list.append(record_item)
        
        return results_list


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--worker_node', type=int, required=True)
    # 添加一个 max-pixels 参数
    parser.add_argument('--max_pixels', type=int, required=True, default=2408448, help="Maximum number of pixels for resizing images.")
    parser.add_argument('--min_pixels', type=int, required=True, default=1*(1024*28*28), help="Minimum number of pixels for resizing images.")
    
    parser.add_argument('--n_times', type=int, required=True)
    args = parser.parse_args()
    return args


### TODO: KILL 掉会有图片没有smart save ，在parallel 中解决
def smart_size_all_raw(tasks_to_run:list[dict], max_pixels:int):
    print("smart size all raw images")
    all_smart_image_dir = "cache/Screenspot-v2_resize_image"
    os.makedirs(all_smart_image_dir, exist_ok=True)
    subdirs = set([name for name in os.listdir(all_smart_image_dir) if os.path.isdir(os.path.join(all_smart_image_dir, name))])
    resize_image_dir = os.path.join(all_smart_image_dir, str(max_pixels))
    os.makedirs(resize_image_dir, exist_ok=True)

    if str(max_pixels) in subdirs:
        print(f"{max_pixels}max pixels smart size images already exist in")
        print(subdirs)
    else:
        print(f"{max_pixels} no no")
        for task in tasks_to_run:
            raw_img_filename = task["img_filename"]

            assert os.path.exists(raw_img_filename)

            file_exs = "_".join(raw_img_filename.split("/")[-2:])
            resize_image_url = os.path.join(resize_image_dir, file_exs)
            raw_image, resize_image = load_and_resize_image(
                image_url= raw_img_filename,
                factor= PROCESSOR.image_processor.patch_size * PROCESSOR.image_processor.merge_size,
                min_pixels= PROCESSOR.image_processor.min_pixels,
                max_pixels=max_pixels,
            )
            resize_image.save(resize_image_url)

    output_lst = []
    for item in tasks_to_run:
        raw_img_filename = item["img_filename"]
        file_exs = "_".join(raw_img_filename.split("/")[-2:])
        resize_image_url = os.path.join(resize_image_dir, file_exs)
        assert os.path.exists(resize_image_url)
        output = copy.deepcopy(item)
        output["resize_image_url"] = resize_image_url
        output_lst.append(output)
    return output_lst

def eval_data_reader() -> list[dict]:
    print("loading data!")
    tasks_to_run = []
    with open(ANNOTATION_FILE, "r") as fin:
        dataset = json.load(fin)
        for task in dataset:
            task["prompt_to_evaluate"] = task["instruction"]
            # filename = task["image_path"]
            # task["img_filename"] = os.path.join(IMGS_FOLDER, filename)
            task['img_filename'] = task['image']
            # bbox = task["box_coordinates"]
            # task["bbox"] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            task["task_filename"] = task["instruction"]
            tasks_to_run.append(task)
    print(f"Total tasks: {len(tasks_to_run)}")
    return tasks_to_run

def batch_inf(args):
    setup_ray()  ##debug
    tasks_to_run = eval_data_reader()
    tasks_to_run = smart_size_all_raw(tasks_to_run, MAX_PIXELS)
    print("len of tasks_to_run: ", len(tasks_to_run))
    # import pdb; pdb.set_trace()

    eval_chunks = np.array_split(tasks_to_run, WORKER_NODE)
    # import pdb; pdb.set_trace()

    infer_actors = [InferenceActor.remote(
        model_name= MODEL_NAME,
        actor_id= i
    ) for i in range(WORKER_NODE)]  

    ##TODO: 这里的温度和top可能需要调一下

    sampling_params = SamplingParams(
        n= 1,
        temperature= TEMPERATURE,
        top_p= TOP_P,
        max_tokens= MAX_TOKENS,
    )

    futures = []

    for chunk, infer_actor in zip(eval_chunks, infer_actors):

        # chunk = chunk[0:20]   #dubug

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
            infer_results.extend(ray.get(future))
        except Exception as e:  ##可能得改
            print(f"ray get error: {e}")
            infer_results.append(None)

    if None in infer_results:
        raise RuntimeError(f"Some infer actors are None, Skip this sub-bench")

    ###cal result
    ### ref: GUI-grounding/ScreenSpot-Pro-GUI-Grounding/eval_screenspot_pro.py
    results = []
    for sample in infer_results:
        sample_result = copy.deepcopy(sample)
        # import pdb; pdb.set_trace()
        n_focus_correctness, tryn_results, correctness = eval_sample_positive_gt(sample, sample["bbox"])
        sample_result.update({
            "n_focus_correctness": n_focus_correctness,
            "n_correctness": tryn_results,
            "correctness": correctness,
        })
        results.append(sample_result)
        
    result_report = evaluate(results)
    # Save to file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpot finished.")

if __name__ == "__main__":
    args = parse_args()

    N_TIMES = args.n_times
    if N_TIMES > 1:
        TEMPERATURE = 0.7
        TOP_P = 0.8
    else:
        TEMPERATURE = 0.0
        TOP_P = 1.0


    MODEL_NAME = args.model_name_or_path
    WORKER_NODE = args.worker_node
    BATCH_SIZE = 16 #### debug
    print("[debug]: batch size: ", BATCH_SIZE)

    PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)

    MAX_PIXELS = args.max_pixels # 这个要和训练保持一致
    MIN_PIXELS = args.min_pixels
    ANNOTATION_FILE = args.annotation_file
    IMGS_FOLDER = args.img_folder
    
    assert MAX_PIXELS > MIN_PIXELS
    MAX_NUM_SEQS = 16
    print(f"[INFO] MAX_PIXELS: {MAX_PIXELS}. MIN_PIXELS: {MIN_PIXELS}")
    MAX_TOKENS = 1024 # 1024应该够了

    batch_inf(args) 