####TODO: 1、extend_box 的逻辑,   2、源数据预处理
from vllm import SamplingParams, LLM
from tqdm import tqdm
from laser.utils.misc import load_and_resize_image, get_visible_gpus, setup_ray, insertion_area, adjust_box, box_area, transfer_box, box_contains, box_valid
from collections import Counter
from PIL import Image
import os
import yaml
import json
import ray
import numpy as np
from transformers import AutoProcessor  
import re
import copy
from math import ceil
from laser.actor.single_step_dpo_actor import SingleStepDpoActor
from laser.processor.single_step_dpo_processor import SingleStepDpoProcessor
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging      # 用于日志记录
import torch        # 用于深度学习及设置随机种子

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

class SingleStepDpoGenerator():

    def __init__(self):
        with open("src/laser/configs/single_step_dpo.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)

        ###input
        self.model_name_or_path = self.cfg["input"]["model_name_or_path"]
        self.source_dataset_path = self.cfg["input"]["source_dataset_path"]

        ###output_data_dir
        self.synthetic_data_dir = self.cfg["output_data_dir"]["synthetic_data_dir"]
        self.train_data_dir = self.cfg["output_data_dir"]["train_data_dir"]
        
        ###geneartion
        self.batch_size = self.cfg["generation"]["batch_size"]
        self.max_samples = self.cfg["generation"]["max_samples"]
        self.max_pixels = self.cfg["generation"]["max_pixels"]
        self.max_tokens = self.cfg["generation"]["max_tokens"]
        self.temperature = self.cfg["generation"]["temperature"]
        self.top_p = self.cfg["generation"]["top_p"]
        self.smart_image_dir = self.cfg["generation"]["smart_image_dir"]
        self.passN= self.cfg["generation"]["passN"]
        self.max_roll= self.cfg["generation"]["max_roll"]
        self.max_num = self.cfg["generation"]["max_num"]

        self.worker_node = len(get_visible_gpus())

        self.single_step_dpo_processor = SingleStepDpoProcessor()

        self.PROCESSOR = AutoProcessor.from_pretrained(self.model_name_or_path)

        
    def smart_size_all_raw(self, tasks_to_run:list[dict], max_pixels:int):

        ###图像用id命名
        print("smart size all raw images")
        resize_image_dir = os.path.join(self.smart_image_dir, str(self.max_pixels))
        os.makedirs(resize_image_dir, exist_ok=True)

        existed_imgs = (os.listdir(resize_image_dir))

        ###并行保存没有smartsize的图片
        smart_size_tasks = []
        for item in tasks_to_run:
            file_exs = item["id"] + ".png"
            if file_exs not in existed_imgs:
                item["resize_image_url"] = os.path.join(resize_image_dir, file_exs)
                smart_size_tasks.append(item)
        self.parallel_smart_size(smart_size_tasks, self.max_pixels, max_workers=40)
        

        ###给item加上resize_image_url
        output_lst = []
        for item in tasks_to_run:
            file_exs = item["id"] + ".png"
            resize_image_url = os.path.join(resize_image_dir, file_exs)
            assert os.path.exists(resize_image_url)
            item["resize_image_url"] = resize_image_url
            output_lst.append(item)
        return output_lst

    def smart_size_trunk(self, chunk:list[dict], max_pixels:int):
        for data in chunk:
            img_path = data["img_path"]
            raw_image, resize_image = load_and_resize_image(
                image_url= img_path,
                factor= self.PROCESSOR.image_processor.patch_size * self.PROCESSOR.image_processor.merge_size,
                min_pixels= self.PROCESSOR.image_processor.min_pixels,
                max_pixels=self.max_pixels,
            )
            resize_image.save(data["resize_image_url"])


    def parallel_smart_size(self, tasks_to_run:list[dict], max_pixels:int, max_workers:int=30):
        if len(tasks_to_run) == 0:
            print("No tasks to resize")
            return
        chunk_size = ceil(len(tasks_to_run) / max_workers)
        chunks = [
            tasks_to_run[i:i+chunk_size]
            for i in range(0, len(tasks_to_run), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.smart_size_trunk, chunk, self.max_pixels) for chunk in chunks]
            for future in futures:
                future.result()
        return None

    def crop_candidates_data_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break
            
            json_items = []
            print(f"[INFO] load from [{input_data_path}]")
            with open(input_data_path, 'r') as f:     ###showui 
                for line in f:
                    item = json.loads(line)
                    if item['id'] in existed_ids: continue
                    item['prompt_to_evaluate'] = item['instruction']
                    json_items.append(item)
                    if len(json_items) >= max_samples:
                        break
            
            print(f"load : [{len(json_items)}]")
            print(f"skip existed: [{len(existed_ids)}]")

            if json_items:
                yield json_items
            else:
                break

    def score_candidates_data_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break
            
            json_items = []
            print(f"[INFO] load from [{input_data_path}]")
            with open(input_data_path, 'r') as f:    
                for line in f:
                    item = json.loads(line)
                    if item['id'] in existed_ids: continue
                    json_items.append(item)
                    if len(json_items) >= max_samples:
                        break
            
            print(f"load : [{len(json_items)}]")
            print(f"skip existed: [{len(existed_ids)}]")

            if json_items:
                yield json_items
            else:
                break

    def batch_inf(self, stage: str, initial_stage: str , input_data_path:str, output_data_path: str):
        setup_ray(self.worker_node)

        infer_actors = [SingleStepDpoActor.remote(
            model_name= self.model_name_or_path,
            actor_id= i
        ) for i in range(self.worker_node)]
        for infer_actor in infer_actors:
            infer_actor.set_response_file.remote(output_data_path)

        if stage == "crop_candidates":
            train_data_reader = self.crop_candidates_data_reader
        elif stage == "score_candidates":
            train_data_reader = self.score_candidates_data_reader
        else:
            raise Exception("[ERROR] stage not support")
        
        for train_dataset in train_data_reader(input_data_path, output_data_path, self.max_samples):
            tasks_to_run = self.smart_size_all_raw(train_dataset, self.max_pixels)
            print(f"[INFO] load train dataset: {len(tasks_to_run)} samples")

            train_chunks = np.array_split(tasks_to_run, self.worker_node)

            sampling_params = SamplingParams(
                n=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            futures = []

            for chunk, infer_actor in zip(train_chunks, infer_actors):
                print(f"[DEBUG] chunk size: {len(chunk)}")
                if stage == "crop_candidates":
                    future = infer_actor.crop_candidates_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
                        continue_or_not=False,
                        batch_size=self.batch_size,
                        max_pixels=self.max_pixels,
                        passN= self.passN,
                        max_roll= self.max_roll,
                    )
                    futures.append(future)
                elif stage == "score_candidates": 
                    future = infer_actor.score_candidates_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
                        initial_stage= initial_stage,
                        continue_or_not=True,
                        batch_size=self.batch_size,
                        max_pixels=self.max_pixels,
                        passN= self.passN,
                        max_roll= self.max_roll,
                    )
                    futures.append(future)
                else:
                    raise Exception("[ERROR] stage not support")
        
            # get results
            infer_results = []
            for future in futures:
                try:
                    infer_results.append(ray.get(future))
                except Exception as e:
                    print(f"ray get error: {e}")
                    infer_results.append(None)

        if ray.is_initialized():
            ray.shutdown()
        return None

    # def set_setting(self, stage: str):
    #     pass

    def crop_candidates_generate(self, input_data_path:str, output_data_path:str):
        self.batch_inf("crop_candidates", "", input_data_path, output_data_path)

    def score_candidates_generate(self, initial_stage: str, input_data_path: str, output_data_path:str):
        self.batch_inf("score_candidates", initial_stage, input_data_path, output_data_path)

    def pre_process(self, source_dataset_path: str):
        new_filename = source_dataset_path.replace(".jsonl", "_transfer.jsonl")
        with open(source_dataset_path, 'r') as fin, open(new_filename, 'w') as fout:
            for line in fin:
                item = json.loads(line)
                new_item = {
                    "id": item['id'],
                    "img_path":  os.path.join("data/opensource_data/",item['image_url']),
                    "img_filename": os.path.join("data/opensource_data/",item['image_url']),
                    "instruction": item['instruction'],
                    "bbox": item["coordinate"],
                    "source": item["source"]
                }
                fout.write(json.dumps(new_item) + "\n")

        return new_filename
                


    def generate(self):
        
        model_name = self.model_name_or_path.split("/")[-1]

        new_filename = self.pre_process(self.source_dataset_path)

        single_step_dpo_syn_data_dir = os.path.join(self.synthetic_data_dir, model_name, "single_step_dpo")
        single_step_dpo_train_data_dir = os.path.join(self.train_data_dir, model_name, "single_step_dpo")
        
        os.makedirs(single_step_dpo_syn_data_dir, exist_ok=True)
        os.makedirs(single_step_dpo_train_data_dir, exist_ok=True)

        ####crop candidates 生成
        crop_candidates_output_path = os.path.join(single_step_dpo_syn_data_dir, "crop_candidates_dataset.jsonl")
        self.crop_candidates_generate(new_filename, crop_candidates_output_path)

        ###处理crop candidates
        select_crop_candidates_output_path = os.path.join(single_step_dpo_syn_data_dir, "select_crop_candidates_dataset.jsonl")
        self.single_step_dpo_processor.select_crop_candidates(crop_candidates_output_path, select_crop_candidates_output_path)

        ###对crop candidates打分（click pass@n）
        score_candidates_output_path = os.path.join(single_step_dpo_syn_data_dir, "score_candidates_dataset.jsonl")
        self.score_candidates_generate("double", select_crop_candidates_output_path, score_candidates_output_path)

        ###filter, 生成dpo训练数据，会生成两个，一个click，一个crop的dpo数据
        dpo_crop_train_data_path = os.path.join(single_step_dpo_train_data_dir, "dpo_crop_train.jsonl")
        dpo_click_train_data_path = os.path.join(single_step_dpo_train_data_dir, "dpo_click_train.jsonl")
        
        self.single_step_dpo_processor.filter_dpo_train_data(score_candidates_output_path, dpo_crop_train_data_path, dpo_click_train_data_path)



if __name__ == "__main__":
    single_step_dpo_generator = SingleStepDpoGenerator()
    single_step_dpo_generator.generate()