from vllm import SamplingParams, LLM
from tqdm import tqdm
from laser.utils.misc import load_and_resize_image, get_visible_gpus, setup_ray, crop_image_by_box, resize_image
from collections import Counter
from PIL import Image
import os
import yaml
import json
import ray
import numpy as np
import re
import copy
from math import ceil
from laser.actor.multi_step_sft_actor import MultiStepSftActor
from laser.actor.single_step_dpo_actor import SingleStepDpoActor
from laser.processor.multi_step_sft_processor import MultiStepSftProcessor
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoProcessor  
import logging      # 用于日志记录
import torch        # 用于深度学习及设置随机种子

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

class MultiStepSftGenerator:
    def __init__(self):
        with open("src/laser/configs/multi_step_sft.yaml") as fin:
            self.cfg = yaml.safe_load(fin)

        ###input
        self.model_name_or_path = self.cfg["input"]["model_name_or_path"]
        self.source_dataset_path = self.cfg["input"]["source_dataset_path"]

        ###output_data_dir
        self.synthetic_data_dir = self.cfg["output_data_dir"]["synthetic_data_dir"]
        self.train_data_dir = self.cfg["output_data_dir"]["train_data_dir"]

        ###generation
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

        self.multiStepSftProcessor = MultiStepSftProcessor()

        self.PROCESSOR = AutoProcessor.from_pretrained(self.model_name_or_path)

    def first_smart_size_trunk(self, chunk:list[dict], max_pixels:int):
        for data in chunk:
            img_path = data["img_path"]
            raw_image, resize_image = load_and_resize_image(
                image_url= img_path,
                factor= self.PROCESSOR.image_processor.patch_size * self.PROCESSOR.image_processor.merge_size,
                min_pixels= self.PROCESSOR.image_processor.min_pixels,
                max_pixels= self.max_pixels,
            )
            resize_image.save(data["resize_image_url"])
    def first_parallel_smart_size(self, tasks_to_run:list[dict], max_pixels:int, max_workers:int=30):
        if len(tasks_to_run) == 0:
            print("No tasks to resize")
            return
        chunk_size = ceil(len(tasks_to_run) / max_workers)
        chunks = [
            tasks_to_run[i:i+chunk_size]
            for i in range(0, len(tasks_to_run), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.first_smart_size_trunk, chunk, max_pixels) for chunk in chunks]
            for future in futures:
                future.result()
        return None
        
    def first_crop_smart_size_all_raw(self, tasks_to_run:list[dict], max_pixels:int):

        ###图像用id命名
        print("smart size all raw images")
        resize_image_dir = os.path.join(self.smart_image_dir, "first_crop" ,str(max_pixels))
        os.makedirs(resize_image_dir, exist_ok=True)

        existed_imgs = (os.listdir(resize_image_dir))

        ###并行保存没有smartsize的图片
        smart_size_tasks = []
        for item in tasks_to_run:
            file_exs = item["id"] + ".png"
            if file_exs not in existed_imgs:
                item["resize_image_url"] = os.path.join(resize_image_dir, file_exs)
                smart_size_tasks.append(item)
        self.first_parallel_smart_size(smart_size_tasks, self.max_pixels, max_workers=40)
        

        ###给item加上resize_image_url
        output_lst = []
        for item in tasks_to_run:
            file_exs = item["id"] + ".png"
            resize_image_url = os.path.join(resize_image_dir, file_exs)
            assert os.path.exists(resize_image_url)
            item["resize_image_url"] = resize_image_url
            output_lst.append(item)
        return output_lst


    def second_smart_size_trunk(self, chunk:list[dict], max_pixels:int):
        for resize_task in chunk:
            # load raw
            raw_image = Image.open(resize_task['raw_image_url']).convert('RGB')
            # get crop image
            crop_image = crop_image_by_box(raw_image, resize_task["crop_box_base_raw"])
            crop_image.save(resize_task['crop_image_save_url'])
            # get resize crop image
            resize_crop_image = resize_image(
                image = crop_image,
                factor= self.PROCESSOR.image_processor.patch_size * self.PROCESSOR.image_processor.merge_size,
                min_pixels= self.PROCESSOR.image_processor.min_pixels,
                max_pixels=self.max_pixels,
            )
            resize_crop_image.save(resize_task['resize_crop_image_save_url'])
            # done

    def second_parallel_smart_size(self, tasks_to_run:list[dict], max_pixels:int, max_workers:int=30):
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
            futures = [executor.submit(self.second_smart_size_trunk, chunk, max_pixels) for chunk in chunks]
            for future in futures:
                future.result()
        return None

    def second_crop_smart_size_all_raw(self, tasks_to_run:list[dict], max_pixels:int):

        ###图像用id命名
        print("[INFO] smart size all raw images")
        resize_image_dir = os.path.join(self.smart_image_dir, "second_crop", str(max_pixels))
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
        self.second_parallel_smart_size(smart_size_tasks, self.max_pixels, max_workers=40)
        
        new_tasks_to_run = []
        for item in tasks_to_run:
            if not os.path.exists(item["resize_image_url"]) or not os.path.exists(item["img_filename"]):
                continue
            new_tasks_to_run.append(item)
        return new_tasks_to_run

    def first_crop_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break
            
            json_items = []
            # show-ui desktop
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

    def second_crop_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break
            
            json_items = []
            # show-ui desktop
            print(f"[INFO] load from [{input_data_path}]")
            with open(input_data_path, 'r') as f:     
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

    def batch_inf(self, stage: str, input_data_path:str, output_data_path: str):
        setup_ray(self.worker_node)

        ####multi_step的第一步生成和single_dpo 的第一步生成时一样的，可以复用，只是continue_or_not、passN，不一样
        if stage == "first_crop":
            train_data_reader = self.first_crop_reader
            infer_actors = [SingleStepDpoActor.remote(
                model_name= self.model_name_or_path,
                actor_id= i
            ) for i in range(self.worker_node)]
            for infer_actor in infer_actors:
                infer_actor.set_response_file.remote(output_data_path)
        elif stage == "second_crop":
            train_data_reader = self.second_crop_reader
            infer_actors = [MultiStepSftActor.remote(
                model_name= self.model_name_or_path,
                actor_id= i
            ) for i in range(self.worker_node)]
            for infer_actor in infer_actors:
                infer_actor.set_response_file.remote(output_data_path)
        else:
            raise Exception("[ERROR] stage not support")
        
        for train_dataset in train_data_reader(input_data_path, output_data_path, self.max_samples):
            if stage == "first_crop":
                smart_size_all_raw = self.first_crop_smart_size_all_raw
            elif stage == "second_crop":
                smart_size_all_raw = self.second_crop_smart_size_all_raw

            tasks_to_run = smart_size_all_raw(train_dataset, self.max_pixels)
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
                if stage == "first_crop":
                    
                    future = infer_actor.crop_candidates_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
                        continue_or_not=True,
                        batch_size=self.batch_size,
                        max_pixels=self.max_pixels,
                        passN= self.passN,
                        max_roll= self.max_roll,
                    )
                    futures.append(future)
                elif stage == "second_crop": 
                    future = infer_actor.second_crop_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
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



    def first_crop_generate(self, input_data_path: str, output_data_path:str):
        self.batch_inf("first_crop", input_data_path, output_data_path)

    def second_crop_generate(self, input_data_path: str, output_data_path:str):
        self.batch_inf("second_crop", input_data_path, output_data_path)


    def generate(self):

        model_name = self.model_name_or_path.split("/")[-1]

        single_step_dpo_syn_data_dir = os.path.join(self.synthetic_data_dir, model_name, "multi_step_sft")
        single_step_dpo_train_data_dir = os.path.join(self.train_data_dir, model_name, "multi_step_sft")

        os.makedirs(single_step_dpo_syn_data_dir, exist_ok=True)
        os.makedirs(single_step_dpo_train_data_dir, exist_ok=True)

        ###第一步crop+click 生成
        first_crop_output_path = os.path.join(single_step_dpo_syn_data_dir, "first_crop_dataset.jsonl")
        self.first_crop_generate(self.source_dataset_path, first_crop_output_path)
        
        ###处理一步crop+click 数据
        process_first_crop_output_path = os.path.join(single_step_dpo_syn_data_dir, "process_first_crop_dataset.jsonl")
        self.multiStepSftProcessor.process_first_crop(first_crop_output_path, process_first_crop_output_path)

        ###基于第一步的多步crop+click 生成

        second_crop_output_path = os.path.join(single_step_dpo_syn_data_dir, "second_crop_dataset.jsonl")
        self.second_crop_generate(process_first_crop_output_path, second_crop_output_path)

        ###filter，生成单步crop 和多步crop的数据
        one_step_sft_train_data_path = os.path.join(single_step_dpo_train_data_dir, "one_step_sft.jsonl")
        two_step_sft_train_data_path = os.path.join(single_step_dpo_train_data_dir, "two_step_sft.jsonl")
        self.multiStepSftProcessor.filter_multi_step_sft_train_data(second_crop_output_path, one_step_sft_train_data_path, two_step_sft_train_data_path)

if __name__ == "__main__":
    multi_step_sft_generator = MultiStepSftGenerator()
    multi_step_sft_generator.generate()