####TODO:   
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
import re
import copy
from laser.actor.single_step_sft_actor import SingleStepSftActor
from laser.processor.single_step_sft_processor import SingleStepSftProcessor
import logging      # 用于日志记录
import torch        # 用于深度学习及设置随机种子

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)


class SingleStepSftGenerator:
    def __init__(self):
        with open("src/laser/configs/single_step_sft.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)

        self.model_name_or_path = self.cfg["input"]["model_name_or_path"]
        self.source_dataset_path = self.cfg["input"]["source_dataset_path"]

        self.synthetic_data_dir = self.cfg["output_data_dir"]["synthetic_data_dir"]
        self.train_data_dir = self.cfg["output_data_dir"]["train_data_dir"]

        self.batch_size = self.cfg["generation"]["batch_size"]
        self.max_samples = self.cfg["generation"]["max_samples"]
        self.max_pixels = self.cfg["generation"]["max_pixels"]

        self.worker_node = len(get_visible_gpus())

        self.max_reject_sampling_crop_num = self.cfg["generation"]["max_reject_sampling_crop_num"]
        self.max_reject_sampling_click_num = self.cfg["generation"]["max_reject_sampling_click_num"]

        self.tmp_output_image = self.cfg["process"]["tmp_output_image"]
        os.makedirs(self.tmp_output_image, exist_ok=True)

        self.processor = SingleStepSftProcessor()
    
    def click_data_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_reject_sampling_click_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break


            json_items = []
            print(f"[INFO] load from [{input_data_path}]")
            with open(input_data_path, 'r') as f:     ###showui 
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

    def crop_data_reader(self, input_data_path: str, output_data_path: str, max_samples: int= 20_000):
        while True:
            existed_ids = set()
            if os.path.exists(output_data_path):
                existed_ids = set([json.loads(line)['id'] for line in open(output_data_path)])
            if len(existed_ids) >= self.max_reject_sampling_crop_num:
                print(f"[INFO] skip existed: [{len(existed_ids)}], get max samples")
                break
            
            json_items = []
            print(f"[INFO] load from [{input_data_path}]")
            with open(input_data_path, 'r') as f:     ###showui 
                for line in f:
                    item = json.loads(line)
                    if item['id'] in existed_ids: continue
                    if len(item['coordinate']) != 4: continue
                    item['image_url'] = os.path.join("data/opensource_data/", item['image_url'])
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

    def batch_inf(self, stage: str, input_file_path: str, output_file_path: str, sampling_params: SamplingParams):

        setup_ray(self.worker_node)
        infer_actors = [SingleStepSftActor.remote(
            model_name= self.model_name_or_path,
            actor_id= i
        ) for i in range(self.worker_node)]
        for infer_actor in infer_actors:
            infer_actor.set_response_file.remote(output_file_path)
        
        # ray.get([actor.is_ready.remote() for actor in infer_actors])

        if stage == "crop":
            data_reader = self.crop_data_reader
        elif stage == "click":
            data_reader = self.click_data_reader
        else:
            raise Exception("[ERROR] stage not support")
        
        ###TODO:
        for train_dataset in data_reader(input_file_path, output_file_path, self.max_samples):

            print(f"[INFO] load train dataset: {len(train_dataset)} samples")
            train_chunks = np.array_split(train_dataset, self.worker_node)

            futures = []

            for chunk, infer_actor in zip(train_chunks, infer_actors):
                print(f"[DEBUG] chunk size: {len(chunk)}")
                if stage == "crop":
                    future = infer_actor.crop_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
                        # max_pixels=self.max_pixels,
                        MAX_PIXELS=self.max_pixels,
                        batch_size=self.batch_size,
                    )
                    futures.append(future)
                elif stage == "click":
                    future = infer_actor.click_infer.remote(
                        samples= chunk,
                        sampling_params= sampling_params,
                        MAX_PIXELS=self.max_pixels,
                        batch_size=self.batch_size,
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
    
    def set_setting(self, stage: str):
        if stage != "click" and stage != "crop":
            raise Exception("[ERROR] stage not support")
        
        sampling_params = SamplingParams(
            n=self.cfg["generation"][stage]["voting_times"],
            temperature=self.cfg["generation"]["temperature"],
            top_p=self.cfg["generation"]["top_p"],
            max_tokens=self.cfg["generation"]["max_tokens"],
        )
        
        return sampling_params
    def crop_data_generate(self, input_data_path:str, output_data_path:str):
        ###加载crop的配置
        sampling_params = self.set_setting("crop")
        self.batch_inf("crop", input_data_path, output_data_path, sampling_params)

    def click_data_generate(self, input_data_path: str, output_data_path:str):
        ###加载click的配置
        sampling_params = self.set_setting("click")
        self.batch_inf("click", input_data_path, output_data_path, sampling_params)

    def generate(self):

        model_name = self.model_name_or_path.split("/")[-1]
        single_step_sft_syn_data_dir = os.path.join(self.synthetic_data_dir, model_name, "single_step_sft")
        single_step_sft_train_data_dir = os.path.join(self.train_data_dir, model_name, "single_step_sft")
        
        os.makedirs(single_step_sft_syn_data_dir, exist_ok=True)
        os.makedirs(single_step_sft_train_data_dir, exist_ok=True)
        
        ###合成一步crop数据
        crop_dataset_output_path = os.path.join(single_step_sft_syn_data_dir, "crop_dataset.jsonl")
        self.crop_data_generate(self.source_dataset_path, crop_dataset_output_path)

        ###拒绝采样crop
        reject_sampling_crop_path = os.path.join(single_step_sft_syn_data_dir, "process_crop_dataset.jsonl")

        # self.processor.reject_sampling_crop(crop_dataset_output_path, reject_sampling_crop_path, self.tmp_output_image)

        ###基于一步crop数据，生成click数据
        click_dataset_output_path = os.path.join(single_step_sft_syn_data_dir, "click_dataset.jsonl")
        self.click_data_generate(reject_sampling_crop_path, click_dataset_output_path)

        #拒绝采样click,形成llama训练数据
        single_step_sft_train_data_path = os.path.join(single_step_sft_train_data_dir, "single_step_sft.jsonl")

        self.processor.reject_sampling_click(click_dataset_output_path, single_step_sft_train_data_path)

if __name__ == "__main__":
    single_step_sft_generator = SingleStepSftGenerator()
    single_step_sft_generator.generate()