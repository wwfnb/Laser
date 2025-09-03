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
from concurrent.futures import ProcessPoolExecutor, as_completed
from laser.generator.singe_step_dpo_genertor import SingleStepDpoGenerator
from laser.processor.single_step_dpo_processor import SingleStepDpoProcessor
from laser.processor.multi_step_dpo_processor import SecondProcessor, TribleProcessor, MultiStepDpoProcessor, TrainDataProcessor
import logging      # 用于日志记录
import torch        # 用于深度学习及设置随机种子

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

class MultiStepDpoGenerator(SingleStepDpoGenerator):

    def __init__(self):
        # super().__init__()

        with open("src/laser/configs/multi_step_dpo.yaml", "r") as fin:
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
        # self.multi_step_dpo_processor = MultiStepDpoProcessor()
        self.second_processor = SecondProcessor()
        self.trible_processor = TribleProcessor()

        self.train_data_processor = TrainDataProcessor()
        self.PROCESSOR = AutoProcessor.from_pretrained(self.model_name_or_path)
            
    def generate(self):

        ####multistep dpo与single step dpo 前两步生成数据程序相同， 以及第一步生成数据后的处理程序相同，所以直接用single step dpo
        model_name = self.model_name_or_path.split("/")[-1]
        multi_step_dpo_syn_data_dir = os.path.join(self.synthetic_data_dir, model_name, "multi_step_dpo")
        multi_step_dpo_train_data_dir = os.path.join(self.train_data_dir, model_name, "multi_step_dpo")
        
        os.makedirs(multi_step_dpo_syn_data_dir, exist_ok=True)
        os.makedirs(multi_step_dpo_train_data_dir, exist_ok=True)

        ####crop candidates 生成
        crop_candidates_output_path = os.path.join(multi_step_dpo_syn_data_dir, "crop_candidates_dataset.jsonl")
        self.crop_candidates_generate(self.source_dataset_path, crop_candidates_output_path)

        ###处理crop candidates
        select_crop_candidates_output_path = os.path.join(multi_step_dpo_syn_data_dir, "select_crop_candidates_dataset.jsonl")
        self.single_step_dpo_processor.select_crop_candidates(crop_candidates_output_path, select_crop_candidates_output_path)

        ###生成第二步，对第一步得crop candidates打分（click pass@n）
        score_candidates_output_path = os.path.join(multi_step_dpo_syn_data_dir, "score_candidates_dataset.jsonl")
        self.score_candidates_generate("double", select_crop_candidates_output_path, score_candidates_output_path)

        ###################################################
        ###处理第二步生成的各种dpo对, 以及为第三步准备数据
        second_step_dpo_data = os.path.join(multi_step_dpo_train_data_dir, "second_step_dpo_data.jsonl")
        for_trible_score_candidates_output_path = os.path.join(multi_step_dpo_syn_data_dir, "for_trible_score_candidates.jsonl")
        self.second_processor.process_second(score_candidates_output_path, for_trible_score_candidates_output_path, second_step_dpo_data)
        
        ###生成第三步，对第二步的crop打分
        score_candidates_by_trible_output_path = os.path.join(multi_step_dpo_syn_data_dir, "score_candidates_by_trible.jsonl")
        self.score_candidates_generate("trible", for_trible_score_candidates_output_path, score_candidates_by_trible_output_path)

        ###处理第三步生成的，生成第三步的各种dpo对
        trible_step_dpo_train_output_path = os.path.join(multi_step_dpo_train_data_dir, "trible_step_dpo_data.jsonl")
        self.trible_processor.process_trible(score_candidates_by_trible_output_path, trible_step_dpo_train_output_path)

        ###生成dpo训练数据， crop的和click的dpo数据对，crop：第一步的和第二步的，click： 第二步的, 输入3个文件   最后merge在一起， 形成final.jsonl 
        multi_step_dpo_train_output_path = os.path.join(multi_step_dpo_train_data_dir, "multi_step_dpo_train_data-final.jsonl")
        self.train_data_processor.transfer_to_train_data(
            score_candidates_by_second_output_path= score_candidates_output_path,
            score_candidates_by_trible_output_path = score_candidates_by_trible_output_path,
            second_step_dpo_data= second_step_dpo_data,
            train_data_output_path= multi_step_dpo_train_output_path
            )
        

if __name__ == "__main__":
    multi_step_dpo_generator = MultiStepDpoGenerator()
    multi_step_dpo_generator.generate()