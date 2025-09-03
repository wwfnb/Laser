import yaml
from laser.utils.misc import load_and_resize_image, get_visible_gpus, setup_ray, insertion_area, adjust_box, box_area, transfer_box, box_contains, box_valid
from tqdm import tqdm
import json
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math
from laser.prompt_builder.single_step_dpo_prompt_builder import SingleStepDpoPromptBuilder
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from laser.utils.misc import generate_id, get_date, load_and_resize_image, crop_image_by_box, resize_image

class SingleStepDpoProcessor:

    def __init__(self):
        with open("src/laser/configs/single_step_dpo.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)
        self.prompt_builder = SingleStepDpoPromptBuilder.ProcessPromptBuilder()
        self.max_pixels = self.cfg["process"]["max_pixels"]
        self.max_workers = self.cfg["process"]["max_workers"]

    def select_crop_candidates(self, input_data_path: str, output_data_path: str):
        

        count_dict = {
            0:0,
            1:0,
            2:0,
            3:0,
            4:0,
            5:0
        }

        def process_trajs(trajs, bbox):
            count_true = 0
            for t in trajs:
                pre_box = t["raw_box"][0]
                assert len(pre_box) == 4
                if pre_box[0] <= bbox[0] and pre_box[1] <= bbox[1] and pre_box[2] >= bbox[2] and pre_box[3] >= bbox[3]:
                    focus = True
                    count_true += 1
                else:
                    focus = False
                t["focus"] = focus
            count_dict[count_true] = count_dict[count_true] + 1

        def select(trajs):

            output_trajs = []
            f_lst = []
            t_lst = []
            for t in trajs:
                if t["focus"] == True:
                    t_lst.append(t)
                else:
                    f_lst.append(t)
            if len(t_lst) == 0:
                return []
            elif len(t_lst) == 1:
                output_trajs.append(f_lst[0])
                output_trajs.append(t_lst[0])
                # print(111)
            else:
                output_trajs.append(t_lst[0])
                output_trajs.append(t_lst[1])
            return output_trajs

        with open(input_data_path, "r") as fin, open(output_data_path, "w") as fout:
            for line in fin:
                item = json.loads(line)
                
                bbox = item["bbox"]
                trajs = item["trajs"]
                count1 = process_trajs(trajs, bbox)
                if count1 == 0:
                    continue

                item["trajs"] = select(trajs)
                if len(item["trajs"]) != 2:
                    continue
                assert len(item["trajs"]) == 2
                fout.write(json.dumps(item) + "\n")

        print(count_dict)
        


    def save_image(self, resize_image: Image.Image, output_image_folder: str, image_filename: str):
        image_path = os.path.join(output_image_folder, image_filename)
        if not os.path.exists(image_path):
            resize_image.save(image_path)
        assert os.path.exists(image_path)
        return image_path
    def resize_crop_and_save_image(self, raw_image: Image.Image, box: list, output_image_folder:str, image_filename: str):
        crop_image = crop_image_by_box(raw_image, box)
        resize_crop_image_height, resize_crop_image_width = smart_resize(
            crop_image.height,
            crop_image.width,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_crop_image = crop_image.resize((resize_crop_image_width, resize_crop_image_height))
        resize_crop_image_url = self.save_image(resize_crop_image, output_image_folder, image_filename)
        return crop_image, resize_crop_image, resize_crop_image_url

    def crop_process_json_line(self, line, out_put_image):
        SYS_MSG = self.prompt_builder.build_sys_msg()
        data = json.loads(line)

        ###处理图片
        raw_image_url = data["img_path"]
        raw_image, resize_raw_image = load_and_resize_image(
            image_url=raw_image_url,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_raw_image_url = os.path.join(out_put_image, data["id"] + "resize_raw.png")
        if not os.path.exists(resize_raw_image_url):
            resize_raw_image.save(resize_raw_image_url)

        instruction = data["instruction"]
        sorted_choices = sorted(data["choices"], key=lambda x: x["score"])

        rejected_choice = sorted_choices[0]
        chosen_choice = sorted_choices[1]

        ##分差大于3
        if chosen_choice["score"] - rejected_choice["score"] < 4:
            return None
        
        ###IOU小于0.8
        rejected_box = rejected_choice["raw_box"][0]
        chosen_box = chosen_choice["raw_box"][0]
        i_area = insertion_area(rejected_box, chosen_box)
        if i_area / (box_area(rejected_box) + box_area(chosen_box) - i_area) > 0.8:
            return None
        
        assert rejected_choice["score"] < chosen_choice["score"], "reject not smaller than chosen!!!!!!"

        user_msg_1 = self.prompt_builder.build_user_msg_1(instruction, resize_raw_image)

        assert rejected_choice["raw_response"][0] != "" and  chosen_choice["raw_response"][0] != ""

        rejected_assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(rejected_choice["raw_response"][0], rejected_choice["raw_box"][0], raw_image, resize_raw_image)
        chosen_assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(chosen_choice["raw_response"][0], chosen_choice["raw_box"][0], raw_image, resize_raw_image)

        try:
            with Image.open(resize_raw_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_raw_image_url} | Error: {e}")
            return None


        assert os.path.exists(resize_raw_image_url)
        return {
            "message": [SYS_MSG, user_msg_1],
            "chosen": chosen_assistant_msg_1,
            "rejected": rejected_assistant_msg_1,
            "images": [resize_raw_image_url],
        }


    def click_process_json_line(self, line, out_put_image):
        SYS_MSG = self.prompt_builder.build_sys_msg()
        data = json.loads(line)

        instruction = data["instruction"]

        ###选择得分较高的一个crop的路径, 但是得分较高的那个可能全对，所以如果全对，选择得分低的那个

        sorted_choices = sorted(data["choices"], key=lambda x: x["score"])
        assert sorted_choices[0]["score"] <= sorted_choices[1]["score"], "chosen one must has bigger score"

        selected_crop_choice = sorted_choices[1]
        if selected_crop_choice["score"] == 5:
            selected_crop_choice = sorted_choices[0]
            return None

        ###处理图片
        ##原始图片
        raw_image_url = data["img_path"]
        raw_image, resize_raw_image = load_and_resize_image(
            image_url=raw_image_url,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_raw_image_url = os.path.join(out_put_image, data["id"] + "resize_raw.png")
        if not os.path.exists(resize_raw_image_url):
            resize_raw_image.save(resize_raw_image_url)
        
        ###crop图片
        crop_image, resize_crop_image, resize_crop_image_url=self.resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][0], out_put_image, data["id"] + "resize_crop.png")
        ####

        ###挑选chosen click, 从selected_crop_choice的trajs里面挑一个click 正确和一个click错误的
        chosen_traj = None
        rejected_traj = None
        for t in selected_crop_choice["trajs"]:
            if t["correctness"] == True:
                chosen_traj = t
                break
        for t in selected_crop_choice["trajs"]:
            if t["correctness"] == False:
                rejected_traj = t
                break
        assert chosen_traj != None and rejected_traj != None
        assert chosen_traj["correctness"] == True and rejected_traj["correctness"] == False

        ###build
        user_msg_1 = self.prompt_builder.build_user_msg_1(instruction, resize_raw_image)
        assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(selected_crop_choice["raw_response"][0], selected_crop_choice["raw_box"][0], raw_image, resize_raw_image)
        user_msg_2 = self.prompt_builder.bulid_user_msg_2(resize_crop_image)
        try:
            chosen_assistant_msg_2 = self.prompt_builder.build_assistant_msg_2(chosen_traj["raw_response"][1], chosen_traj["raw_point"], chosen_traj["raw_box"][0], crop_image, resize_crop_image)
            rejected_assistant_msg_2 = self.prompt_builder.build_assistant_msg_2(rejected_traj["raw_response"][1], rejected_traj["raw_point"], rejected_traj["raw_box"][0], crop_image, resize_crop_image)
        except IndexError as e:
            return None
        

        try:
            with Image.open(resize_raw_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_raw_image_url} | Error: {e}")
            return None
        
        try:
            with Image.open(resize_crop_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_crop_image_url} | Error: {e}")
            return None
        

        assert os.path.exists(resize_raw_image_url) and os.path.exists(resize_crop_image_url)
        return {
            "message": [SYS_MSG, user_msg_1, assistant_msg_1, user_msg_2],
            "chosen": chosen_assistant_msg_2,
            "rejected": rejected_assistant_msg_2,
            "images": [resize_raw_image_url, resize_crop_image_url],
        }

    def crop_filter_sample(self, lines):
        print("filtering the sample!")
        print("[info] sum of sample: ", len(lines))

        filter_lines = []

        for l in lines:
            item = json.loads(l)

            choices = item["choices"]
            assert len(choices) == 2

            if choices[0]["score"] != choices[1]["score"]:
                filter_lines.append(l)
        print("[info] the sum of after filter: ", len(filter_lines))
        return filter_lines

    def click_filter_sample(self, lines):
        print("filtering the sample!")
        print("[info] sum of sample: ", len(lines))

        filter_lines = []

        for l in lines:
            item = json.loads(l)

            choices = item["choices"]
            assert len(choices) == 2

            ###不是click全对或者全错的
            if (choices[0]["score"] != 0 and choices[0]["score"] != 5) or (choices[1]["score"] != 0 and choices[1]["score"] != 5):
                filter_lines.append(l)
        print("[info] the sum of after filter: ", len(filter_lines))
        return filter_lines


    def process_chunk(self, chunk: list, stage: str, out_put_image):
        res = []
        for line in tqdm(chunk):
            if stage == "crop":
                res.append(self.crop_process_json_line(line, out_put_image))
            elif stage == "click":
                res.append(self.click_process_json_line(line, out_put_image))
            else:
                raise Exception("stage error")
        return res

    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, stage: str, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                ##过滤出有score不同的
                if stage == "crop":
                    lines = self.crop_filter_sample(fin.readlines())
                elif stage == "click":
                    lines = self.click_filter_sample(fin.readlines())
                else: 
                    raise Exception("stage error")

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, stage, out_put_image) for chunk in chunks]
        with open(train_data_json, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item in chunk_result:
                    if item is None: continue
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')


    def filter_dpo_train_data(self, input_data_path: str, output_dpo_crop_data_path: str, output_dpo_click_data_path: str):

        pre_dir  = os.path.dirname(output_dpo_crop_data_path)
        crop_output_image = os.path.join(pre_dir, "dpo_crop_image")
        click_output_image = os.path.join(pre_dir, "dpo_click_image")
        os.makedirs(crop_output_image, exist_ok=True)
        os.makedirs(click_output_image, exist_ok=True)

        print("生成crop dpo 数据..........")
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_dpo_crop_data_path,
            out_put_image=crop_output_image,
            stage="crop",
            max_workers=self.max_workers

        )

        print("生成click dpo 数据..........")
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_dpo_click_data_path,
            out_put_image=click_output_image,
            stage="click",
            max_workers=self.max_workers
        )

