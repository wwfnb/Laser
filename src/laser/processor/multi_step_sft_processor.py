import yaml
from laser.utils.misc import load_and_resize_image, get_visible_gpus, setup_ray, insertion_area, adjust_box, box_area, transfer_box, box_contains, box_valid
from tqdm import tqdm
import json
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math
from laser.prompt_builder.multi_step_sft_prompt_builder import MultiStepSftPromptBuilder
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from laser.utils.misc import generate_id, get_date, load_and_resize_image, crop_image_by_box, resize_image


class MultiStepSftProcessor:

    def __init__(self):
        with open("src/laser/configs/multi_step_sft.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)

        self.prompt_builder = MultiStepSftPromptBuilder.ProcessPromptBuilder()
        self.max_pixels = self.cfg["process"]["max_pixels"]
        self.max_workers = self.cfg["process"]["max_workers"]
        self.DOUBLE_CROP_RATE = self.cfg["process"]["DOUBLE_CROP_RATE"]
    

    def eval_sample_positive_gt(self, trajs, bbox):
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

    def eval_trajs(self, item):
        trajs = item["trajs"]
        for t in trajs:
            n_focus_correctness, result_lst, correctness = self.eval_sample_positive_gt([t], item["bbox"])
            if len(n_focus_correctness[0]) == 0:
                t["n_focus_correctness"] = "wrong_format"
            else:
                t["n_focus_correctness"] = n_focus_correctness[0][0]
            t["correctness"] = correctness

    def keep_one_traj_crop_right_click_right(self, center: list, trajs: list[dict]):
        # click right
        correct_trajs = list(filter(lambda t: t['correctness'] == 'correct', trajs))
        # crop right
        correct_trajs = list(filter(lambda t: t['n_focus_correctness'] == 'correct', correct_trajs))

        if len(correct_trajs) < 3:
            return None
        def dis2center(t: dict):
            click_point = t['raw_point']
            return (click_point[0] - center[0]) ** 2 + (click_point[1] - center[1]) ** 2
        correct_trajs.sort(key=dis2center)
        return correct_trajs[0]

    def keep_one_traj_crop_right_click_wrong(self, center: list, trajs: list[dict]):
        # click right
        correct_trajs = list(filter(lambda t: t['correctness'] == 'wrong', trajs))
        # crop right
        correct_trajs = list(filter(lambda t: t['n_focus_correctness'] == 'correct', correct_trajs))

        if len(correct_trajs) < 1:
            return None
        return correct_trajs[0]
        
    def process_first_crop(self, input_data_path: str, output_data_path: str):

        with open(input_data_path, "r") as fin, open(output_data_path, "w") as fout:
            for line in fin:
                item = json.loads(line)
                self.eval_trajs(item)
                
                golden_box = item['bbox']
                golden_center = [(golden_box[0] + golden_box[2]) / 2, (golden_box[1] + golden_box[3]) / 2]
                keep_traj = self.keep_one_traj_crop_right_click_wrong(center= golden_center, trajs= item['trajs'])
                if keep_traj is None:
                    continue
                transfer_item = copy.deepcopy(item)
                transfer_item.pop('trajs')

                transfer_item['old_traj'] = keep_traj
                fout.write(json.dumps(transfer_item) + "\n")
    def clear_cot(self, cot: str):
        # 删除cot中的 <box> ..</box>
        clear_cot = re.sub(r'<box>.*?</box>', '', cot)
        return clear_cot
    

    def box_area(self, box: list):
        return (box[2] - box[0]) * (box[3] - box[1])

    def eval_box(self, bbox, first_box, second_box):
        second_box_base_raw = [second_box[0] + first_box[0], second_box[1] + first_box[1], second_box[2] + first_box[0], second_box[3] + first_box[1]]
        return box_contains(second_box_base_raw, bbox)

    def eval_point(self, bbox, first_box, second_point):
        second_point_base_raw = [second_point[0] + first_box[0], second_point[1] + first_box[1]]
        return bbox[0] <= second_point_base_raw[0] <= bbox[2] and bbox[1] <= second_point_base_raw[1] <= bbox[3]

    def select_second_traj(self, item, raw_image_url):

        raw_image = Image.open(raw_image_url).convert("RGB")
        raw_image_size = raw_image.width * raw_image.height
        first_box = item["old_traj"]["raw_box"][0]
        ### 第一个框的大小 在 0.3 - 0.8之间
        if not (raw_image_size * self.DOUBLE_CROP_RATE > box_area(first_box)):
            return None
        trajs = item["trajs"]
        for t in trajs:
            try:
                if len(t["raw_box"]) < 1 or len(t["raw_box"][0]) != 4 or len(t["raw_point"]) != 2:
                    continue
                second_box = t["raw_box"][0]
                second_point = t["raw_point"]

                if self.eval_box(bbox = item["bbox"], first_box=first_box, second_box=second_box) and self.eval_point(bbox = item["bbox"], first_box=first_box, second_point=second_point):
                    ###第二个框的大小
                    if box_area(second_box) <= box_area(first_box) * 0.5:
                        # print(1111)
                        return t
            except Exception as e:
                print(e)
        return None

    def save_image(self, resize_image: Image.Image, output_image_folder: str, image_filename: str):
        image_path = os.path.join(output_image_folder, image_filename)
        if not os.path.exists(image_path):
            resize_image.save(image_path)
        assert os.path.exists(image_path)
        return image_path
    ###返回resized 后的crop图像
    def resize_crop_and_save_image(self, raw_image: Image.Image, box: list, save_image_url: str):
        crop_image = crop_image_by_box(raw_image, box)
        resize_crop_image_height, resize_crop_image_width = smart_resize(
            crop_image.height,
            crop_image.width,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_crop_image = crop_image.resize((resize_crop_image_width, resize_crop_image_height))
        if not os.path.exists(save_image_url):
            resize_crop_image.save(save_image_url)
        return crop_image, resize_crop_image




    def process_json_line(self, line, out_put_image):

        SYS_MSG = self.prompt_builder.build_sys_msg()

        data = json.loads(line)

        instruction = data["instruction"]
        raw_image_url = data["raw_image_url"]

        first_crop_box_base_raw = data["old_traj"]["raw_box"][0]

        raw_image, resize_raw_image = load_and_resize_image(
            image_url=raw_image_url,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )

        ###处理第一步的图像和raw 图像
        resize_raw_image_url = os.path.join(out_put_image, data["id"] + "resize_raw.png")
        if not os.path.exists(resize_raw_image_url):
            resize_raw_image.save(resize_raw_image_url)

        first_resize_crop_image_url = os.path.join(out_put_image, data["id"] + "resize_first_crop.png")
        first_crop_image, first_resize_crop_image = self.resize_crop_and_save_image(raw_image= raw_image, box= first_crop_box_base_raw, save_image_url= first_resize_crop_image_url)
        
        #####
        user_msg_1 = self.prompt_builder.build_user_msg_1(instruction, resize_raw_image)

        first_crop_assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(data["old_traj"]["raw_response"][0])

        first_crop_user_msg_2  = self.prompt_builder.bulid_user_msg_2(first_resize_crop_image)

        ###combine 中trajs为None 的是没有生成second_traj 的crop_right, click_right 数据，或者是生成second_traj时，第二条没有正确的
        ### 挑一条的raw_box
        if data["trajs"] == None or self.select_second_traj(data, raw_image_url) == None:
            
            ###统计第一个
            area_rate = (box_area(first_crop_box_base_raw) / (raw_image.size[0] * raw_image.size[1]))
            # area_sum.value += area_rate
            # count_sum.value += 1

            if area_rate > 0.3:
                return None
            
            for image_url in [resize_raw_image_url, first_resize_crop_image_url]:
                try:
                    with Image.open(image_url) as img:
                        img.verify()  # 快速验证图像是否损坏
                except Exception as e:
                    print(f"Invalid image: {resize_raw_image_url} | Error: {e}")
                    return None
                
            assistant_msg_2 = self.prompt_builder.build_assistant_msg_2(data["old_traj"]["raw_response"][1])
            return {
            "message": [SYS_MSG, user_msg_1, first_crop_assistant_msg_1, first_crop_user_msg_2, assistant_msg_2],
            "images": [resize_raw_image_url, first_resize_crop_image_url],
        }
        second_traj = self.select_second_traj(data, raw_image_url)  
        second_resize_crop_image_url = os.path.join(out_put_image, data["id"] + f"resize_second_crop{self.DOUBLE_CROP_RATE}.png")
        ###处理第二步图像
        second_crop_image, second_resize_crop_image = self.resize_crop_and_save_image(raw_image= first_crop_image, box= second_traj["raw_box"][0], save_image_url= second_resize_crop_image_url)

        ###
        second_crop_assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(second_traj["raw_response"][0])

        second_crop_user_msg_2 = self.prompt_builder.bulid_user_msg_2(second_resize_crop_image)

        assistant_msg_2 = self.prompt_builder.build_assistant_msg_2(second_traj["raw_response"][1])

        for image_url in [resize_raw_image_url, first_resize_crop_image_url, second_resize_crop_image_url]:
            try:
                with Image.open(image_url) as img:
                    img.verify()  # 快速验证图像是否损坏
            except Exception as e:
                print(f"Invalid image: {resize_raw_image_url} | Error: {e}")
                return None
            
        return {
            "message": [SYS_MSG, user_msg_1, first_crop_assistant_msg_1, first_crop_user_msg_2, second_crop_assistant_msg_1, second_crop_user_msg_2, assistant_msg_2],
            "images": [resize_raw_image_url, first_resize_crop_image_url, second_resize_crop_image_url],
        }


    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res


    def process_all_jsons(self, input_json_list, train_data_one_json, train_data_two_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        lines = []
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                lines.extend(fin.readlines())

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, out_put_image) for chunk in chunks]
        with open(train_data_one_json, "w", encoding="utf-8") as fout_one, open(train_data_two_json, "w", encoding="utf-8") as fout_two:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item in chunk_result:
                    if item is None: continue
                    if len(item["images"]) == 2:
                        fout_one.write(json.dumps(item, ensure_ascii=False) + '\n')
                    elif len(item["images"]) == 3:
                        fout_two.write(json.dumps(item, ensure_ascii=False) + '\n')
                    else:
                        print("error!!")

    def combine_two_data(self, first_input_data_path: str, second_input_data_path: str, output_data_path: str):

        set_id_second = set()

        with open(output_data_path, "w") as fout:
        
            with open(second_input_data_path, "r") as fin:
                for line in fin:
                    data = json.loads(line)
                    raw_image_url = data["raw_image_url"]
                    second_traj = self.select_second_traj(data, raw_image_url)   

                    if second_traj != None:
                        set_id_second.add(data["id"])
                        fout.write(json.dumps(data) + '\n')

            with open(first_input_data_path, "r") as fin:
                for line in fin:
                    data = json.loads(line)
                    if data["id"] not in set_id_second:
                        self.eval_trajs(data)

                        golden_box = data['bbox']
                        golden_center = [(golden_box[0] + golden_box[2]) / 2, (golden_box[1] + golden_box[3]) / 2]
                        keep_traj = self.keep_one_traj_crop_right_click_right(center= golden_center, trajs= data['trajs'])
                        if keep_traj is None:
                            continue
                        transfer_item = copy.deepcopy(data)
                        transfer_item.pop('trajs')

                        transfer_item['old_traj'] = keep_traj
                        transfer_item['trajs'] = None
                        transfer_item["raw_image_url"] = transfer_item["img_path"]

                        fout.write(json.dumps(transfer_item) + '\n')

    
    def filter_multi_step_sft_train_data(self, first_input_data_path:str, second_input_data_path: str, output_one_step_path: str, output_two_step_path: str):



        pre_dir  = os.path.dirname(output_one_step_path)
        output_image = os.path.join(pre_dir, "multi_step_sft_image")
        os.makedirs(output_image, exist_ok=True)

        input_data_path = os.path.join(os.path.dirname(first_input_data_path), "combine_two_data.jsonl")

        self.combine_two_data(first_input_data_path=first_input_data_path, second_input_data_path=second_input_data_path, output_data_path=input_data_path)

        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_one_json=output_one_step_path,
            train_data_two_json=output_two_step_path,
            out_put_image=output_image,
            max_workers=self.max_workers
        )
        