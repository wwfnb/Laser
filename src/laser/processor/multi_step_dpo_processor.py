import yaml
from laser.utils.misc import load_and_resize_image, get_visible_gpus, setup_ray, insertion_area, adjust_box, box_area, transfer_box, box_contains, box_valid
from tqdm import tqdm
import json
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math
from laser.prompt_builder.multi_step_dpo_prompt_builder import MultiStepDpoPromptBuilder
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from laser.utils.misc import generate_id, get_date, load_and_resize_image, crop_image_by_box, resize_image

class MultiStepDpoProcessor():

    def __init__(self):
        with open("src/laser/configs/multi_step_dpo.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)
        self.prompt_builder = MultiStepDpoPromptBuilder.ProcessPromptBuilder()
        self.max_pixels = self.cfg["process"]["max_pixels"]
        self.max_workers = self.cfg["process"]["max_workers"]
        self.iou_rate = self.cfg["process"]["iou_rate"]


    def save_image(self, resize_image: Image.Image, output_image_folder: str, image_filename: str):
        image_path = os.path.join(output_image_folder, image_filename)
        if not os.path.exists(image_path):
            resize_image.save(image_path)
        assert os.path.exists(image_path)
        return image_path
    ###返回resized 后的crop图像
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

    def select_choice(self, choices: list):
        sorted_choices = sorted(choices, key=lambda x: x["score"])
        assert sorted_choices[0]["score"] <= sorted_choices[1]["score"], "chosen one must has bigger score"
        selected_crop_choice = sorted_choices[1]

        return selected_crop_choice
    
    def parser_res(self, res: str):
        # print("paser_res.........")
        def get_para(action_str: str):
            box_match = re.findall(r"\d+(?:\.\d+)?", action_str)
            box = [float(x) for x in box_match]
            return box

        think = None
        action = None
        para = None
        res = res.strip()
        think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        if not res.endswith("</action>"):
            res += "</action>"
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", res, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
            if "click" in action_text:
                action = "click"
            elif "crop" in  action_text:
                action = "crop"
            para = get_para(action_text)
        # print(f"action: {action}, para: {para}")
        return think, action, para
    
    def select_choice(self, choices: list):
        sorted_choices = sorted(choices, key=lambda x: x["score"])
        assert sorted_choices[0]["score"] <= sorted_choices[1]["score"], "chosen one must has bigger score"
        selected_crop_choice = sorted_choices[1]

        return selected_crop_choice
    
    def crop_right(self, raw_box, bbox):
        if raw_box[0] <= bbox[0] and raw_box[1] <= bbox[1] and raw_box[2] >= bbox[2] and raw_box[3] >= bbox[3]:
            return True
        return False
    
    def click_right(self, raw_point, bbox):
        if (bbox[0] <= raw_point[0] <= bbox[2]) and (bbox[1] <= raw_point[1] <= bbox[3]):
            return True
        return False
    
    def to_id_set(self, trajs):
        return set(str(traj["index"]) for traj in trajs)

    def select_click_by_distance(self, trajs, bbox, flat):
        def center_of_bbox(bbox):
            x1, y1, x2, y2 = bbox
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        
        if len(trajs) == 1:
            return trajs[0]

        # 按距离排序
        new_trajs = [
            (t, distance(t["raw_point"], bbox))
            for t in trajs
        ]

        new_trajs.sort(key=lambda x: x[1])

        if flat:
            return new_trajs[0][0]
        return new_trajs[-1][0]

    def cal_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集坐标
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # 计算交集面积
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # 计算各自面积
        area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 避免除以 0
        if union_area == 0:
            return 0.0

        # 计算 IoU
        iou = inter_area / union_area
        return iou
    def select_crop_by_area(self, trajs):
        def bbox_area(bbox):
            x1, y1, x2, y2 = bbox
            return (x2- x1)*(y2- y1)
        
        return min(trajs, key=lambda t: bbox_area(t['raw_box'][1])) 

    
class SecondProcessor(MultiStepDpoProcessor):
    def __init__(self):
        super().__init__()


    def build_crop_message(self, raw_response, first_crop_box_in_raw, second_crop_box_in_raw,  crop_image: Image.Image, resize_crop_image: Image.Image):
        second_crop_box_in_first = [
            second_crop_box_in_raw[0] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[1] - first_crop_box_in_raw[1],
            second_crop_box_in_raw[2] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[3] - first_crop_box_in_raw[1]
        ]

        ####在这里面转换box到模型处理的分辨率， 基于最近处理的一张图片， raw_box 也是需要基于最近的一张图片
        return self.prompt_builder.build_assistant_msg_1(raw_response=raw_response, raw_box=second_crop_box_in_first, raw_image=crop_image, resize_raw_image=resize_crop_image)

    def build_click_message(self, raw_response:str, click_point_in_raw:list, crop_box_in_raw:list, crop_image: Image.Image, resize_crop_image: Image.Image):
        return self.prompt_builder.build_assistant_msg_2(raw_response=raw_response, click_point_in_raw=click_point_in_raw, crop_box_in_raw=crop_box_in_raw, crop_image=crop_image, resize_crop_image=resize_crop_image)



    def build_pair(self, chosen_message, rejected_message ,message_and_image_dict, chosen_type, rejected_type):
        return {
            "message": message_and_image_dict["message"],
            "chosen": chosen_message,
            "rejected": rejected_message,
            "images": message_and_image_dict["images"],
            "chosen_type": chosen_type,
            "rejected_type": rejected_type
        }
    
    def build_message_and_image(self, data, selected_crop_choice, output_image):
        SYS_MSG = self.prompt_builder.build_sys_msg()
        ###处理图片
        raw_image_url = data["img_path"]
        raw_image, resize_raw_image = load_and_resize_image(
            image_url=raw_image_url,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_raw_image_url = os.path.join(output_image, data["id"] + "resize_raw.png")
        if not os.path.exists(resize_raw_image_url):
            resize_raw_image.save(resize_raw_image_url)

        crop_image, resize_crop_image, resize_crop_image_url=self.resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][0], output_image, data["id"] + "resize_crop.png")
        
        ###build
        user_msg_1 = self.prompt_builder.build_user_msg_1(data["instruction"], resize_raw_image)
        assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(selected_crop_choice["raw_response"][0], selected_crop_choice["raw_box"][0], raw_image, resize_raw_image)
        user_msg_2 = self.prompt_builder.bulid_user_msg_2(resize_crop_image)

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

        return {
            "message": [SYS_MSG, user_msg_1, assistant_msg_1, user_msg_2],
            "images": [resize_raw_image_url, resize_crop_image_url],
            "raw_images": [raw_image, crop_image]
        }
    def select_click_or_crop_trajs(self, trajs, action_type, bbox, flat:bool):
        assert action_type == "click" or action_type == "crop"
        result = []
        if action_type == "click":
            for i in range(len(trajs)):
                try:
                    if self.second_type(trajs[i]) == action_type and self.click_right(raw_point= trajs[i]["raw_point"], bbox=bbox) == flat:
                        trajs[i]["index"] = i
                        result.append(trajs[i])
                except ValueError as e:
                    print("[second_type_error]")
                
        elif action_type == "crop":
            for i in range(len(trajs)):
                try:
                    if self.second_type(trajs[i]) == action_type and self.crop_right(raw_box= trajs[i]["raw_box"][1], bbox=bbox) == flat:
                        trajs[i]["index"] = i
                        result.append(trajs[i])
                except ValueError as e:
                    print("[second_type_error]")
        return result

    def construction_chosen_click_right(self, click_right_trajs, click_wrong_trajs, crop_right_trajs, crop_wrong_trajs, message_and_image_dict, selected_crop_choice):

        dpo_pairs = []

        raw_image = message_and_image_dict["raw_images"][0]

        resize_raw_image_url = message_and_image_dict["images"][0]
        resize_raw_image = Image.open(resize_raw_image_url)

        crop_image = message_and_image_dict["raw_images"][1]

        resize_crop_image_url = message_and_image_dict["images"][1]
        resize_crop_image = Image.open(resize_crop_image_url)

        ###将click right当作chosen 构造dpo数据
        rejected_trajs = []
        ####挑选一个离golden answer 中心点更近的
        chosen_click_right = self.select_click_by_distance(trajs=click_right_trajs, bbox= selected_crop_choice["bbox"], flat=True)
        chosen_click_right["chosen_type"] = "click_right"
        chosen_message = self.build_click_message(
            raw_response= chosen_click_right["raw_response"][1], 
            click_point_in_raw= chosen_click_right["raw_point"],
            crop_box_in_raw= chosen_click_right["raw_box"][0],
            crop_image=crop_image,
            resize_crop_image=resize_crop_image
            )
        if len(click_wrong_trajs) != 0:
            ####挑选一个距离gold answer 中心点更远的
            rejected_click_wrong = self.select_click_by_distance(trajs=click_wrong_trajs, bbox= selected_crop_choice["bbox"], flat=False)
            rejected_click_wrong["rejected_type"] = "click_wrong"
            rejected_trajs.append(rejected_click_wrong)
        if len(crop_right_trajs) != 0:
            ###TODO：随机挑选一个
            rejected_crop_right = crop_right_trajs[0]
            rejected_crop_right["rejected_type"] = "crop_right"
            rejected_trajs.append(rejected_crop_right)

        if len(crop_wrong_trajs) != 0:
            ###TODO: 随机选一个
            rejected_crop_wrong = crop_wrong_trajs[0]
            rejected_crop_wrong["rejected_type"] = "crop_wrong"
            rejected_trajs.append(rejected_crop_wrong)

        for rejected in rejected_trajs:
            if "click" in rejected["rejected_type"]:
                rejected_message= self.build_click_message(
                    raw_response= rejected["raw_response"][1], 
                    click_point_in_raw= rejected["raw_point"],
                    crop_box_in_raw= rejected["raw_box"][0],
                    crop_image=crop_image,
                    resize_crop_image=resize_crop_image
                )
                dpo_pairs.append(self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_click_right["chosen_type"], rejected_type=rejected["rejected_type"]))

            elif "crop" in rejected["rejected_type"]:
                ####TODO：这里有BUG, 应该将第二步的raw_box, 转化到crop 后的图像的， 然后再转化为resize后的图像的
                rejected_message= self.build_crop_message(
                    raw_response= rejected["raw_response"][1],
                    first_crop_box_in_raw= rejected["raw_box"][0],
                    second_crop_box_in_raw= rejected["raw_box"][1],
                    crop_image= crop_image,
                    resize_crop_image= resize_crop_image
                )
                dpo_pairs.append(self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_click_right["chosen_type"], rejected_type=rejected["rejected_type"]))

        return dpo_pairs

    def cal_box_area_proportion(self, box:list,width,height):
        return ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
    def second_type(self, t):
        if len(t["raw_box"]) == 1 and t["raw_point"] != None:
            return "click"
        elif len(t["raw_box"]) >= 2:
            return "crop"
        else:
            raise ValueError(f"Invalid data format")
    def construction_chosen_crop_right(self, click_right_trajs, click_wrong_trajs, crop_right_trajs, crop_wrong_trajs, message_and_image_dict, selected_crop_choice):
        dpo_pairs=[]

        raw_image = message_and_image_dict["raw_images"][0]

        resize_raw_image_url = message_and_image_dict["images"][0]
        resize_raw_image = Image.open(resize_raw_image_url)

        crop_image = message_and_image_dict["raw_images"][1]

        resize_crop_image_url = message_and_image_dict["images"][1]
        resize_crop_image = Image.open(resize_crop_image_url)

        ###将crop_right当作chosen 构造dpo数据
        rejected_trajs = []
        ###怎么挑, 挑更小的一个
        chosen_crop_right = self.select_crop_by_area(trajs= crop_right_trajs)
        chosen_crop_right["chosen_type"] = "crop_right"
        chosen_message= self.build_crop_message(
            raw_response= chosen_crop_right["raw_response"][1],
            first_crop_box_in_raw= chosen_crop_right["raw_box"][0],
            second_crop_box_in_raw= chosen_crop_right["raw_box"][1],
            crop_image= crop_image,
            resize_crop_image= resize_crop_image
        )

        if len(click_wrong_trajs) != 0:
            ####挑选一个距离gold answer 中心点更远的
            rejected_type = "click_wrong"
            rejected_click_wrong = self.select_click_by_distance(trajs=click_wrong_trajs, bbox= selected_crop_choice["bbox"], flat=False)
            rejected_click_wrong["rejected_type"] = "click_wrong"
            rejected_trajs.append(rejected_click_wrong)

        if len(crop_right_trajs) != 0:
            rejected_type = "crop_right"
            ###TODO：需要之后sample5次， 挑比分差距>= 4的 并且iou <0.8的
            rejected_crop_right = crop_right_trajs[0]
            if len(crop_right_trajs) >= 2:
                for i in range(len(crop_right_trajs)):
                    ### 与chosen 不同 并且 iou < 0.8
                    if crop_right_trajs[i]["index"] != chosen_crop_right["index"] and self.cal_iou(crop_right_trajs[i]["raw_box"][1], chosen_crop_right["raw_box"][1]) < self.iou_rate:
                        rejected_crop_right= crop_right_trajs[i]
                        rejected_crop_right["rejected_type"] = "crop_right"
                        rejected_trajs.append(rejected_crop_right)
                        break

        if len(crop_wrong_trajs) != 0:
            ###IOU需要<0.8或者其他值, 只挑一个
            for i in range(len(crop_wrong_trajs)):
                if self.cal_iou(chosen_crop_right["raw_box"][1], crop_wrong_trajs[i]["raw_box"][1]) < self.iou_rate:
                    rejected_crop_wrong= crop_wrong_trajs[i]
                    rejected_crop_wrong["rejected_type"] = "crop_wrong"
                    rejected_trajs.append(rejected_crop_wrong)
                    break
        
        for rejected in rejected_trajs:
            if "click" in rejected["rejected_type"]:
                rejected_message= self.build_click_message(
                    raw_response= rejected["raw_response"][1], 
                    click_point_in_raw= rejected["raw_point"],
                    crop_box_in_raw= rejected["raw_box"][0],
                    crop_image=crop_image,
                    resize_crop_image=resize_crop_image
                )
                dpo_pairs.append(self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_crop_right["chosen_type"], rejected_type=rejected["rejected_type"]))

            if "crop" in rejected["rejected_type"]:
                rejected_message= self.build_crop_message(
                    raw_response= rejected["raw_response"][1],
                    first_crop_box_in_raw= rejected["raw_box"][0],
                    second_crop_box_in_raw= rejected["raw_box"][1],
                    crop_image= crop_image,
                    resize_crop_image= resize_crop_image
                )
                pair = self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_crop_right["chosen_type"], rejected_type=rejected["rejected_type"])
                ####增加traj的信息， 用来第三轮生成
                pair["for_trible_turn"] = {
                    "id": selected_crop_choice["id"],
                    "prompt_to_evaluate": selected_crop_choice["prompt_to_evaluate"],
                    "img_path": selected_crop_choice["img_path"],
                    "img_filename": selected_crop_choice["img_filename"],
                    "instruction": selected_crop_choice["instruction"],
                    "bbox": selected_crop_choice["bbox"],
                    "trajs" :[chosen_crop_right, rejected]
                }
                dpo_pairs.append(pair)

        return dpo_pairs
        
    def construction_dpo_data(self, selected_crop_choice, message_and_image_dict):

        dpo_pairs = []
        
        click_right_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "click", bbox= selected_crop_choice["bbox"], flat=True)
        click_wrong_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "click", bbox= selected_crop_choice["bbox"], flat=False)
        crop_right_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "crop", bbox= selected_crop_choice["bbox"], flat=True)
        crop_wrong_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "crop", bbox= selected_crop_choice["bbox"], flat=False)

        ###确保这四个没有重复
        click_right_ids = self.to_id_set(click_right_trajs)
        click_wrong_ids = self.to_id_set(click_wrong_trajs)
        crop_right_ids = self.to_id_set(crop_right_trajs)
        crop_wrong_ids = self.to_id_set(crop_wrong_trajs)

        all_sets = [click_right_ids, click_wrong_ids, crop_right_ids, crop_wrong_ids]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                intersection = all_sets[i] & all_sets[j]
                assert not intersection, f"Overlap detected between set {i} and set {j}: {intersection}"

        ###click_right 作为chosen
        if len(click_right_trajs) != 0:
            dpo_pairs.extend(self.construction_chosen_click_right(
                    click_right_trajs = click_right_trajs,
                    click_wrong_trajs = click_wrong_trajs,
                    crop_right_trajs = crop_right_trajs,
                    crop_wrong_trajs = crop_wrong_trajs,
                    message_and_image_dict = message_and_image_dict,
                    selected_crop_choice = selected_crop_choice
                )
            )
        ### crop_right 作为chosen
        if len(crop_right_trajs) != 0:
            dpo_pairs.extend(self.construction_chosen_crop_right(
                    click_right_trajs = click_right_trajs,
                    click_wrong_trajs = click_wrong_trajs,
                    crop_right_trajs = crop_right_trajs,
                    crop_wrong_trajs = crop_wrong_trajs,
                    message_and_image_dict = message_and_image_dict,
                    selected_crop_choice = selected_crop_choice
                )
            )
        return dpo_pairs
        
    def process_json_line(self, line, out_put_image):

        data = json.loads(line)

        ####选择chosen 的那一条
        selected_crop_choice = self.select_choice(choices=data["choices"])

        ####只是chosen和rejected不同而已，所以把message中相同的部分先建立好，{message, images}
        try:
            message_and_image_dict = self.build_message_and_image(data=data, selected_crop_choice=selected_crop_choice, output_image=out_put_image)
        except Exception as e:
            print(e)
            return None

        if message_and_image_dict == None:
            return None

        ####所有的dpo pairs
        dpo_pairs = self.construction_dpo_data(selected_crop_choice=selected_crop_choice, message_and_image_dict=message_and_image_dict)
        
        return dpo_pairs
        
    def filter_sample(self, lines):
        print("filtering the sample!")
        print("[info] sum of sample: ", len(lines))

        filter_lines = []

        for l in lines:
            item = json.loads(l)

            choices = item["choices"]
            assert len(choices) == 2

            ###过滤掉全错的
            if choices[0]["score"] == 0 and choices[1]["score"] == 0:
                continue
            filter_lines.append(l)
        print("[info] the sum of after filter: ", len(filter_lines))
        return filter_lines

    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res
    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        lines = []
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                # lines = fin.readlines()[:max_workers*2]
                ##过滤出有score不同的
                lines.extend(self.filter_sample(fin.readlines()))
                print("all data: ", len(lines))

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, out_put_image) for chunk in chunks]
        with open(train_data_json, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item_lst in chunk_result:
                    if item_lst is None: continue
                    for item in item_lst:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_second(self, input_data_path: str, for_trible_output_data_path: str, dpo_pair_output_data_path: str):

        pre_dir = os.path.dirname(dpo_pair_output_data_path)
        output_image = os.path.join(pre_dir, "second_image")

        os.makedirs(output_image, exist_ok=True)
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=dpo_pair_output_data_path,
            out_put_image=output_image,
            max_workers=self.max_workers
        )

        type_2sample= {
            "click_rightcrop_right":[],
            "click_rightcrop_wrong":[],
            "click_rightclick_wrong":[],
            "crop_rightclick_wrong":[],
            "crop_rightcrop_wrong":[],
            "crop_rightcrop_right":[]
        }

        with open(dpo_pair_output_data_path, "r") as fin:
            for line in fin:
                item = json.loads(line)
                ttype = item["chosen_type"] + item["rejected_type"]
                type_2sample[ttype].append(item)

        lst = type_2sample["crop_rightcrop_right"]
        with open(for_trible_output_data_path, "w") as fout:
            for item in lst:
                item["for_trible_turn"]["id"] = item["for_trible_turn"]["id"][:-1]
                for t in item["for_trible_turn"]["trajs"]:
                    t["sample_id"] = t["sample_id"][:-1] 
                fout.write(json.dumps(item["for_trible_turn"]) + "\n")



class TribleProcessor(MultiStepDpoProcessor):
    def __init__(self):
        super().__init__()


    def filter_choice(self, selected_crop_choice):
        raw_image = Image.open(selected_crop_choice["img_path"])
        raw_image_size = raw_image.size[0] * raw_image.size[1]
        if box_area(selected_crop_choice["raw_box"][0]) > 0.8 * raw_image_size:
            return False
        if box_area(selected_crop_choice["raw_box"][1]) > 0.8 * box_area(selected_crop_choice["raw_box"][0]):
            return False
        return True
    
    def build_message_and_image(self, data, selected_crop_choice, out_put_image):

        SYS_MSG=self.prompt_builder.build_sys_msg()
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

        first_crop_image, resize_first_crop_image, resize_first_crop_image_url=self.resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][0], out_put_image, data["id"] + "resize_first_crop.png")
        second_crop_image, resize_second_crop_image, resize_second_crop_image_url= self.resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][1], out_put_image, data["id"] + "resize_second_crop.png")

        ###build
        user_msg_1 = self.prompt_builder.build_user_msg_1(data["instruction"], resize_raw_image)
        assistant_msg_1_1 = self.prompt_builder.build_assistant_msg_1(selected_crop_choice["raw_response"][0], selected_crop_choice["raw_box"][0], raw_image, resize_raw_image)
        user_msg_2_1 = self.prompt_builder.bulid_user_msg_2(resize_first_crop_image)
        try:
            assistant_msg_1_2 = self.build_crop_message(selected_crop_choice["raw_response"][1], selected_crop_choice["raw_box"][0], selected_crop_choice["raw_box"][1], first_crop_image, resize_first_crop_image)
        except AssertionError as e:
            print("zero zero zero")
            return None
        user_msg_2_2 = self.prompt_builder.bulid_user_msg_2(resize_second_crop_image)

        try:
            with Image.open(resize_raw_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_raw_image_url} | Error: {e}")
            return None
        
        try:
            with Image.open(resize_first_crop_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_first_crop_image_url} | Error: {e}")
            return None

        try:
            with Image.open(resize_second_crop_image_url) as img:
                img.verify()  # 快速验证图像是否损坏
        except Exception as e:
            print(f"Invalid image: {resize_second_crop_image_url} | Error: {e}")
            return None

        return {
            "message": [SYS_MSG, user_msg_1, assistant_msg_1_1, user_msg_2_1, assistant_msg_1_2, user_msg_2_2],
            "images": [resize_raw_image_url, resize_first_crop_image_url, resize_second_crop_image_url],
            "raw_images": [raw_image, first_crop_image, second_crop_image]
        }

    def build_pair(self, chosen_message, rejected_message ,message_and_image_dict, chosen_type, rejected_type):
        return {
            "message": message_and_image_dict["message"],
            "chosen": chosen_message,
            "rejected": rejected_message,
            "images": message_and_image_dict["images"],
            "chosen_type": chosen_type,
            "rejected_type": rejected_type
        }
    def build_crop_message(self, raw_response, first_crop_box_in_raw, second_crop_box_in_raw,  crop_image: Image.Image, resize_crop_image: Image.Image):
        second_crop_box_in_first = [
            second_crop_box_in_raw[0] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[1] - first_crop_box_in_raw[1],
            second_crop_box_in_raw[2] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[3] - first_crop_box_in_raw[1]
        ]
        assert all(coord > 0 for coord in second_crop_box_in_first)

        ####在这里面转换box到模型处理的分辨率， 基于最近处理的一张图片， raw_box 也是需要基于最近的一张图片
        return self.prompt_builder.build_assistant_msg_1(raw_response=raw_response, raw_box=second_crop_box_in_first, raw_image=crop_image, resize_raw_image=resize_crop_image)

    def build_click_message(self, raw_response:str, click_point_in_raw:list, crop_box_in_raw:list, crop_image: Image.Image, resize_crop_image: Image.Image):
        return self.prompt_builder.build_assistant_msg_2(raw_response=raw_response, click_point_in_raw=click_point_in_raw, crop_box_in_raw=crop_box_in_raw, crop_image=crop_image, resize_crop_image=resize_crop_image)

    def construction_chosen_click_right(self, click_right_trajs, click_wrong_trajs, crop_right_trajs, crop_wrong_trajs, message_and_image_dict, selected_crop_choice):

        dpo_pairs = []

        raw_image = message_and_image_dict["raw_images"][0]

        resize_raw_image_url = message_and_image_dict["images"][0]
        resize_raw_image = Image.open(resize_raw_image_url)

        first_crop_image = message_and_image_dict["raw_images"][1]
        second_crop_image =  message_and_image_dict["raw_images"][2]

        resize_first_crop_image_url = message_and_image_dict["images"][1]
        resize_second_crop_image_url = message_and_image_dict["images"][2]
        resize_first_crop_image = Image.open(resize_first_crop_image_url)
        resize_second_crop_image = Image.open(resize_second_crop_image_url)

        ###将click right当作chosen 构造dpo数据
        rejected_trajs = []
        ####挑选一个离golden answer 中心点更近的
        chosen_click_right = self.select_click_by_distance(trajs=click_right_trajs, bbox= selected_crop_choice["bbox"], flat=True)
        chosen_click_right["chosen_type"] = "click_right"
        chosen_message = self.build_click_message(
            raw_response= chosen_click_right["raw_response"][2], 
            click_point_in_raw= chosen_click_right["raw_point"],
            crop_box_in_raw= chosen_click_right["raw_box"][1],
            crop_image=second_crop_image,
            resize_crop_image=resize_second_crop_image
            )
        if len(click_wrong_trajs) != 0:
            ####挑选一个距离gold answer 中心点更远的
            rejected_click_wrong = self.select_click_by_distance(trajs=click_wrong_trajs, bbox= selected_crop_choice["bbox"], flat=False)
            rejected_click_wrong["rejected_type"] = "click_wrong"
            rejected_trajs.append(rejected_click_wrong)
        if len(crop_right_trajs) != 0:
            ###TODO：随机挑选一个
            rejected_crop_right = crop_right_trajs[0]
            rejected_crop_right["rejected_type"] = "crop_right"
            rejected_trajs.append(rejected_crop_right)

        if len(crop_wrong_trajs) != 0:
            ###TODO: 随机选一个
            rejected_crop_wrong = crop_wrong_trajs[0]
            rejected_crop_wrong["rejected_type"] = "crop_wrong"
            rejected_trajs.append(rejected_crop_wrong)

        for rejected in rejected_trajs:
            if "click" in rejected["rejected_type"]:
                rejected_message= self.build_click_message(
                    raw_response= rejected["raw_response"][2], 
                    click_point_in_raw= rejected["raw_point"],
                    crop_box_in_raw= rejected["raw_box"][1],
                    crop_image=second_crop_image,
                    resize_crop_image=resize_second_crop_image
                )
                dpo_pairs.append(self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_click_right["chosen_type"], rejected_type=rejected["rejected_type"]))

            elif "crop" in rejected["rejected_type"]:
                ####TODO：这里有BUG, 应该将第二步的raw_box, 转化到crop 后的图像的， 然后再转化为resize后的图像的
                try:
                    rejected_message= self.build_crop_message(
                        raw_response= rejected["raw_response"][2],
                        first_crop_box_in_raw= rejected["raw_box"][1],
                        second_crop_box_in_raw= rejected["raw_box"][2],
                        crop_image=second_crop_image,
                        resize_crop_image=resize_second_crop_image
                    )
                except AssertionError as e:
                    continue
                dpo_pairs.append(self.build_pair(chosen_message= chosen_message, rejected_message= rejected_message, message_and_image_dict= message_and_image_dict, chosen_type= chosen_click_right["chosen_type"], rejected_type=rejected["rejected_type"]))

        return dpo_pairs

    def third_type(self, t):
        if len(t["raw_box"]) == 2 and t["raw_point"] != None:
            return "click"
        elif len(t["raw_box"]) >= 3:
            return "crop"
        else:
            raise ValueError(f"Invalid data format")
    def select_click_or_crop_trajs(self, trajs, action_type, bbox, flat:bool):
        assert action_type == "click" or action_type == "crop"
        result = []
        if action_type == "click":
            for i in range(len(trajs)):
                try:
                    if self.third_type(trajs[i]) == action_type and self.click_right(raw_point= trajs[i]["raw_point"], bbox=bbox) == flat:
                        trajs[i]["index"] = i
                        result.append(trajs[i])
                except ValueError as e:
                    print("[second_type_error]")
                
        elif action_type == "crop":
            for i in range(len(trajs)):
                try:
                    if self.third_type(trajs[i]) == action_type and self.crop_right(raw_box= trajs[i]["raw_box"][2], bbox=bbox) == flat:
                        trajs[i]["index"] = i
                        result.append(trajs[i])
                except ValueError as e:
                    print("[second_type_error]")
        return result
    
    def construction_dpo_data(self, selected_crop_choice, message_and_image_dict):

        dpo_pairs = []
        
        click_right_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "click", bbox= selected_crop_choice["bbox"], flat=True)
        click_wrong_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "click", bbox= selected_crop_choice["bbox"], flat=False)
        crop_right_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "crop", bbox= selected_crop_choice["bbox"], flat=True)
        crop_wrong_trajs = self.select_click_or_crop_trajs(trajs= selected_crop_choice["trajs"], action_type= "crop", bbox= selected_crop_choice["bbox"], flat=False)

        ###确保这四个没有重复
        click_right_ids = self.to_id_set(click_right_trajs)
        click_wrong_ids = self.to_id_set(click_wrong_trajs)
        crop_right_ids = self.to_id_set(crop_right_trajs)
        crop_wrong_ids = self.to_id_set(crop_wrong_trajs)

        all_sets = [click_right_ids, click_wrong_ids, crop_right_ids, crop_wrong_ids]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                intersection = all_sets[i] & all_sets[j]
                assert not intersection, f"Overlap detected between set {i} and set {j}: {intersection}"

        ###click_right 作为chosen
        if len(click_right_trajs) != 0:
            dpo_pairs.extend(self.construction_chosen_click_right(
                    click_right_trajs = click_right_trajs,
                    click_wrong_trajs = click_wrong_trajs,
                    crop_right_trajs = crop_right_trajs,
                    crop_wrong_trajs = crop_wrong_trajs,
                    message_and_image_dict = message_and_image_dict,
                    selected_crop_choice = selected_crop_choice
                )
            )
        return dpo_pairs

    def process_json_line(self, line, out_put_image):

        data = json.loads(line)

        ####选择chosen 的那一条
        selected_crop_choice = self.select_choice(choices=data["choices"])

        ###筛掉chosen 的crop没有比前一步的图像的0.8小的
        if not self.filter_choice(selected_crop_choice):
            return None
        ###每条找一个chosen 的， 第三步为click 且正确的

        ####只是chosen和rejected不同而已，所以把message中相同的部分先建立好，{message, images}
        # try:
        message_and_image_dict = self.build_message_and_image(data=data, selected_crop_choice=selected_crop_choice, out_put_image=out_put_image)


        if message_and_image_dict == None:
            return None

        ####所有的dpo pairs
        dpo_pairs = self.construction_dpo_data(selected_crop_choice=selected_crop_choice, message_and_image_dict=message_and_image_dict)
        
        return dpo_pairs

    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res

    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        lines = []
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                # lines = fin.readlines()[:max_workers*2]
                ##过滤出有score不同的
                lines.extend(fin.readlines())
                print("all data: ", len(lines))

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, out_put_image) for chunk in chunks]
        with open(train_data_json, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item_lst in chunk_result:
                    if item_lst is None: continue
                    for item in item_lst:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    def process_trible(self, input_data_path: str, output_data_path: str):

        pre_dir = os.path.dirname(output_data_path)
        output_image = os.path.join(pre_dir, "trible_image")

        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_data_path,
            out_put_image=output_image,
            max_workers=self.max_workers
        )


class FirstCropDpoProcessor(MultiStepDpoProcessor):

    def __init__(self):
        super().__init__()

    def process_json_line(self, line, out_put_image):
        SYS_MSG=self.prompt_builder.build_sys_msg()

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

        ###加上限制条件：   chosen 的 crop 比原图小，0.5
        chosen_box = chosen_choice["raw_box"][0]
        if box_area(chosen_box) > 0.8 * raw_image.size[0] * raw_image.size[1]:
            print("filter_rate")
            print((box_area(chosen_box) / (raw_image.size[0] * raw_image.size[1])) )
            return None
        ##分差大于3
        if rejected_choice["focus"] == False:
            return None

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

    def filter_sample(self, lines):
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

    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res

    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        lines = []
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                # lines = fin.readlines()[:max_workers*2]
                ##过滤出有score不同的
                lines.extend(self.filter_sample(fin.readlines()))

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, out_put_image) for chunk in chunks]
        with open(train_data_json, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item in chunk_result:
                    if item is None: continue
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    def generate_first_crop_dpo_data(self, input_data_path: str, output_data_path: str):

        pre_dir = os.path.dirname(output_data_path)
        output_image = os.path.join(pre_dir, "first_crop_image")
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_data_path,
            out_put_image=output_image,
            max_workers=self.max_workers
        )
       

class SecondCropDpoProcessor(MultiStepDpoProcessor):

    def __init__(self):
        super().__init__()

    def build_message_and_image(self, data, selected_crop_choice, out_put_image):

        SYS_MSG = self.prompt_builder.build_sys_msg()
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

        crop_image, resize_crop_image, resize_crop_image_url=self.resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][0], out_put_image, data["id"] + "resize_crop.png")
        
        ###build
        user_msg_1 = self.prompt_builder.build_user_msg_1(data["instruction"], resize_raw_image)
        assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(selected_crop_choice["raw_response"][0], selected_crop_choice["raw_box"][0], raw_image, resize_raw_image)
        user_msg_2 = self.prompt_builder.bulid_user_msg_2(resize_crop_image)

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

        return {
            "message": [SYS_MSG, user_msg_1, assistant_msg_1, user_msg_2],
            "images": [resize_raw_image_url, resize_crop_image_url],
            "raw_images": [raw_image, crop_image]
        }

    def build_crop_message(self, raw_response, first_crop_box_in_raw, second_crop_box_in_raw,  crop_image: Image.Image, resize_crop_image: Image.Image):
        second_crop_box_in_first = [
            second_crop_box_in_raw[0] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[1] - first_crop_box_in_raw[1],
            second_crop_box_in_raw[2] - first_crop_box_in_raw[0],
            second_crop_box_in_raw[3] - first_crop_box_in_raw[1]
        ]
        for x in second_crop_box_in_first:
            if x < 0: 
                raise ValueError(f"Invalid box value: {x} is less than 0")

        ####在这里面转换box到模型处理的分辨率， 基于最近处理的一张图片， raw_box 也是需要基于最近的一张图片
        return self.prompt_builder.build_assistant_msg_1(raw_response=raw_response, raw_box=second_crop_box_in_first, raw_image=crop_image, resize_raw_image=resize_crop_image)

    def process_json_line(self, line, out_put_image):

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
        ###

        instruction = data["instruction"]
        sorted_choices = sorted(data["choices"], key=lambda x: x["score"])

        rejected_choice = sorted_choices[0]
        chosen_choice = sorted_choices[1]

        ###加上限制条件：   第一步crop 比原图的0.8小
        chosen_box = chosen_choice["raw_box"][0]
        if box_area(chosen_box) > 0.8 * raw_image.size[0] * raw_image.size[1]:
            print("filter_rate")
            print((box_area(chosen_box) / (raw_image.size[0] * raw_image.size[1])) )
            return None

        # chosen 的crop比上一步的0.5小
        if box_area(chosen_choice["raw_box"][1]) > 0.8 * box_area(chosen_choice["raw_box"][0]):
            print("filter_rate")
            print(( box_area(chosen_choice["raw_box"][0]) / box_area(chosen_choice["raw_box"][1])))
            return None
        if chosen_choice["score"] - rejected_choice["score"] < 4:
            return None

        ###IOU小于0.8
        rejected_box = rejected_choice["raw_box"][1]
        chosen_box = chosen_choice["raw_box"][1]
        i_area = insertion_area(rejected_box, chosen_box)
        if i_area / (box_area(rejected_box) + box_area(chosen_box) - i_area) > 0.8:
            return None
        
        message_and_image = self.build_message_and_image(data, chosen_choice, out_put_image)

        # rejected_choice = sorted_choices[0]
        # chosen_choice = sorted_choices[1]

        try:
            chosen_assistant_msg_1 = self.build_crop_message(
                raw_response= chosen_choice["raw_response"][1],
                first_crop_box_in_raw= chosen_choice["raw_box"][0],
                second_crop_box_in_raw= chosen_choice["raw_box"][1],
                crop_image= message_and_image["raw_images"][1],
                resize_crop_image= (Image.open(message_and_image["images"][1]))
            )

            rejected_assistant_msg_1 = self.build_crop_message(
                raw_response= rejected_choice["raw_response"][1],
                first_crop_box_in_raw= rejected_choice["raw_box"][0],
                second_crop_box_in_raw= rejected_choice["raw_box"][1],
                crop_image= message_and_image["raw_images"][1],
                resize_crop_image= (Image.open(message_and_image["images"][1]))
            )
        except ValueError as e:
            print("valueerror")
            return None


        return {
            "message":message_and_image["message"],
            "chosen": chosen_assistant_msg_1,
            "rejected": rejected_assistant_msg_1,
            "images": message_and_image["images"]
        }

    def filter_sample(self, lines):
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
    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res

    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        lines = []
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                # lines = fin.readlines()[:max_workers*2]
                ##过滤出有score不同的
                lines.extend(self.filter_sample(fin.readlines()))

        # lines分成若干块 
        chunk_size = len(lines) // max_workers + 1
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, out_put_image) for chunk in chunks]
        with open(train_data_json, "w", encoding="utf-8") as fout:
            for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
                chunk_result = future.result()
                for item in chunk_result:
                    if item is None: continue
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    def generate_second_crop_dpo_data(self, input_data_path: str, output_data_path: str):
        pre_dir = os.path.dirname(output_data_path)
        output_image = os.path.join(pre_dir, "second_crop_image")
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_data_path,
            out_put_image=output_image,
            max_workers=self.max_workers
        )

class TrainDataProcessor():
    def __init__(self):
        self.first_crop_dpo_processor = FirstCropDpoProcessor()
        self.second_crop_dpo_processor = SecondCropDpoProcessor()
    def generate_second_click_dpo_data(self, input_data_path: str, output_data_path: str):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        def parser_res(res: str):
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
        def filter_box(lst):
            result = []
            for item in lst:
                crop_message= item["message"][2]["content"]
                action, model_box = parser_res(crop_message)
                resize_raw_image_url = item["images"][0]
                resize_raw_image = Image.open(resize_raw_image_url)
                if box_area(model_box) < resize_raw_image.size[0] * resize_raw_image.size[1] * 0.8:
                    result.append(item)
            return result
        

        type_2sample= {
            "click_rightcrop_right":[],
            "click_rightcrop_wrong":[],
            "click_rightclick_wrong":[],
            "crop_rightclick_wrong":[],
            "crop_rightcrop_wrong":[],
            "crop_rightcrop_right":[]
        }
        with open(input_data_path, "r") as fin:
            for line in fin:
                data = json.loads(line)
                new_data = {
                    "message":data["message"],
                    "chosen":data["chosen"],
                    "rejected": data["rejected"],
                    "images":data["images"]
                }
                item = data
                ttype = item["chosen_type"] + item["rejected_type"]
                type_2sample[ttype].append(item)
        filter_lst = filter_box(type_2sample["click_rightclick_wrong"])

        with open(output_data_path, "w") as fout:
            for item in filter_lst:
                new_item = {
                    "message": item["message"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "images": item["images"]
                }
                fout.write(json.dumps(new_item) + "\n")

    def merge(self, input_jsonl_lst:list, output_data_path: str):
        with open(output_data_path, "w") as fout:
            for input_json in input_jsonl_lst:
                with open(input_json, "r") as fin:
                    for line in fin:
                        fout.write(line)

    def transfer_to_train_data(self, score_candidates_by_second_output_path: str, score_candidates_by_trible_output_path: str, second_step_dpo_data:str, train_data_output_path: str):
        
        pre_dir = os.path.dirname(train_data_output_path)
        first_crop_dpo_output_path = os.path.join(pre_dir, "first_crop_dpo_data.jsonl")
        self.first_crop_dpo_processor.generate_first_crop_dpo_data(score_candidates_by_second_output_path, first_crop_dpo_output_path)

        second_crop_dpo_output_path = os.path.join(pre_dir, "second_crop_dpo_data.jsonl")
        self.second_crop_dpo_processor.generate_second_crop_dpo_data(score_candidates_by_trible_output_path, second_crop_dpo_output_path)

        second_click_dpo_output_path = os.path.join(pre_dir, "second_click_dpo_data.jsonl")
        self.generate_second_click_dpo_data(second_step_dpo_data, second_click_dpo_output_path)

        self.merge([first_crop_dpo_output_path, second_crop_dpo_output_path, second_click_dpo_output_path], train_data_output_path)




