import yaml
from laser.utils.misc import load_and_resize_image, insertion_area, adjust_box, box_area, transfer_box, box_contains, box_valid
from tqdm import tqdm
import json
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import math
from laser.prompt_builder.single_step_sft_prompt_builder import SingleStepSftPromptBuilder
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from pixel_grounding.misc import load_and_resize_image, crop_image_by_box, resize_image


class SingleStepSftProcessor:
    def __init__(self):
        with open("src/laser/configs/single_step_sft.yaml", "r") as fin:
            self.cfg = yaml.safe_load(fin)

        self.prompt_builder = SingleStepSftPromptBuilder.ProcessPromptBuilder()
        self.max_pixels = self.cfg["process"]["reject_sampling_click"]["max_pixels"]
        self.max_workers = self.cfg["process"]["reject_sampling_click"]["max_workers"]

    def clear_cot(self, cot: str):
        # 删除cot中的 <box> ..</box>
        clear_cot = re.sub(r'<box>.*?</box>', '', cot)
        return clear_cot
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
                para = get_para(action_text)
        # print(f"action: {action}, para: {para}")
        return think, action, para

    def select_click_response(self, click_response_lst: list[str], grounded_box:list, raw_crop_image_size:list, resize_crop_image_size:list, crop_box:list):

        result_lst = []

        centre_point = [(grounded_box[0] + grounded_box[2]) / 2, (grounded_box[1] + grounded_box[3]) / 2]

        for res in click_response_lst:
            think, action, para = self.parser_res(res)
            if None in (think, action, para) or "" in (think, action, para) or len(para) != 2:
                continue
            click_point_in_crop = [para[0] / resize_crop_image_size[0] * raw_crop_image_size[0], para[1] / resize_crop_image_size[1] * raw_crop_image_size[1]]
            click_point_in_raw = [click_point_in_crop[0] + crop_box[0], click_point_in_crop[1] + crop_box[1]]
            if grounded_box[0] <= click_point_in_raw[0] <= grounded_box[2] and grounded_box[1] <= click_point_in_raw[1] <= grounded_box[3]:
                distance = math.sqrt((centre_point[0] - click_point_in_raw[0]) ** 2 + (centre_point[1] - click_point_in_raw[1]) ** 2)
                result_lst.append([think, "click", click_point_in_raw, distance])
        if len(result_lst) == 0:
            return None, None, None
        sorted_result = sorted(result_lst, key=lambda x: x[3])
        return sorted_result[0][:-1]
    
    
    def process_json_line(self, line, out_put_image):

        SYS_MSG = self.prompt_builder.build_sys_msg()

        data = json.loads(line)
        idx = data["id"]
        instruction = data["instruction"]
        raw_image_url = data["raw_image_url"]
        grounded_box = data["bbox"]
        crop_box_in_raw = data["crop_box"]
        crop_cot = data["cot"]
        click_response_lst = data["running"]["response"]

        raw_crop_image_size = [data["running"]["run_image_info"]["raw_width"], data["running"]["run_image_info"]["raw_height"]]
        resize_crop_image_size = [data["running"]["run_image_info"]["width"], data["running"]["run_image_info"]["height"]]

        click_cot, click_action, click_point_in_raw = self.select_click_response(click_response_lst, grounded_box, raw_crop_image_size, resize_crop_image_size, crop_box_in_raw)
        
        if None in (click_cot, click_action, click_point_in_raw):
            return None
    
        ###build user_msg_1
        raw_image, resize_raw_image = load_and_resize_image(
            image_url=raw_image_url,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_raw_image_url = os.path.join(out_put_image, idx+"resize_raw.png")
        resize_raw_image.save(resize_raw_image_url)
        user_msg_1 = self.prompt_builder.build_user_msg_1(instruction, resize_raw_image)

        ###assistant_msg_1
        crop_image = crop_image_by_box(raw_image, crop_box_in_raw)
        resize_crop_image = resize_image(
            crop_image,
            factor=28,
            min_pixels=3136,
            max_pixels=self.max_pixels,
        )
        resize_crop_image_url = os.path.join(out_put_image, idx+"resize_crop.png")
        resize_crop_image.save(resize_crop_image_url)
        assistant_msg_1 = self.prompt_builder.build_assistant_msg_1(self.clear_cot(crop_cot), crop_box_in_raw, raw_image, resize_raw_image)
        ###user_msg_2

        user_msg_2 = self.prompt_builder.bulid_user_msg_2(resize_crop_image)
        ###assistant_msg_2
        assert len(click_point_in_raw) == 2 and click_cot != ""
        assistant_msg_2 = self.prompt_builder.build_assistant_msg_2(click_point_in_raw, crop_box_in_raw, self.clear_cot(click_cot), crop_image, resize_crop_image)


        assert os.path.exists(resize_raw_image_url) and os.path.exists(resize_crop_image_url)
        return {
            "message": [SYS_MSG, user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2],
            "images": [resize_raw_image_url, resize_crop_image_url]
        }
    

    def reject_sampling_crop(self, input_data_path: str, output_data_path: str, output_image:str):

        def get_to_generate(data, output_image):
            instruction = data["instruction"]
            image_url = data["image_url"]
            idx = data["id"]

            try:
                # raw_screen_width = data['running']["run_image_info"]["raw_width"]
                # raw_screen_height = data['running']["run_image_info"]["raw_height"]
                cot_correct = data["running"]["cot_correct"]
                cots = data['running']["cots"]
                boxes = data['running']["boxes"]
                bbox = data["coordinate"]
            except KeyError as e:
                print(f"KeyError: {e}")
                return None

            if any(cot_correct):
                for i in range(len(cot_correct)):

                    if cot_correct[i]:
                        crop_box = boxes[i]
                        raw_image = Image.open(image_url).convert('RGB')
                        crop_image = crop_image_by_box(raw_image, crop_box)
                        crop_image_url = os.path.join(output_image, idx + "_crop.png")
                        crop_image.save(crop_image_url)
                        output_json = {
                            "id": idx,
                            "bbox": bbox,
                            "instruction": instruction, 
                            "crop_image_url": crop_image_url,
                            "raw_image_url": image_url,
                            "crop_box": crop_box,
                            "cot": cots[i]
                        }
                        return output_json
            return None


        I_RATIO = self.cfg["process"]["reject_sampling_crop"]["I_RATIO"]
        EXPANED_RATIO = self.cfg["process"]["reject_sampling_crop"]["EXPANED_RATIO"]
        MAX_SIZE = self.cfg["process"]["reject_sampling_crop"]["MAX_SIZE"]

        count_expanded_box_data = 0
        count_zero = 0
        has_correct_box_num = 0
        lines = []
        set_id = set()
        # for file in input_data_path:
        with open(input_data_path, 'r', encoding='utf-8') as fin:
            lines.extend(fin.readlines())

        # if os.path.exists(output_data_path):
        #     raise ValueError("output_data_path already exists")

        with open(output_data_path, "w", encoding="utf-8") as fout:
            for line in tqdm(lines, desc="Processing"):
                item = json.loads(line)

                iid = item["id"]
                if iid in set_id:
                    continue
                set_id.add(iid)
                try:
                    run_image_width = item["running"]["run_image_info"]["width"]
                    run_image_height = item["running"]["run_image_info"]["height"]
                    raw_image_width = item["running"]["run_image_info"]["raw_width"]
                    raw_image_height = item["running"]["run_image_info"]["raw_height"]
                except KeyError as e:
                    continue
                response = item["running"]["response"]
                cot_list = []
                box_list = []
                cot_correct = []
                grounded_bbox = item["coordinate"]

                for res in response:
                    think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
                    think_content = think_match.group(1) if think_match else None

                    if think_content == None:
                        continue
                    
                    #### 1.res 没有以</box>结尾， 给其添加上去
                    if not res.endswith("</box>"):
                        res += "</box>"

                    box_match = re.search(r"<box>(.*?)</box>", res, re.DOTALL)
                    box_content = box_match.group(1) if box_match else None
                    if box_content== None:
                        continue

                    box = list(map(float, re.findall(r"\d+\.?\d*", box_content)))
                    #### 2.res 中有多个box， 取第一个
                    if len(box) > 4:
                        box = box[:4]
                    if not box_valid(box): continue
                    
                    ###截掉超出屏幕的
                    box = adjust_box(box, width= run_image_width, height= run_image_height)
                    if not box_valid(box): continue

                    try:
                        assert insertion_area(box, [0, 0, run_image_width, run_image_height]) <= (box_area(box)+1)
                    except Exception:
                        import pdb; pdb.set_trace()

                    box = transfer_box(box, now_width= run_image_width, now_height= run_image_height, dest_width= raw_image_width, dest_height= raw_image_height)

                    ori_box = box.copy()
                    # model_box_list.append(model_box)
                    if box_contains(box, grounded_bbox):
                        if box_area(box) > (raw_image_width * raw_image_height) * MAX_SIZE: 
                            continue
                        # import pdb; pdb.set_trace()
                        cot_list.append(think_content)
                        cot_correct.append(True)
                        box_list.append(box)
                    else:
                        ####扩大框逻辑
                        ###如果预测框包含了grounded box 的0.75， 则扩大基于预测框的0.2
                        i_area = insertion_area(box, grounded_bbox)
                        try:
                            if (i_area / box_area(grounded_bbox)) >= I_RATIO:
                                ####哪边没包含，扩大哪边基于预测框的0.2
                                expand_width = (box[2] - box[0]) * EXPANED_RATIO
                                expand_height = (box[3] - box[1]) * EXPANED_RATIO
                                if box[0] >=  grounded_bbox[0]:
                                    box[0] = box[0] - expand_width
                                if box[1] >=  grounded_bbox[1]:
                                    box[1] = box[1] - expand_height
                                if box[2] <=  grounded_bbox[2]:
                                    box[2] = box[2] + expand_width
                                if box[3] <=  grounded_bbox[3]:
                                    box[3] = box[3] + expand_height

                                ###截掉超出屏幕的
                                box = adjust_box(box, width= raw_image_width, height= raw_image_height)
                                if not box_valid(box): continue

                                if box_contains(box, grounded_bbox):
                                    if box_area(box) > (raw_image_width * raw_image_height) * MAX_SIZE:
                                        continue
                                    else:
                                        cot_list.append(think_content)
                                        cot_correct.append(True)
                                        box_list.append(box)
                                        count_expanded_box_data += 1
                        except ZeroDivisionError as e:
                            count_zero += 1
                            continue

                assert len(box_list) == len(cot_correct) == len(cot_list)
                if any(cot_correct): 
                    has_correct_box_num += 1
                output_data = copy.deepcopy(item)
                # output_data['running']['model_boxes'] = model_box_list
                output_data['running']['cots'] =  cot_list
                output_data['running']['boxes'] = box_list
                output_data['running']['cot_correct'] = cot_correct
                output_data = get_to_generate(output_data, output_image)
                if output_data:
                    fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

        print(f"has_correct_box_num: {has_correct_box_num}")
        print(f"count_zero: {count_zero}")
        print(f"count_expanded_box_data: {count_expanded_box_data}")
    def process_chunk(self, chunk: list, out_put_image):
        res = []
        for line in tqdm(chunk):
            res.append(self.process_json_line(line, out_put_image))
        return res
    def process_all_jsons(self, input_json_list, train_data_json, out_put_image, max_workers=8):
        os.makedirs(out_put_image, exist_ok=True)
        for input_json in input_json_list:
            with open(input_json, 'r', encoding='utf-8') as fin:
                # lines = fin.readlines()[:max_workers*2]
                lines = fin.readlines()
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
    def reject_sampling_click(self, input_data_path: str, output_data_path: str):



        # if os.path.exists(output_data_path):
        #     raise Exception("output_data_path exists")
        pre_dir = os.path.dirname(output_data_path)
        out_put_image = os.path.join(pre_dir, "single_step_sft_resize_image")
        self.process_all_jsons(
            input_json_list=[input_data_path],
            train_data_json=output_data_path,
            out_put_image=out_put_image,
            max_workers=self.max_workers
        )