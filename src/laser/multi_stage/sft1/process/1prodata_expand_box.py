import re
import json
import os
import copy
from tqdm import tqdm

input_json = ["/public/home/ljt/hqp/GUI-grounding/generate/0627cot-generate-without-example/grounding-R1-cot_focus_07.jsonl", "/public/home/ljt/hqp/GUI-grounding/generate/0627cot-generate-without-example/grounding-R1-cot_focus.jsonl"]

output_json_data = "/public/home/ljt/hqp/GUI-grounding/generate/cot/grounding-R1-cot_focus_expand_limit_box.jsonl"

I_RATIO = 0.75
EXPANED_RATIO = 0.2
count_expanded_box_data = 0
count_out_of_screen = 0

MAX_SIZE = 0.5
count_zero = 0

set_id = set()


def insertion_area(box1: list, box2: list) -> float:
    """
    计算两个矩形框的相交区域面积。
    
    参数：
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]
        
    返回：
        float: 相交区域面积
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有交集

    return (x_right - x_left) * (y_bottom - y_top)

def adjust_box(box: list, width, height) -> list:
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return [x1, y1, x2, y2]

def box_area(box: list) -> float:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def transfer_box(box: list, now_width, now_height, dest_width, dest_height) -> list:
    return [box[0] * dest_width / now_width, box[1] * dest_height / now_height, box[2] * dest_width / now_width, box[3] * dest_height / now_height]

def box_contains(box1: list, box2: list) -> bool:
    return (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3])

def box_valid(box: list) -> bool:
    return len(box) == 4 and (box[0] < box[2] and box[1] < box[3])


lines = []

for file in input_json:
    with open(file, 'r', encoding='utf-8') as fin:
        lines.extend(fin.readlines())

has_correct_box_num = 0
with open(output_json_data, "w", encoding="utf-8") as fout:
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

        ### box、cot and cot_correct
        cot_list = []
        box_list = []
        cot_correct = []
        # model_box_list = []

        grounded_bbox = item["coordinate"]


        for res in response:
            think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
            think_content = think_match.group(1) if think_match else None

            if think_content == None:
                continue
            
            #### 
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

            ### 框超出屏幕   ##TODO: 修改框超出屏幕的逻辑， 有些框只是超出一点而已，但是他的cot是有用的，并且也能狂刀焦点区域，缩小到去掉超出屏幕的部分
            if not box_valid(box): continue
            
            


            ###截掉超出屏幕的
            box = adjust_box(box, width= run_image_width, height= run_image_height)
            if not box_valid(box): continue

            try:
                assert insertion_area(box, [0, 0, run_image_width, run_image_height]) <= (box_area(box)+1)
            except Exception:
                import pdb; pdb.set_trace()
                ...

            # model_box = box.copy() 
            box = transfer_box(box, now_width= run_image_width, now_height= run_image_height, dest_width= raw_image_width, dest_height= raw_image_height)

            # import pdb;pdb.set_trace()

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
        fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"has_correct_box_num: {has_correct_box_num}")
print(f"out of screen{count_out_of_screen}")
print(f"expanded_box_data: {count_expanded_box_data}")
