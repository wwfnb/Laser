import json
import os
from pixel_grounding.misc import generate_id, get_date, load_and_resize_image, crop_image_by_box, resize_image
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import threading
import re
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import math
from pixel_grounding.build_prompt import build_sys_msg_crop_cot_click_cot

def trans_box_to_box(self, box, src_image, ob_image):
    return [box[0] / src_image.size[0] * ob_image.size[0], 
            box[1] / src_image.size[1] * ob_image.size[1], 
            box[2] / src_image.size[0] * ob_image.size[0], 
            box[3] / src_image.size[1] * ob_image.size[1]]

def trans_point_to_point(self, point, src_image, ob_image):
    return [point[0] / src_image.size[0] * ob_image.size[0], 
            point[1] / src_image.size[1] * ob_image.size[1]]

def parser_res(res: str):
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


# def build_sys_msg():
#     sys_str = '''
# You are a GUI reasoning agent. Your task is to identify where to interact within a given GUI screenshot to fulfill the user’s instruction. You have access to two tools:

# - Crop(left, top, right, bottom): Use this tool to focus on the key area of the screenshot if it is too broad or contains distracting elements.  After the crop, I will send you the cropped image for further steps.
# - Click: Use this tool to do a click on the recent image.

# Coordinates and crop bounding boxes are in pixels relative to the image.

# Response Format Examples:
# <think>
# <Your step-by-step reasoning explaining why and how you select the crop area>
# </think>
# <action>
# crop(left, top, right, bottom)
# </action>
# <think>
# <Your step-by-step reasoning explaining why and how you click the coordinate>
# </think>
# <action>
# click(x, y)
# </action>
# '''
#     sys_msg = {
#         "role": "system",
#         "content": sys_str,
#     }
#     return sys_msg

SYS_MSG = build_sys_msg_crop_cot_click_cot()

def build_user_msg_1(instruction: str, resized_raw_image: Image.Image):
    ori_image_width, ori_image_height = resized_raw_image.size
    user_ins = """Instruction: {instruction}
The screenshot below has resolution: {ori_image_width}x{ori_image_height}\n<image>""".format(instruction= instruction, ori_image_width= ori_image_width, ori_image_height= ori_image_height)

    user_msg_1 = {
        "role": "user",
        "content": user_ins
    }
    return user_msg_1

####在这里面转换box到模型处理的分辨率
def build_assistant_msg_1(raw_response, raw_box, raw_image: Image.Image, resize_raw_image: Image.Image):

    assert raw_response != "" and "crop" in raw_response
    raw_width, raw_height = raw_image.size
    resize_width, resize_height = resize_raw_image.size
    assert len(raw_box) == 4
    resize_box = [
        raw_box[0] / raw_width * resize_width,
        raw_box[1] / raw_height * resize_height,
        raw_box[2] / raw_width * resize_width,
        raw_box[3] / raw_height * resize_height,
    ]
    resize_box = [round(x) for x in resize_box]
    box_str = ",".join([str(x) for x in resize_box])

    cot, action, para = parser_res(raw_response)
    assert action == "crop"
    assert cot != None and cot != ""
    assistant_msg_1 = {
        "role": "assistant",
        "content":  "<think>" + cot + "</think>" + "\n" + "<action>crop({box})</action>".format(box=box_str)
    }
    return assistant_msg_1
####在这里面转换box到模型处理的分辨率
# def build_assistant_msg_1(crop_response):
#     assistant_msg_1 = {
#         "role": "assistant",
#         "content":  crop_response
#     }
#     return assistant_msg_1


def bulid_user_msg_2(resized_crop_image: Image.Image):
    crop_image_width, crop_image_height = resized_crop_image.size
    user_msg_2 = {
        "role": "user",
        "content": "The cropped screenshot has resolution: {crop_image_width}x{crop_image_height}\nThis is the cropped screenshot.\n<image>".format(crop_image_width= crop_image_width, crop_image_height= crop_image_height)
    }
    return user_msg_2
from pixel_grounding.misc import box_contains
##在这里面转换point到模型处理的分辨率
##需要把基于原图的click转换为基于resize_crop_image的
#raw->crop->crop_resize
def build_assistant_msg_2(raw_response:str, click_point_in_raw:list, crop_box_in_raw:list, crop_image: Image.Image, resize_crop_image: Image.Image):

    assert raw_response != "" and "click" in raw_response
    # assert crop_box_in_raw[0] <= click_point_in_raw[0] <= crop_box_in_raw[2] and crop_box_in_raw[1] <= click_point_in_raw[1] <= crop_box_in_raw[3]

    click_point_in_crop = [click_point_in_raw[0] - crop_box_in_raw[0], click_point_in_raw[1] - crop_box_in_raw[1]]
    click_point_in_resize = [click_point_in_crop[0] / crop_image.size[0] * resize_crop_image.size[0], click_point_in_crop[1] / crop_image.size[1] * resize_crop_image.size[1]]

    click_point_in_resize = [round(x) for x in click_point_in_resize]
    point = ",".join([str(x) for x in click_point_in_resize])

    click_cot, action, para = parser_res(raw_response)
    assert action == "click"
    assert click_cot != None and click_cot != ""

    assistant_msg_2={
        "role": "assistant",
        "content": ("<think>{click_cot}</think>" + "\n" + "<action>click({point})</action>").format(click_cot=click_cot,point=point)
    }
    return assistant_msg_2

# def build_assistant_msg_2(click_response):
#     assistant_msg_2={
#         "role": "assistant",
#         "content": click_response
#     }
#     return assistant_msg_2

clear_cot_num = 0
def clear_cot(cot: str):
    global clear_cot_num
    # 删除cot中的 <box> ..</box>
    clear_cot = re.sub(r'<box>.*?</box>', '', cot)
    if len(clear_cot) != len(cot):
        clear_cot_num += 1
    return clear_cot

def save_image(resize_image: Image.Image, output_image_folder: str, image_filename: str):
    image_path = os.path.join(output_image_folder, image_filename)
    if not os.path.exists(image_path):
        resize_image.save(image_path)
    assert os.path.exists(image_path)
    return image_path
###返回resized 后的crop图像
def resize_crop_and_save_image(raw_image: Image.Image, box: list, output_image_folder:str, image_filename: str):
    crop_image = crop_image_by_box(raw_image, box)
    resize_crop_image_height, resize_crop_image_width = smart_resize(
        crop_image.height,
        crop_image.width,
        factor=28,
        min_pixels=3136,
        max_pixels=MAX_PIXELS,
    )
    resize_crop_image = crop_image.resize((resize_crop_image_width, resize_crop_image_height))
    resize_crop_image_url = save_image(resize_crop_image, output_image_folder, image_filename)
    return crop_image, resize_crop_image, resize_crop_image_url

key_error_count = 0



count_ins_correct = 0
count = 0

image_count = 1


count_len_error = 0

MODEL_NAME = "/public/home/ljt/hqp/models/qwen2.5_vl_7b"
MAX_PIXELS = 802816 # set here !    1024 * 28 * 28

input_json_list = ["/public/home/ljt/hqp/GUI-grounding/generate/0715_mutil_turn_dpo.jsonl"]
train_data_json = f"/public/home/ljt/hqp/GUI-grounding/0731_ablation_dpo_train_data/0731_mutil_turn_dpo_crop_click_{MAX_PIXELS}.jsonl"
out_put_image =  f"/public/home/ljt/hqp/GUI-grounding/0731_ablation_dpo_train_data/0731_dpo_just_crop_image_{MAX_PIXELS}_without_iou"     ### raw用的之前resized好的图, crop_resize 这里处理保存
os.makedirs(out_put_image, exist_ok=True)

def process_json_line(line, out_put_image):
    global count_ins_correct, count, key_error_count, count_len_error

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
        max_pixels=MAX_PIXELS,
    )
    resize_raw_image_url = os.path.join(out_put_image, data["id"] + "resize_raw.png")
    if not os.path.exists(resize_raw_image_url):
        resize_raw_image.save(resize_raw_image_url)
    
    ###crop图片
    crop_image, resize_crop_image, resize_crop_image_url=resize_crop_and_save_image(raw_image, selected_crop_choice["raw_box"][0], out_put_image, data["id"] + "resize_crop.png")
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
    user_msg_1 = build_user_msg_1(instruction, resize_raw_image)
    assistant_msg_1 = build_assistant_msg_1(selected_crop_choice["raw_response"][0], selected_crop_choice["raw_box"][0], raw_image, resize_raw_image)
    user_msg_2 = bulid_user_msg_2(resize_crop_image)
    try:
        chosen_assistant_msg_2 = build_assistant_msg_2(chosen_traj["raw_response"][1], chosen_traj["raw_point"], chosen_traj["raw_box"][0], crop_image, resize_crop_image)
        rejected_assistant_msg_2 = build_assistant_msg_2(rejected_traj["raw_response"][1], rejected_traj["raw_point"], rejected_traj["raw_box"][0], crop_image, resize_crop_image)
    except IndexError as e:
        count_len_error += 1
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

from tqdm import tqdm

def filter_sample(lines):
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



def process_chunk(chunk: list, out_put_image):
    res = []
    for line in tqdm(chunk):
        res.append(process_json_line(line, out_put_image))
    return res
def process_all_jsons(input_json_list, train_data_json, out_put_image, max_workers=8):
    os.makedirs(out_put_image, exist_ok=True)
    for input_json in input_json_list:
        with open(input_json, 'r', encoding='utf-8') as fin:
            # lines = fin.readlines()[:max_workers*2]
            ##过滤出有score不同的
            lines = filter_sample(fin.readlines())

    # lines分成若干块 
    chunk_size = len(lines) // max_workers + 1
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, out_put_image) for chunk in chunks]
    with open(train_data_json, "w", encoding="utf-8") as fout:
        for future in tqdm(as_completed(futures), total=len(futures), desc="writing"):
            chunk_result = future.result()
            for item in chunk_result:
                if item is None: continue
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

process_all_jsons(
    input_json_list=input_json_list,
    train_data_json=train_data_json,
    out_put_image=out_put_image,
    max_workers=40
)
                

print(count_len_error)