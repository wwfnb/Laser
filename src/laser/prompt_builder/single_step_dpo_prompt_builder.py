from PIL import Image
from laser.utils.misc import convert_pil_image_to_base64
import re

class SingleStepDpoPromptBuilder:

    class GeneratePromptBuilder:
        def __init__(self):
            pass

        def build_sys_msg(self):
            sys_str = '''
You are a GUI reasoning agent. Your task is to identify where to interact within a given GUI screenshot to fulfill the user’s instruction. You have access to two tools:

- Crop(left, top, right, bottom): Use this tool to focus on the key area of the screenshot if it is too broad or contains distracting elements.  After the crop, I will send you the cropped image for further steps.
- Click: Use this tool to do a click on the recent image.

Coordinates and crop bounding boxes are in pixels relative to the image.

Response Format Examples:
<think>
<Your step-by-step reasoning explaining why and how you select the crop area>
</think>
<action>
crop(left, top, right, bottom)
</action>
<think>
<Your step-by-step reasoning explaining why and how you click the coordinate>
</think>
<action>
click(x, y)
</action>
'''
            sys_msg = {
                "role": "system",
                "content": sys_str,
            }
            return sys_msg
        

        def bulid_assistant_msg(self, assistant_content:str):
            assistant_msg = {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": assistant_content
                    }
                ]
            }
            return assistant_msg
        def build_user_msg_1(self, instruction: str, resized_raw_image: Image.Image):
            ori_image_width, ori_image_height = resized_raw_image.size
            user_ins = """Instruction: {instruction}
The screenshot below has resolution: {ori_image_width}x{ori_image_height}""".format(instruction= instruction, ori_image_width= ori_image_width, ori_image_height= ori_image_height)

            user_msg_1 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_ins
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + convert_pil_image_to_base64(resized_raw_image)
                        }
                    }
                ]
            }
            return user_msg_1
        
        def bulid_user_msg_2(self, resized_crop_image: Image.Image):
            crop_image_width, crop_image_height = resized_crop_image.size
            user_ins_2 = "The cropped screenshot has resolution: {crop_image_width}x{crop_image_height}\nThis is the cropped screenshot ".format(crop_image_width= crop_image_width, crop_image_height= crop_image_height)
            user_msg_2 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_ins_2
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + convert_pil_image_to_base64(resized_crop_image)
                        }
                    }
                ]
            }
            return user_msg_2
        

    class ProcessPromptBuilder:
        def __init__(self):
            pass

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

        def build_sys_msg(self):
            sys_str = '''
You are a GUI reasoning agent. Your task is to identify where to interact within a given GUI screenshot to fulfill the user’s instruction. You have access to two tools:

- Crop(left, top, right, bottom): Use this tool to focus on the key area of the screenshot if it is too broad or contains distracting elements.  After the crop, I will send you the cropped image for further steps.
- Click: Use this tool to do a click on the recent image.

Coordinates and crop bounding boxes are in pixels relative to the image.

Response Format Examples:
<think>
<Your step-by-step reasoning explaining why and how you select the crop area>
</think>
<action>
crop(left, top, right, bottom)
</action>
<think>
<Your step-by-step reasoning explaining why and how you click the coordinate>
</think>
<action>
click(x, y)
</action>
'''
            sys_msg = {
                "role": "system",
                "content": sys_str,
            }
            return sys_msg
                
        def build_user_msg_1(self, instruction: str, resized_raw_image: Image.Image):
            ori_image_width, ori_image_height = resized_raw_image.size
            user_ins = """Instruction: {instruction}
The screenshot below has resolution: {ori_image_width}x{ori_image_height}\n<image>""".format(instruction= instruction, ori_image_width= ori_image_width, ori_image_height= ori_image_height)

            user_msg_1 = {
                "role": "user",
                "content": user_ins
            }
            return user_msg_1
        
        def build_assistant_msg_1(self, raw_response, raw_box, raw_image: Image.Image, resize_raw_image: Image.Image):

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

            cot, action, para = self.parser_res(raw_response)
            assert action == "crop"
            assert cot != None and cot != ""
            assistant_msg_1 = {
                "role": "assistant",
                "content":  "<think>" + cot + "</think>" + "\n" + "<action>crop({box})</action>".format(box=box_str)
            }
            return assistant_msg_1
        
        def bulid_user_msg_2(self, resized_crop_image: Image.Image):
            crop_image_width, crop_image_height = resized_crop_image.size
            user_msg_2 = {
                "role": "user",
                "content": "The cropped screenshot has resolution: {crop_image_width}x{crop_image_height}\nThis is the cropped screenshot.\n<image>".format(crop_image_width= crop_image_width, crop_image_height= crop_image_height)
            }
            return user_msg_2
        
        def build_assistant_msg_2(self, raw_response:str, click_point_in_raw:list, crop_box_in_raw:list, crop_image: Image.Image, resize_crop_image: Image.Image):

            assert raw_response != "" and "click" in raw_response
            # assert crop_box_in_raw[0] <= click_point_in_raw[0] <= crop_box_in_raw[2] and crop_box_in_raw[1] <= click_point_in_raw[1] <= crop_box_in_raw[3]

            click_point_in_crop = [click_point_in_raw[0] - crop_box_in_raw[0], click_point_in_raw[1] - crop_box_in_raw[1]]
            click_point_in_resize = [click_point_in_crop[0] / crop_image.size[0] * resize_crop_image.size[0], click_point_in_crop[1] / crop_image.size[1] * resize_crop_image.size[1]]

            click_point_in_resize = [round(x) for x in click_point_in_resize]
            point = ",".join([str(x) for x in click_point_in_resize])

            click_cot, action, para = self.parser_res(raw_response)
            assert action == "click"
            assert click_cot != None and click_cot != ""

            assistant_msg_2={
                "role": "assistant",
                "content": ("<think>{click_cot}</think>" + "\n" + "<action>click({point})</action>").format(click_cot=click_cot,point=point)
            }
            return assistant_msg_2