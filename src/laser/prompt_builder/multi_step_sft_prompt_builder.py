from PIL import Image
from laser.utils.misc import convert_pil_image_to_base64
import re
from laser.prompt_builder.single_step_dpo_prompt_builder import SingleStepDpoPromptBuilder

class MultiStepSftPromptBuilder:
    class GeneratePromptBuilder(SingleStepDpoPromptBuilder.GeneratePromptBuilder):
        def __init__(self):
            super().__init__()


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

        def build_assistant_msg_1(self, crop_response):
            assistant_msg_1 = {
                "role": "assistant",
                "content":  crop_response
            }
            return assistant_msg_1
        
        def bulid_user_msg_2(self, resized_crop_image: Image.Image):
            crop_image_width, crop_image_height = resized_crop_image.size
            user_msg_2 = {
                "role": "user",
                "content": "The cropped screenshot has resolution: {crop_image_width}x{crop_image_height}\nThis is the cropped screenshot.\n<image>".format(crop_image_width= crop_image_width, crop_image_height= crop_image_height)
            }
            return user_msg_2
        
        def build_assistant_msg_2(self, click_response):
            assistant_msg_2={
                "role": "assistant",
                "content": click_response
            }
            return assistant_msg_2
        
