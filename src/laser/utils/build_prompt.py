# 修改一下example中think的内容
# 试一下把example放到user的部分
from io import BytesIO
from laser.utils.misc import convert_pil_image_to_base64
from PIL import Image

#############以下是single_step_sft的##############
############single_step_sft:generate#############################
##### single_step_sft_crop_msg ####
def get_single_step_sft_crop_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI reasoning agent."
                },
                {
                    "type": "text",
                    "text": """
You are given an  **instruction** and a **screenshot** of a user interface.  
Your task is to **think step by step** about which area in the screenshot is most relevant to the instruction.  
Then, output the result in the following format:

<think>your step-by-step reasoning here</think>  
<box>(x1,y1,x2,y2)</box>

The <think> section should explain why you focus on that area in relation to the instruction.  
The <box> should define a bounding box that covers the **focus region**: a sufficiently large area likely to contain the target interface element(s) for completing the instruction. It doesn't need to be pixel-precise, but it should tightly bound the visually relevant region.

- Make sure the box includes enough visual context for reasoning, typically greater 400x400 pixels in size.
- Center the box on the relevant region but include sufficient margin to avoid cropping important contextual cues (e.g., labels, icons, nearby buttons).
"""
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Instruction: " + instruction + "\n"
                },
                {
                    "type": "text",
                    "text": "The screenshot below has resolution: {{screen_width}}x{{screen_height}}\n".replace("{{screen_width}}", str(screen_width)).replace("{{screen_height}}", str(screen_height))
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
            ]
        }
    ]

#### single_step_sft_click_msg ####
def get_single_step_sft_click_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI reasoning agent."
                },
                {
                    "type": "text",
                    "text": """
You are given an  **instruction** and a **screenshot** of a user interface.  
Your task is to **think step by step** about which point to click in the screenshot to complish the instruction.
Then, output the result in the following format:
<think>your step-by-step reasoning here</think>  
<action>click(x,y)</action>

The <think> section should include the reason why you click on a certain point, such as an icon or text's label, and how it relates to the instruction.
"""
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Instruction: " + instruction + "\n"
                },
                {
                    "type": "text",
                    "text": "The screenshot below has resolution: {{screen_width}}x{{screen_height}}\n".replace("{{screen_width}}", str(screen_width)).replace("{{screen_height}}", str(screen_height))
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
            ]
        }
    ]

#############single_step_sft:process###################


def build_single_step_sft_sys_msg():
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




########################
# build system prompt
def build_sys_msg_crop_cot_click_cot():
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

def build_sys_msg_crop_cot_click_no_cot():
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
<think></think>
<action>
click(x, y)
</action>
'''
    sys_msg = {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text":  sys_str
            }
        ]
    }
    return sys_msg

# build user prompt
def build_user_msg_1(instruction: str, resized_raw_image: Image.Image):
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
                    "url": convert_pil_image_to_base64(resized_raw_image)
                }
            }
        ]
    }
    return user_msg_1

def bulid_user_msg_2(resized_crop_image: Image.Image):
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
                    "url": convert_pil_image_to_base64(resized_crop_image)
                }
            }
        ]
    }
    return user_msg_2

# build assistant prompt
def bulid_assistant_msg(assistant_content:str):
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