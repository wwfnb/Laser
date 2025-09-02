from PIL import Image
from io import BytesIO
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import os
import ray
import atexit

DEBUG_FLAG = (True if os.environ.get('DEBUG_FLAG', 'false').lower() == 'true' else False)


def setup_ray(WORKER_NODE):
    if not ray.is_initialized():
        print(f"ray is not initialized, initializing...")
        ray.init(
        num_cpus=16,
        num_gpus=WORKER_NODE,
        address="local"      # 禁止自动连接外部
        )
    def cleanup_ray():
        if ray.is_initialized():
            ray.shutdown()
            print("closing ray...")
    atexit.register(cleanup_ray)
def get_tensor_parallel_size():
    """
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 2  # 如果没有GPU，默认返回2
        
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return 2
        
        # 检查第一个GPU的信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
        
        # 如果是3090或者显存小于30GB，设置为2
        if "3090" in gpu_name or gpu_memory < 30:
            return 2
        return 1
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return 2  # 出错时默认返回2
TENSOR_PARALLEL_SIZE= get_tensor_parallel_size()

import mimetypes, base64
def image_path2openai_base64(image_path: str) -> str:
    """
    Convert image path to base64 string
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode('utf-8')
    mime_type, _ = mimetypes.guess_type(image_path)
    return f"data:{mime_type};base64,{base64_str}"

def convert_pil_image_to_base64(image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

import uuid
def generate_id():
    """
    Generate a unique ID 
    """
    return str(uuid.uuid4())
import time
def get_date():
    """
    Get current date
    """
    return time.strftime("%m-%d", time.localtime())


def load_and_resize_image(image_url: str, factor, min_pixels, max_pixels):
    assert os.path.exists(image_url) and os.path.isfile(image_url), f"Invalid input image_url[{image_url}]"
    image = Image.open(image_url).convert('RGB')
    # Calculate the real image size sent into the model
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        # factor: 28 for qwen2.5 vl
        factor=factor, 
        min_pixels=min_pixels,
        # max_pixels=self.processor.image_processor.max_pixels,
        max_pixels=max_pixels,
    )
    print("Resized image size: {}x{}".format(resized_width, resized_height))
    resized_image = image.resize((resized_width, resized_height))

    return image, resized_image

def resize_image(image: Image.Image, factor, min_pixels, max_pixels):
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image object, but got {type(image)}")
    # Calculate the real image size sent into the model
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        # factor: 28 for qwen2.5 vl
        factor=factor, 
        min_pixels=min_pixels,
        # max_pixels=self.processor.image_processor.max_pixels,
        max_pixels=max_pixels,
    )
    # print("Resized image size: {}x{}".format(resized_width, resized_height))
    resized_image = image.resize((resized_width, resized_height))

    return resized_image


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

def extend_box(box: list)-> list:
    x1, y1, x2, y2 = box
    min_size = 50
    width = x2 - x1
    height = y2 - y1
    assert width > 0 and height > 0
    extend_width = min_size - width
    extend_height = min_size - height
    if extend_width > 0:
        if x1 - extend_width  <= 0:
            x2 += extend_width 
        else:
            x1 -= extend_width 
    if extend_height > 0:
        if y1 - extend_height <= 0:
            y2 += extend_height
        else:
            y1 -= extend_height
    return [x1, y1, x2, y2]

from PIL import Image

def smart_extend_box(box: list, image: Image.Image, min_size: int = 50) -> list:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    assert width > 0 and height > 0

    img_w, img_h = image.size

    # extend width
    if width < min_size:
        extend = min_size - width
        half = extend // 2
        x1 = max(0, x1 - half)
        x2 = min(img_w, x2 + (extend - (x1 - max(0, x1 - half))))
        if x2 - x1 < min_size:  # extend left
            x1 = max(0, x2 - min_size)
        if x2 - x1 < min_size:  # extend right
            x2 = min(img_w, x1 + min_size)
    # limit x1, x2
    x1 = max(0, x1)
    x2 = min(img_w, x2)
    # extend height
    if height < min_size:
        extend = min_size - height
        half = extend // 2
        y1 = max(0, y1 - half)
        y2 = min(img_h, y2 + (extend - (y1 - max(0, y1 - half))))
        if y2 - y1 < min_size:  # extend top
            y1 = max(0, y2 - min_size)
        if y2 - y1 < min_size:  # extend bottom
            y2 = min(img_h, y1 + min_size)
    # limit y1, y2
    y1 = max(0, y1)
    y2 = min(img_h, y2)

    return [x1, y1, x2, y2]


def crop_image_by_box(image: Image.Image, box: list) -> Image.Image:

    cropped = image.crop(box)  # box 格式：[left, top, right, bottom]

    return cropped


import signal
from contextlib import contextmanager
from tqdm import tqdm

# 定义超时异常类
class TimeoutException(Exception):
    pass

# 超时上下文管理器
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # 设置信号处理
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)  # 设置超时秒数
    
    try:
        yield
    finally:
        signal.alarm(0)  # 取消定时器


def crop_centered_on_box(image_path, box, save_path, ratio = 2):
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    # 目标裁剪区域宽高
    crop_w = W // ratio
    crop_h = H // ratio

    # box 中心坐标
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 以 box 中心为中心点，计算裁剪区域
    left = cx - crop_w // 2
    top = cy - crop_h // 2
    right = left + crop_w
    bottom = top + crop_h

    # 调整坐标以不超出图片边界
    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right > W:
        left -= (right - W)
        right = W
    if bottom > H:
        top -= (bottom - H)
        bottom = H

    # 再次确保不会出界
    left = max(0, left)
    top = max(0, top)
    right = min(W, right)
    bottom = min(H, bottom)

    # 裁剪图像
    cropped = img.crop((left, top, right, bottom))

    if save_path:
        # import pdb;pdb.set_trace()
        cropped.save(save_path)
        print(f"save crop_image{save_path}")

    return cropped, [left, top, right, bottom]

def box_contains(box1: list, box2: list) -> bool:
    return (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3])

K1_TOKEN_pixels = 1024*28*28
K2_TOKEN_pixels = 2*1024*28*28
def smart_min_pixels(now_image_pixels: int, now_set_max_pixels: int, beta: float) -> int:
    base_unit = 28 * 28  # 784
    target_pixels = int(now_image_pixels * beta)

    min_pixels = (target_pixels // base_unit) * base_unit

    if min_pixels <= now_set_max_pixels and min_pixels >= base_unit:
        return min_pixels
    else:
        # at least 1k*28*28
        # return K1_TOKEN_pixels
        return K2_TOKEN_pixels
