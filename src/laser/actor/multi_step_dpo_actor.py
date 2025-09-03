from laser.actor.base_actor import BaseActor
from vllm import SamplingParams, LLM
from tqdm import tqdm
from laser.utils.misc import generate_id, get_date, load_and_resize_image, extend_box, crop_image_by_box, resize_image,smart_extend_box
from collections import Counter
import ray
import re
import copy
from PIL import Image
import os
from transformers import AutoProcessor  
from laser.prompt_builder.single_step_dpo_prompt_builder import SingleStepDpoPromptBuilder
from laser.actor.base_actor import BaseActor
from laser.actor.single_step_dpo_actor import SingleStepDpoActor
from laser.utils.misc import TENSOR_PARALLEL_SIZE

@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class MultiStepDpoActor(SingleStepDpoActor):

    def __init__(self):
        super().__init__()