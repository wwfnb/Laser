from PIL import Image
from laser.utils.misc import convert_pil_image_to_base64
import re
from laser.prompt_builder.single_step_dpo_prompt_builder import SingleStepDpoPromptBuilder

class MultiStepDpoPromptBuilder():
    def __init__(self):
        pass
    class ProcessPromptBuilder(SingleStepDpoPromptBuilder.ProcessPromptBuilder):
        def __init__(self):
            super().__init__()