from laser.actor.base_actor import BaseActor
from vllm import SamplingParams, LLM
from tqdm import tqdm
from laser.utils.misc import generate_id, get_date, load_and_resize_image
from collections import Counter
import ray
import re
import copy
from laser.prompt_builder.single_step_sft_prompt_builder import SingleStepSftPromptBuilder
from laser.utils.misc import TENSOR_PARALLEL_SIZE


@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class SingleStepSftActor(BaseActor):
    def __init__(self, model_name: str, actor_id: int):
        super().__init__(model_name, actor_id)
        self.generate_prompt_builder = SingleStepSftPromptBuilder.GeneratePromptBuilder()

    def crop_infer(self, samples: list, sampling_params: SamplingParams, MAX_PIXELS: str, batch_size: int= 16):
        # conversation -> get image -> replace with and height
        llm_inputs = []
        raw_images = []
        for sample in samples:
            # 注意设置好 min-pixex 和 max-pixel
            image_url = sample['image_url']
            instruction = sample['instruction']
            raw_image, resize_image = load_and_resize_image(
                image_url= image_url,
                factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                min_pixels= self.processor.image_processor.min_pixels,
                max_pixels=MAX_PIXELS,
            )
            # organize data: ref: screenspot-pro/models/qwen2_5vl.py
            messages = self.generate_prompt_builder.get_single_step_sft_crop_msg(
                image= resize_image,
                instruction= instruction,
                screen_width= resize_image.width,
                screen_height= resize_image.height,
            )
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": [resize_image]}
            }
            raw_images.append(raw_image)
            llm_inputs.append(llm_input)
        for batch_start in tqdm(range(0, len(llm_inputs), batch_size), desc= "generation"):
            batch_inputs = llm_inputs[batch_start:batch_start+batch_size]
            batch_samples = samples[batch_start:batch_start+batch_size]
            batch_raw_images = raw_images[batch_start:batch_start+batch_size]
            # sampling parameter 设置采样的次数
            responses = []
            # batch infer
            batch_outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
            responses = [[o.text for o in output.outputs] for output in batch_outputs]
            finish_reasons = [[o.finish_reason for o in output.outputs] for output in batch_outputs]
            log_response_answers = []
            for idx_in_batch, (llm_input, response, finish_reason, sample, raw_image) in enumerate(zip(batch_inputs, responses, finish_reasons, batch_samples, batch_raw_images)):
                # just write into file now.
                # 注意image的大小信息
                assert len(response) == len(finish_reason) == sampling_params.n
                run_image = llm_input['multi_modal_data']['image'][0]
                item = copy.deepcopy(sample)
                item['running'] = {}
                item['running']['prompt'] = llm_input['prompt']
                item['running']['response'] = response
                item['running']['finish_reason'] = finish_reason
                item['running']['run_image_info'] = {
                    "raw_width": raw_image.width,
                    "raw_height": raw_image.height,
                    "width": run_image.width,
                    "height": run_image.height,
                }

                ### box、cot and cot_correct
                cot_list = []
                box_list = []
                cot_correct = []
                model_box_list = []

                grounded_bbox = item["coordinate"]


                for res in response:
                    think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
                    think_content = think_match.group(1) if think_match else None

                    if think_content == None:
                        continue

                    box_match = re.search(r"<box>(.*?)</box>", res, re.DOTALL)
                    box_content = box_match.group(1) if box_match else None
                    if box_content== None:
                        continue

                    box = list(map(float, re.findall(r"\d+\.?\d*", box_content)))  

                     ### 框超出屏幕   ##TODO: 修改框超出屏幕的逻辑， 有些框只是超出一点而已，但是他的cot是有用的，并且也能狂刀焦点区域(1)缩小到去掉超出屏幕的部分（2）也要排除过大的>75%
                    if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3] or box[2] > run_image.width or box[3] > run_image.height:  
                        continue

                    model_box = box.copy() 

                    box = [round(box[0] / run_image.width * raw_image.width, 4), round(box[1] / run_image.height * raw_image.height, 4), round(box[2] / run_image.width * raw_image.width, 4), round(box[3] / run_image.height * raw_image.height, 4)]

                    if len(box) != 4:
                        continue

                    cot_list.append(think_content)
                    box_list.append(box)
                    model_box_list.append(model_box)

                    if box[0] <= grounded_bbox[0] and box[1] <= grounded_bbox[1] and box[2] >= grounded_bbox[2] and box[3] >= grounded_bbox[3]:
                        cot_correct.append(True)
                    else:
                        cot_correct.append(False)

                ###应该把模型生成的框也放进去，方便处理
                item['running']['model_boxes'] = model_box_list
                item['running']['cots'] =  cot_list
                item['running']['boxes'] = box_list
                item['running']['cot_correct'] = cot_correct

                log_response_answers.append(item)
            # write to file
            print(f"Actor@{self.actor_id} is writting into file ...")
            self.write_file(log_response_answers)

        metric = Counter()
        return metric
    def click_infer(self, samples: list, sampling_params: SamplingParams,  MAX_PIXELS: str, batch_size: int= 16):

        # conversation -> get image -> replace with and height
        llm_inputs = []
        raw_images = []
        for sample in samples:
            # 注意设置好 min-pixex 和 max-pixel
            image_url = sample['crop_image_url']
            instruction = sample['instruction']
            raw_image, resize_image = load_and_resize_image(
                image_url= image_url,
                factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                min_pixels= self.processor.image_processor.min_pixels,
                max_pixels=MAX_PIXELS,
            )
            # organize data: ref: screenspot-pro/models/qwen2_5vl.py
            messages = self.generate_prompt_builder.get_single_step_sft_click_msg(
                image= resize_image,
                instruction= instruction,
                screen_width= resize_image.width,
                screen_height= resize_image.height,
            )
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": [resize_image]}
            }
            raw_images.append(raw_image)
            llm_inputs.append(llm_input)
        for batch_start in tqdm(range(0, len(llm_inputs), batch_size), desc= "generation"):
            batch_inputs = llm_inputs[batch_start:batch_start+batch_size]
            batch_samples = samples[batch_start:batch_start+batch_size]
            batch_raw_images = raw_images[batch_start:batch_start+batch_size]
            # sampling parameter 设置采样的次数
            responses = []
            # batch infer
            batch_outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
            responses = [[o.text for o in output.outputs] for output in batch_outputs]
            finish_reasons = [[o.finish_reason for o in output.outputs] for output in batch_outputs]
            log_response_answers = []
            for idx_in_batch, (llm_input, response, finish_reason, sample, raw_image) in enumerate(zip(batch_inputs, responses, finish_reasons, batch_samples, batch_raw_images)):
                # just write into file now.
                # 注意image的大小信息
                # assert len(response) == len(finish_reason) == VOTING_TIMES
                run_image = llm_input['multi_modal_data']['image'][0]
                item = copy.deepcopy(sample)
                item['running'] = {}
                item['running']['prompt'] = llm_input['prompt']
                item['running']['response'] = response
                item['running']['finish_reason'] = finish_reason
                item['running']['run_image_info'] = {
                    "raw_width": raw_image.width,
                    "raw_height": raw_image.height,
                    "width": run_image.width,
                    "height": run_image.height,
                }
                log_response_answers.append(item)
                # self.write_one_by_one(item)
            # write to file
            print(f"Actor@{self.actor_id} is writting into file ...")
            self.write_file(log_response_answers)

        metric = Counter()
        return metric