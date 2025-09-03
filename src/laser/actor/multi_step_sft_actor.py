from laser.actor.base_actor import BaseActor
from vllm import SamplingParams, LLM
from tqdm import tqdm
from laser.utils.misc import generate_id, get_date, load_and_resize_image, extend_box, crop_image_by_box, resize_image, smart_extend_box
from collections import Counter
import ray
import re
import copy
from PIL import Image
import os
from transformers import AutoProcessor 
from laser.prompt_builder.multi_step_sft_prompt_builder import MultiStepSftPromptBuilder
from laser.utils.misc import TENSOR_PARALLEL_SIZE

@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class MultiStepSftActor(BaseActor):

    def __init__(self, model_name: str, actor_id: int):
        super().__init__(model_name, actor_id)
        self.generate_prompt_builder = MultiStepSftPromptBuilder.GeneratePromptBuilder()

    # def fisrt_crop_infer(self):
    #     pass
    def trans_msg_to_inputs(self, batch_history_msg: list, batch_resize_img:list):
        batch_input = []
        for h_msg, resize_img_lst in zip(batch_history_msg, batch_resize_img):
            # print("len_hmsg", len(h_msg))
            # print("len resize_img_lst", len(resize_img_lst))
            assert len(resize_img_lst)*2 == len(h_msg)

            prompt = self.processor.apply_chat_template(
                h_msg, tokenize=False, add_generation_prompt=True
            )
            # print("prompt:", prompt)
            inputs = {
                "prompt": prompt, 
                "multi_modal_data": {"image": resize_img_lst}
            }
            batch_input.append(inputs)
        return batch_input
    
    ####TODO: 这个解析可能需要改， 因为crop这个输出的box很多长度非法， 看一下最终的response来改（目前发挥你高度）
    def parser_res(self, res: str):
        # print("paser_res.........")
        def get_para(action_str: str):
            box_match = re.findall(r"\d+(?:\.\d+)?", action_str)
            box = [float(x) for x in box_match]
            return box

        res = res.strip()
        think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        else:
            think= None

        if not res.endswith("</action>"):
            res += "</action>"
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", res, re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
            if "crop" in action_text:
                action = "crop"
                para = get_para(action_text)
            elif "click" in action_text:
                action = "click"
                para = get_para(action_text)
            else:
                action = None
                para = None
        else:
            action = None
            para = None
        # print(f"action: {action}, para: {para}")
        return action, para
    
    def trans_box_to_box(self, box, src_image, ob_image):
        return [box[0] / src_image.size[0] * ob_image.size[0], 
                box[1] / src_image.size[1] * ob_image.size[1], 
                box[2] / src_image.size[0] * ob_image.size[0], 
                box[3] / src_image.size[1] * ob_image.size[1]]
    
    def trans_point_to_point(self, point, src_image, ob_image):
        return [point[0] / src_image.size[0] * ob_image.size[0], 
                point[1] / src_image.size[1] * ob_image.size[1]]

    
    def process_responses(self, sample_ids, responses, sample_id2traj, MAX_PIXELS: int, continue_or_not: bool):
        for (sample_id, response) in zip(sample_ids, responses):
            action, para = self.parser_res(response[0])
            current_traj = sample_id2traj[sample_id]   ### debug
            debugg = 0
            if not(action is not None and para is not None and para != []):
                continue
            try:
                if action == "click":
                    model_point = para[:2]
                    # crop_point = self.trans_point_to_point(model_point, item["resize_crop_image"], item["crop_image"])
                    debugg = 1
                    click_point_on_crop = self.trans_point_to_point(model_point, current_traj["resize_images"][-1], current_traj["images"][-1])
                    debugg= 2
                    if current_traj['raw_box']:
                        pre_raw_box = current_traj['raw_box'][-1]
                        click_point_base_raw = [click_point_on_crop[0] + pre_raw_box[0], click_point_on_crop[1] + pre_raw_box[1]]
                    else:
                        # 直接完成的点击操作
                        click_point_base_raw = click_point_on_crop
                    # finish this traj
                    debugg= 3
                    current_traj['raw_response'].append(response[0])
                    debugg= 4
                    current_traj['raw_point'] = click_point_base_raw
                    debugg= 5
                    current_traj['is_end'] = True
                elif action == "crop":
                    print("")
                    assert current_traj['resize_images'] and current_traj['images']
                    debugg= 6
                    model_box = para[:4]
                    crop_box_on_crop = self.trans_box_to_box(model_box,  current_traj["resize_images"][-1], current_traj["images"][-1])
                    if current_traj['raw_box']:
                        pre_raw_box = current_traj['raw_box'][-1]
                        crop_box_base_raw = [crop_box_on_crop[0] + pre_raw_box[0], crop_box_on_crop[1] + pre_raw_box[1], crop_box_on_crop[2] + pre_raw_box[0], crop_box_on_crop[3] + pre_raw_box[1]]
                    else:
                        crop_box_base_raw = crop_box_on_crop
                    # extend crop box if it is too small !
                    if (crop_box_base_raw[3] - crop_box_base_raw[1] <= 28 or crop_box_base_raw[2] - crop_box_base_raw[0] <= 28):
                        crop_box_base_raw = smart_extend_box(box= crop_box_base_raw, image= current_traj["images"][0])
                        # assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] <= current_traj["images"][0].size[0] and crop_box_base_raw[2] >= 0 and crop_box_base_raw[3] <= current_traj["images"][0].size[1], f"error crop box, failed by assert."
                        assert crop_box_base_raw[0] >= 0 and crop_box_base_raw[1] >= 0 and crop_box_base_raw[2] <= current_traj["images"][0].size[0] and crop_box_base_raw[3] <= current_traj["images"][0].size[1], f"error crop box, failed by assert. crop_box_base_raw(after extend): [{crop_box_base_raw}] but raw_image_size: [{current_traj['images'][0].size}]"
                    # crop image
                    debugg= 7
                    crop_image = crop_image_by_box(current_traj["images"][0], crop_box_base_raw)
                    resize_crop_image = resize_image(
                        image = crop_image, 
                        factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                        min_pixels= self.processor.image_processor.min_pixels,
                        max_pixels=MAX_PIXELS,                 
                    )
                    assistant_msg = self.generate_prompt_builder.bulid_assistant_msg(response[0])
                    user_msg_2 = self.generate_prompt_builder.bulid_user_msg_2(resize_crop_image)
                    
                    current_traj['raw_box'].append(crop_box_base_raw)
                    current_traj['raw_response'].append(response[0])
                    # add crop images
                    current_traj['images'].append(crop_image)
                    current_traj['resize_images'].append(resize_crop_image)
                    # add new messages
                    current_traj['history_msgs'].append(assistant_msg)
                    current_traj['history_msgs'].append(user_msg_2)
                    # just one step ; set is_end to True
                    # current_traj['is_end'] = True
                    if continue_or_not:
                        current_traj['is_end'] = False
                    else:
                        current_traj['is_end'] = True

            except Exception as e:
                print(f"Error processing image [{sample_id}] with response [{response}]: {e}")
                print(debugg)
                continue

    def open_image(self, image_url: str):
        if isinstance(image_url, str):
            assert os.path.exists(image_url) and os.path.isfile(image_url), "Invalid input image path."
            open_image = Image.open(image_url).convert('RGB')
        assert isinstance(open_image, Image.Image), "Invalid input image."
        return open_image

    def initial_batch(self, samples: list) -> dict:
        SYS_MSG = self.generate_prompt_builder.build_sys_msg()
        sample_id2traj = {}
        for sample in samples:
            instruction = sample['prompt_to_evaluate']
            raw_image = self.open_image(sample["img_filename"])
            resize_raw_image = self.open_image(sample["resize_image_url"])
            
            usr_msg_1 = self.generate_prompt_builder.build_user_msg_1(instruction, resize_raw_image)
            history_msgs = [SYS_MSG, usr_msg_1]
            # dict to record traj
            initial_traj = {
                "sample_id": sample['id'],
                "raw_response": [],
                "raw_box": [],
                "images": [raw_image],
                "resize_images": [resize_raw_image],
                "raw_point": None,
                # pop when record
                "history_msgs": history_msgs,
                "is_end": False,
            }
            sample_id2traj[sample['id']] = initial_traj
        return sample_id2traj
    
    def not_end_trajs(self, sample_id2traj: dict):
        sample_ids = []
        batch_history_msgs = []
        batch_history_imgs = []
        for sample_id, traj in sample_id2traj.items():
            if traj['is_end']: continue
            sample_ids.append(sample_id)
            batch_history_msgs.append(traj['history_msgs'])
            batch_history_imgs.append(traj['resize_images'])
        return sample_ids, batch_history_msgs, batch_history_imgs

    def second_crop_infer(self, samples: list, sampling_params: SamplingParams, continue_or_not: bool, max_pixels: int, passN: int=5, max_roll: int=5, batch_size: int= 16):
        sample_id2sample = {}
        for sample in samples:
            sample_id2sample[sample['id']] = sample

        # results_list = []
        for batch_start in range(0, len(samples), batch_size):
            try:
                log_items = []
                batch_samples = samples[batch_start:batch_start + batch_size]
                sample_id2trajs = {}
                for sample in batch_samples:
                    sample_id2trajs.setdefault(sample['id'], [])
                # traj: raw_response: list, raw_box: list, row_point: Point, Image_size: list[Size], resize_Image_size: list[Size]
                for sampling_idx in range(passN):
                    # {sampling_idx} Time: infer this batch
                    sample_id2traj = self.initial_batch(samples= batch_samples)
                    for step in range(max_roll):
                        sample_ids, batch_history_msgs, batch_history_imgs = self.not_end_trajs(sample_id2traj)
                        if not sample_ids:
                            break
                        batch_inputs = self.trans_msg_to_inputs(batch_history_msgs, batch_history_imgs)
                        batch_outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
                        responses = [[o.text for o in output.outputs] for output in batch_outputs]
                        self.process_responses(sample_ids, responses, sample_id2traj, max_pixels, continue_or_not)
                        assert len(responses) == len(sample_ids) == len(batch_history_msgs) == len(batch_history_imgs)
                    for sample in batch_samples:
                        sample_id = sample['id']
                        traj = sample_id2traj[sample_id]
                        # clean traj
                        if "images" in traj: traj.pop("images")
                        if "resize_images" in traj: traj.pop("resize_images")
                        if "history_msgs" in traj: traj.pop("history_msgs")
                        sample_id2trajs[sample_id].append(traj)
                # record this batch
                for sample_id, trajs in sample_id2trajs.items():
                    record_item = copy.deepcopy(sample_id2sample[sample_id])
                    record_item['trajs'] = trajs
                    # record_item['new_trajs_base_raw'] = self.eval_trajs(record_item)
                    log_items.append(record_item)
                self.write_file(log_items)
            except Exception as e:
                pass
        return None