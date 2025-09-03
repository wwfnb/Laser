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
from laser.prompt_builder.single_step_dpo_prompt_builder import SingleStepDpoPromptBuilder
from laser.utils.misc import TENSOR_PARALLEL_SIZE


####
@ray.remote(num_gpus=TENSOR_PARALLEL_SIZE)
class SingleStepDpoActor(BaseActor):
    def __init__(self, model_name: str, actor_id: int):
        super().__init__(model_name, actor_id)
        self.generate_prompt_builder = SingleStepDpoPromptBuilder.GeneratePromptBuilder()
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
    
    def open_image(self, image_url: str):
        if isinstance(image_url, str):
            assert os.path.exists(image_url) and os.path.isfile(image_url), "Invalid input image path."
            open_image = Image.open(image_url).convert('RGB')
        assert isinstance(open_image, Image.Image), "Invalid input image."
        return open_image

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
    
    def process_responses(self, sample_ids, responses, sample_id2traj, MAX_PIXELS: int, continue_or_not: bool):
        for (sample_id, response) in zip(sample_ids, responses):
            action, para = self.parser_res(response[0])
            current_traj = sample_id2traj[sample_id]   ### debug
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
                    current_traj['raw_response'].append(response[0])
                    current_traj['raw_point'] = click_point_base_raw
                    current_traj['is_end'] = True
                elif action == "crop":
                    print("")
                    assert current_traj['resize_images'] and current_traj['images']
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
                    if continue_or_not:
                        current_traj['is_end'] = False
                    else:
                        current_traj['is_end'] = True

            except Exception as e:
                print(f"Error processing image [{sample_id}] with response [{response}]: {e}")
                continue


####crop_candidates_infer和score_candidates_infer可以直接给dpo2复用, generator中使用
####dpo1、dpo2的区别是，1、crop之后继续与否(process_reponse)，passn@n, 这两个变量由generator传入就行
    def crop_candidates_infer(self, samples: list, sampling_params: SamplingParams, continue_or_not: bool, max_pixels: int, passN: int=5, max_roll: int=5, batch_size: int= 16):
        def initial_batch(samples: list) -> dict:
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
        
        
        sample_id2sample = {}
        for sample in samples:
            sample_id2sample[sample['id']] = sample

        ##############################

        # results_list = []
        for batch_start in range(0, len(samples), batch_size):
            log_items = []
            batch_samples = samples[batch_start:batch_start + batch_size]
            sample_id2trajs = {}
            for sample in batch_samples:
                sample_id2trajs.setdefault(sample['id'], [])
            # traj: raw_response: list, raw_box: list, row_point: Point, Image_size: list[Size], resize_Image_size: list[Size]
            for sampling_idx in range(passN):
                # {sampling_idx} Time: infer this batch
                sample_id2traj = initial_batch(samples= batch_samples)
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
                log_items.append(record_item)
            self.write_file(log_items)
        return None

    def eval_samples_for_second(self, sample_lst):

        def eval_point(pre_point, bbox):
            if pre_point == None:
                return False
            else:
                if bbox[0] <= pre_point[0] <= bbox[2] and bbox[1] <= pre_point[1] <= bbox[3]:
                    return True
            return False

        for sample in sample_lst:
            if sample["focus"] == False:
                sample["score"] = 0
            else:
                trajs = sample["trajs"]
                grounded_box = sample["bbox"]
                score = 0
                for t in trajs:
                    raw_point = t["raw_point"]
                    t["correctness"] = eval_point(raw_point, grounded_box)
                    score += eval_point(raw_point, grounded_box)
                sample["score"] = score
    def eval_samples_for_trible(self, sample_lst):

        def eval_point(pre_point, bbox):
            if pre_point == None:
                return False
            else:
                if bbox[0] <= pre_point[0] <= bbox[2] and bbox[1] <= pre_point[1] <= bbox[3]:
                    return True
            return False

        for sample in sample_lst:
            trajs = sample["trajs"]
            grounded_box = sample["bbox"]
            score = 0
            for t in trajs:
                raw_point = t["raw_point"]
                t["correctness"] = eval_point(raw_point, grounded_box)
                score += eval_point(raw_point, grounded_box)
            sample["score"] = score


    def split_samples_for_second(self, samples):

        focus_f_samples = []
        focus_t_samples = []

        for item in samples:
            trajs = item["trajs"]
            assert len(trajs) == 2
            for idx in range(len(trajs)):
                t = trajs[idx]
                copy_item = copy.deepcopy(item)
                if "trajs" in copy_item: copy_item.pop("trajs")
                copy_item["raw_response"] = t["raw_response"]
                copy_item["raw_box"] = t["raw_box"]
                copy_item["focus"] = t["focus"]
                copy_item["id"] = t["sample_id"] + str(idx)
                if t["focus"] == True:
                    focus_t_samples.append(copy_item)
                else:
                    focus_f_samples.append(copy_item)

        return focus_f_samples, focus_t_samples

    def split_samples_for_trible(self, samples):

        focus_f_samples = []
        focus_t_samples = []

        for item in samples:
            trajs = item["trajs"]
            assert len(trajs) == 2
            for idx in range(len(trajs)):
                t = trajs[idx]
                copy_item = copy.deepcopy(item)
                if "trajs" in copy_item: copy_item.pop("trajs")
                copy_item["raw_response"] = t["raw_response"][:2]
                copy_item["raw_box"] = t["raw_box"][:2]
                # copy_item["focus"] = t["focus"]
                copy_item["id"] = t["sample_id"] + str(idx)
                # if t["focus"] == True:
                focus_t_samples.append(copy_item)
                # else:
                # focus_f_samples.append(copy_item)
        return focus_f_samples, focus_t_samples
    
    def combine_samples(self, focus_f_samples, log_items, initial_stage):
        output_lst = []
        id_2sample_lst = dict()

        for item in focus_f_samples:
            item["trajs"] = None

        mix_lst = log_items + focus_f_samples

        for item in mix_lst:
            temp_id = item["id"][:-1]
            if temp_id in id_2sample_lst:
                continue

            copy_item = copy.deepcopy(item)
            copy_item["id"] = temp_id
            if "trajs" in copy_item: copy_item.pop("trajs")
            if "raw_box" in copy_item: copy_item.pop("raw_box")
            if "focus" in copy_item: copy_item.pop("focus")
            if "raw_response" in copy_item: copy_item.pop("raw_response")
            copy_item["choices"] = []
            id_2sample_lst[temp_id] = copy_item
        
        ####给samples打分
        if initial_stage == "double":
            eval_samples = self.eval_samples_for_second
        elif initial_stage == "trible":
            eval_samples = self.eval_samples_for_trible
        else:
            raise ValueError("initial_stage must be double or trible")
        eval_samples(mix_lst)
        
        for item in mix_lst:
            temp_id = item["id"][:-1]

            id_2sample_lst[temp_id]["choices"].append(item)

        for idx, sample in id_2sample_lst.items():
            assert len(sample["choices"]) == 2, "choices length != 2"
            output_lst.append(sample)
        return output_lst
    


    def score_candidates_infer(self, samples: list, sampling_params: SamplingParams, initial_stage: str, continue_or_not: bool, max_pixels: int, passN: int=5, max_roll: int=5, batch_size: int= 16):
        def initial_batch_for_trible(samples: list, MAX_PIXELS: int) -> dict:
            sample_id2traj = {}
            SYS_MSG= self.generate_prompt_builder.build_sys_msg()
            for sample in samples:
                # print("sample", samples)
                instruction = sample['prompt_to_evaluate']
                raw_image = self.open_image(sample["img_filename"])
                resize_raw_image = self.open_image(sample["resize_image_url"])

                first_crop_box_in_raw = sample["raw_box"][0]
                second_crop_box_in_raw = sample["raw_box"][1] 

                first_crop_image = crop_image_by_box(raw_image, first_crop_box_in_raw)
                second_crop_image = crop_image_by_box(raw_image, second_crop_box_in_raw)

                resize_first_crop_image = resize_image(
                    image = first_crop_image, 
                    factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels= self.processor.image_processor.min_pixels,
                    max_pixels=MAX_PIXELS,                 
                )

                resize_second_crop_image = resize_image(
                    image = second_crop_image, 
                    factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels= self.processor.image_processor.min_pixels,
                    max_pixels=MAX_PIXELS,   
                )
                
                usr_msg_1 = self.generate_prompt_builder.build_user_msg_1(instruction, resize_raw_image)
                assistant_msg_1 = self.generate_prompt_builder.bulid_assistant_msg(sample["raw_response"][0])
                user_msg_2_1 = self.generate_prompt_builder.bulid_user_msg_2(resize_first_crop_image)
                assistant_msg_2 = self.generate_prompt_builder.bulid_assistant_msg(sample["raw_response"][1])
                user_msg_2_2 =  self.generate_prompt_builder.bulid_user_msg_2(resize_second_crop_image)

                history_msgs = [SYS_MSG, usr_msg_1, assistant_msg_1 , user_msg_2_1, assistant_msg_2, user_msg_2_2]
                # dict to record traj
                ###add history 
                initial_traj = {
                    "sample_id": sample['id'],
                    "raw_response": copy.deepcopy(sample["raw_response"][:2]),
                    "raw_box": copy.deepcopy(sample["raw_box"][:2]),
                    "images": [raw_image, first_crop_image, second_crop_image],
                    "resize_images": [resize_raw_image, resize_first_crop_image, resize_second_crop_image],
                    "raw_point": None,
                    # pop when record
                    "history_msgs": history_msgs,
                    "is_end": False,
                }
                sample_id2traj[sample['id']] = initial_traj
            return sample_id2traj
        def initial_batch_for_double(samples: list, MAX_PIXELS: int) -> dict:
            sample_id2traj = {}
            SYS_MSG= self.generate_prompt_builder.build_sys_msg()
            for sample in samples:
                instruction = sample['prompt_to_evaluate']
                raw_image = self.open_image(sample["img_filename"])
                resize_raw_image = self.open_image(sample["resize_image_url"])

                raw_box = sample["raw_box"][0]
                raw_response = sample["raw_response"][0]

                crop_image = crop_image_by_box(raw_image, raw_box)
                resize_crop_image = resize_image(
                    image = crop_image, 
                    factor= self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels= self.processor.image_processor.min_pixels,
                    max_pixels=MAX_PIXELS,                 
                )
                
                usr_msg_1 = self.generate_prompt_builder.build_user_msg_1(instruction, resize_raw_image)
                assistant_msg = self.generate_prompt_builder.bulid_assistant_msg(raw_response)
                user_msg_2 = self.generate_prompt_builder.bulid_user_msg_2(resize_crop_image)

                history_msgs = [SYS_MSG, usr_msg_1, assistant_msg, user_msg_2]

                # dict to record traj
                ###add history 
                initial_traj = {
                    "sample_id": sample['id'],
                    "raw_response": [raw_response],
                    "raw_box": [raw_box],
                    "images": [raw_image, crop_image],
                    "resize_images": [resize_raw_image, resize_crop_image],
                    "raw_point": None,
                    # pop when record
                    "history_msgs": history_msgs,
                    "is_end": False,
                }
                sample_id2traj[sample['id']] = initial_traj
            return sample_id2traj
        
        if initial_stage == "double":
            initial_batch = initial_batch_for_double
            split_samples = self.split_samples_for_second
        elif initial_stage == "trible":
            initial_batch = initial_batch_for_trible
            split_samples = self.split_samples_for_trible
        else:
            raise ValueError("initial_stage must be double or trible")


        for batch_start in range(0, len(samples), batch_size):
            log_items = []
            batch_samples = samples[batch_start:batch_start + batch_size]
            ###split将id 变为id + str(i)， 并挑选出focus为true的继续推
            ###最后根据id前缀合并
            focus_f_samples, batch_samples = split_samples(batch_samples)

            sample_id2sample = {}
            for sample in batch_samples:
                sample_id2sample[sample['id']] = sample

            sample_id2trajs = {}
            for sample in batch_samples:
                sample_id2trajs.setdefault(sample['id'], [])
            # traj: raw_response: list, raw_box: list, row_point: Point, Image_size: list[Size], resize_Image_size: list[Size]
            for sampling_idx in range(passN):
                # {sampling_idx} Time: infer this batch
                sample_id2traj = initial_batch(samples= batch_samples, MAX_PIXELS= max_pixels)
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
                log_items.append(record_item)

            ###合并
            output_result = self.combine_samples(focus_f_samples, log_items, initial_stage)
            self.write_file(output_result)
        return None