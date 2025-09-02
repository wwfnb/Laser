from laser.actor.base_actor import BaseActor
from vllm import SamplingParams, LLM


@ray.remote
class SingleStepSftActor(BaseActor):
    def __init__(self, model_name: str, actor_id: int):
        super().__init__(model_name, actor_id)

    def infer():
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
            messages = get_qwen2_5vl_prompt_msg(
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


    

    