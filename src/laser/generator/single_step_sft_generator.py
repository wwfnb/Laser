from laser.actor.single_step_sft_actor import SingleStepSftActor

class SingleStepSftGenerator:
    def __init__():
        pass

    def crop_data_reader():
        pass

    def batch_inf(self, ):
        setup_ray(WORKER_NODE)

        infer_actors = [SingleStepSftActor.remote(
            model_name= MODEL_NAME,
            actor_id= i
        ) for i in range(WORKER_NODE)]
        for infer_actor in infer_actors:
            infer_actor.set_response_file.remote(RESPONSE_FILE)
        
        for train_dataset in crop_data_reader(MAX_SAMPLES):

            print(f"[INFO] load train dataset: {len(train_dataset)} samples")
            train_chunks = np.array_split(train_dataset, WORKER_NODE)
            sampling_params = SamplingParams(
                n=VOTING_TIMES,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )

            futures = []

            for chunk, infer_actor in zip(train_chunks, infer_actors):

                print(f"[DEBUG] chunk size: {len(chunk)}")
                future = infer_actor.infer.remote(
                    samples= chunk,
                    sampling_params= sampling_params,
                    batch_size=BATCH_SIZE,
                )
                futures.append(future)
            
            # get results
            infer_results = []
            for future in futures:
                try:
                    infer_results.append(ray.get(future))
                except Exception as e:
                    print(f"ray get error: {e}")
                    infer_results.append(None)
        
        return None


    def crop_data_generate(self, model_path:str, dataset_path:str, crop_dataset_output_path:str):
        self.batch_inf()


    def generate(self, model_path: str, dataset_path: str, synthetic_dataset_path: str):
        # 
        dataset_prepare()
        #
        crop_dataset = self.crop_data_generate(model_path, dataset_path, crop_dataset_output_path)
        #
        click_dataset = self.click_data_generate(model_path, crop_dataset, click_dataset_output_path)
        #
        final_dataset = self.reject_sampling(click_dataset)
        # write into synthetic_dataset_path
        self.write(final_dataset, synthetic_dataset_path)
        


    
