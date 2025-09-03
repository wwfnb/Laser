import ray
from transformers import AutoProcessor  
from laser.utils.misc import TENSOR_PARALLEL_SIZE
from vllm import SamplingParams, LLM
import json
import filelock


class BaseActor:
    def __init__(self, model_name, actor_id:int):
        self.model_name = model_name
        self.actor_id = actor_id
        self.TENSOR_PARALLEL_SIZE = TENSOR_PARALLEL_SIZE
        self.LIMIT_MM_PER_PROMPT={
            "image": 5
        }
        self.MAX_NUM_SEQS= 16
        self.MAX_MODEL_LEN= (1024*16) 
        
        # processor llm
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"#####model_name: [{self.model_name}]#########")
        print(f"#####TENSOR_PARALLEL_SIZE: [{self.TENSOR_PARALLEL_SIZE}]#########")
        print(f"#####LIMIT_MM_PER_PROMPT: [{self.LIMIT_MM_PER_PROMPT}]#########")
        print(f"#####MAX_NUM_SEQS: [{self.MAX_NUM_SEQS}]#########")
        print(f"#####MAX_MODEL_LEN: [{self.MAX_MODEL_LEN}]#########")
        self.llm = LLM(
            model=self.model_name, 
            dtype="auto", 
            tensor_parallel_size= self.TENSOR_PARALLEL_SIZE,
            limit_mm_per_prompt= self.LIMIT_MM_PER_PROMPT,
            max_num_seqs= self.MAX_NUM_SEQS,
            max_model_len= self.MAX_MODEL_LEN, # max_model_len should be ????
            max_num_batched_tokens= self.MAX_MODEL_LEN
        )

    def set_response_file(self, response_file: str):
        self.response_file = response_file

    def write_file(self, items: list[dict]):
        if not hasattr(self, "response_file"):
            return
        if not hasattr(self, "file_lock"):
            self.file_lock = filelock.FileLock("tmp/example.lock")
        with self.file_lock:
            with open(self.response_file, 'a') as fa:
                for item in items:
                    fa.write(json.dumps(item)+'\n')
    
