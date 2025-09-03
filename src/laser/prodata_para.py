import json
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock, Manager, Process
from tqdm import tqdm  # 进度条

input_json = "data/opensource_data/inp.json"
output_jsonl = "data/opensource_data/grounding_r1.jsonl"
image_folder = "data/opensource_data/image"
NUM_WORKERS = 30
lock = Lock()
manager = Manager()
count = manager.Value('i', 0)  # 共享整数变量
count_error = manager.Value('i', 0)

def process_item(item):
    global count
    try:
        conversations = item["conversations"]
        assert len(conversations) == 2, "conversations length should be 2"
        instruction = conversations[0]["value"].split("<image>")[-1]
        assert instruction, "instruction is empty"

        image_url_relative = item["image"]
        image_url = os.path.join(image_folder, image_url_relative)
        assert os.path.exists(image_url), "image not exists"
        image = Image.open(image_url).convert('RGB')
        image_width, image_height = image.size

        bbox = item["bbox"]
        bbox = [
            round(bbox[0] / 1000 * image_width),
            round(bbox[1] / 1000 * image_height),
            round(bbox[2] / 1000 * image_width),
            round(bbox[3] / 1000 * image_height)
        ]


        image_url_relative =  os.path.join("image", image_url_relative)
        assert os.path.exists(image_url_relative), "image_url_relative"
        result_json = {
            "image_url": image_url_relative,
            "instruction": instruction,
            "action_type": None,
            "coordinate": bbox
        }
        with lock:
            count.value += 1
        return json.dumps(result_json, ensure_ascii=False)
    except Exception as e:
        # print(image_url)
        count_error.value += 1
        print(f"[ERROR] Failed to process item: {e}")
        return None

if __name__ == "__main__":
    with open(input_json, "r") as fin:
        data = json.load(fin)

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                results.append(result)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in results:
            fout.write(line + "\n")
print("error: ", count_error.value)
print("sum:", count.value)