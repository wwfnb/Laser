import json

input_jsonl = "/public/home/ljt/hqp/GUI-grounding/0729_dpo_train_data/0729_dpo_train_data_second_turn_802816_right_all_wrong_34k.jsonl"
output_jsonl = "/public/home/ljt/hqp/GUI-grounding/0728generate_dpo/0730_for_trible_turn.jsonl"

type_2sample= {
    "click_rightcrop_right":[],
    "click_rightcrop_wrong":[],
    "click_rightclick_wrong":[],
    "crop_rightclick_wrong":[],
    "crop_rightcrop_wrong":[],
    "crop_rightcrop_right":[]
}

with open(input_jsonl, "r") as fin:
    for line in fin:
        item = json.loads(line)
        ttype = item["chosen_type"] + item["rejected_type"]
        type_2sample[ttype].append(item)

lst = type_2sample["crop_rightcrop_right"]
with open(output_jsonl, "w") as fout:
    for item in lst:
        item["for_trible_turn"]["id"] = item["for_trible_turn"]["id"][:-1]
        for t in item["for_trible_turn"]["trajs"]:
            t["sample_id"] = t["sample_id"][:-1] 
        fout.write(json.dumps(item) + "\n")
