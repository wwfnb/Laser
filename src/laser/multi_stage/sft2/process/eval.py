import json

input_jsonl = "/public/home/ljt/hqp/GUI-grounding/generate/0725_dpo_model_generate_170k/0725generate_from_100k_all_wrong.jsonl"

output_jsonl = "/public/home/ljt/hqp/GUI-grounding/generate/0725_dpo_model_generate_170k/0725generate_one_crop_eval_from_100k_all_wrong.jsonl"

def eval_sample_positive_gt(trajs, bbox):
    result_lst = []
    n_focus_correctness = []

    for traj in trajs:
        raw_point = traj["raw_point"]
        if raw_point == None:
            result_lst.append("wrong_format")
        elif (bbox[0] <= raw_point[0] <= bbox[2]) and (bbox[1] <= raw_point[1] <= bbox[3]):
            result_lst.append("correct")
        else:
            result_lst.append("wrong")

        focus_lst = []
        raw_box_lst = traj["raw_box"]
        for pre_box in raw_box_lst:
            if pre_box == None:
                focus_lst.append("wrong_format")
            elif pre_box[0] <= bbox[0] and pre_box[1] <= bbox[1] and pre_box[2] >= bbox[2] and pre_box[3] >= bbox[3]:
                focus_lst.append("correct")
            else:
                focus_lst.append("wrong")
        n_focus_correctness.append(focus_lst)

    if "correct" in result_lst:
        correctness = "correct"
    else:
        if "wrong" in result_lst:
            correctness = "wrong"
        else:
            correctness = "wrong_format"
    return n_focus_correctness, result_lst, correctness

def eval_trajs(item):
    trajs = item["trajs"]
    for t in trajs:
        n_focus_correctness, result_lst, correctness = eval_sample_positive_gt([t], item["bbox"])
        if len(n_focus_correctness[0]) == 0:
            t["n_focus_correctness"] = "wrong_format"
        else:
            t["n_focus_correctness"] = n_focus_correctness[0][0]
        t["correctness"] = correctness


with open(input_jsonl, "r") as fin, open(output_jsonl , "w") as fout:
    for line in fin:
        item = json.loads(line)
        eval_trajs(item)
        fout.write(json.dumps(item) + "\n")





