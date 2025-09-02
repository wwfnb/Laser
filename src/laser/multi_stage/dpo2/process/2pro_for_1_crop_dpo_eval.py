import json

input_jsonl = "/public/home/ljt/hqp/GUI-grounding/0728generate_dpo/0728_mutil_turn_just_1_crop.jsonl"
output_jsonl = "/public/home/ljt/hqp/GUI-grounding/0728generate_dpo/0728_mutil_turn_just_1_crop_eval.jsonl"

count_dict = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0
}


def process_trajs(trajs, bbox):
    count_true = 0
    for t in trajs:
        if len(t["raw_box"]) == 0:
            focus = False
        else:
            pre_box = t["raw_box"][0]
            assert len(pre_box) == 4
            if pre_box[0] <= bbox[0] and pre_box[1] <= bbox[1] and pre_box[2] >= bbox[2] and pre_box[3] >= bbox[3]:
                focus = True
                count_true += 1
            else:
                focus = False
        t["focus"] = focus
    count_dict[count_true] = count_dict[count_true] + 1

    return count_true

def select(trajs):

    output_trajs = []
    f_lst = []
    t_lst = []
    for t in trajs:
        if t["focus"] == True:
            t_lst.append(t)
        else:
            f_lst.append(t)
    if len(t_lst) == 1:
        output_trajs.append(f_lst[0])
        output_trajs.append(t_lst[0])
        # print(111)
    else:
        output_trajs.append(t_lst[0])
        output_trajs.append(t_lst[1])
    return output_trajs


with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
    for line in fin:
        item = json.loads(line)
        
        bbox = item["bbox"]
        trajs = item["trajs"]
        count1 = process_trajs(trajs, bbox)
        if count1 == 0:
            continue

        item["trajs"] = select(trajs)
        assert len(item["trajs"]) == 2
        fout.write(json.dumps(item) + "\n")

print(count_dict)