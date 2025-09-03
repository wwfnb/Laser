import os
import base64
from PIL import Image, ImageDraw
import json
from io import BytesIO
# import pyarrow.parquet as pq
res = []
dict_list = []

file_path = 'data/benchmark/ScreenSpot-v2/screenspot_desktop_v2.json'
output_dir = 'data/benchmark/ScreenSpot-v2/screenspotv2_image'
with open(file_path,'r') as f:
    datas = json.load(f)
    for index,data in enumerate(datas) :
        width = data['bbox'][2]
        height = data['bbox'][3]
        data['bbox'][2] = data['bbox'][0] + data['bbox'][2]
        data['bbox'][3] = data['bbox'][1] + data['bbox'][3]
        data['id'] = f'scrennspot_v2-{index}'
        data['box_type'] = 'bbox'
        data['class'] = 'Desktop,' + data['data_type']
    dict_list.extend(datas)
file_path = 'data/benchmark/ScreenSpot-v2/screenspot_mobile_v2.json'
with open(file_path,'r') as f:
    datas = json.load(f)
    for index,data in enumerate(datas) :
        width = data['bbox'][2]
        height = data['bbox'][3]
        data['bbox'][2] = data['bbox'][0] + data['bbox'][2]
        data['bbox'][3] = data['bbox'][1] + data['bbox'][3]
        data['id'] = f'scrennspot_v2-{index}'
        data['box_type'] = 'bbox'
        data['class'] = 'Mobile,' + data['data_type']
    dict_list.extend(datas)
file_path = 'data/benchmark/ScreenSpot-v2/screenspot_web_v2.json'
with open(file_path,'r') as f:
    datas = json.load(f)
    for index,data in enumerate(datas) :
        width = data['bbox'][2]
        height = data['bbox'][3]
        data['bbox'][2] = data['bbox'][0] + data['bbox'][2]
        data['bbox'][3] = data['bbox'][1] + data['bbox'][3]
        data['id'] = f'scrennspot_v2-{index}'
        data['box_type'] = 'bbox'
        data['class'] = 'Web,' + data['data_type']
    dict_list.extend(datas)
print(len(dict_list))
# # 处理每个条目
for item in dict_list:
    try:
        img_name = f"{item['img_filename']}" if 'img_filename' in item else f"{hash(item['image'])}.png"
        img_path = os.path.join(output_dir, img_name)
        item['image'] = img_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        item['image'] = None  # 或保持原值
with open('data/benchmark/ScreenSpot-v2/data_correct.json','w') as f:
    json.dump(dict_list,f,ensure_ascii=False, indent=4)
print('处理完毕！')
