START_TIME=$(date +%s)
ulimit -u 65536
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Number of GPUs: $num_gpus"

# max_pixels=2408448 # @1
max_pixels=4816896 # @1
# max_pixels=6422528 # 8k @1
# max_pixels=999999999 # 8k @1

# min_pixels=2408448 # 8k @1
# min_pixels=100352
min_pixels=802816
# min_pixels=602112
# min_pixels=501760
# min_pixels=301056
# min_pixels=401408
# min_pixels=200704
# min_pixels=1605632
# min_pixels=2408448 # 8k @1
# min_pixels=4816896 # 8k @1

# max_pixels=7225344 # @1
# max_pixels=999999999 # @1


n_times=1 # @2
# set test model
BASE_MODEL='saves/qwen2_5vl-7b/multi_step_sft_model'
model_type="" # @4 
model_name=$BASE_MODEL/$model_type

LOG_PREFIX="result"  # @5

log_path="${LOG_PREFIX}/${model_type}-MAX${max_pixels}-Pass@${n_times}-Min@${min_pixels}.json"
python "src/laser/eval/eval_screenspot_v2.py"  \
    --model_name_or_path $model_name  \
    --log_path $log_path \
    --worker_node $num_gpus \
    --img_folder "data/benchmark/ScreenSpot-v2/screenspotv2_image"   \
    --annotation_file "data/benchmark/ScreenSpot-v2/data_correct.json"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --inst_style "instruction" \
    --max_pixels $max_pixels \
    --min_pixels $min_pixels \
    --n_times $n_times \

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))
echo "总运行时间：$HOURS 小时 $MINUTES 分钟 $SECONDS 秒"