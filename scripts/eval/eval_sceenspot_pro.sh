START_TIME=$(date +%s)
ulimit -u 65536
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Number of GPUs: $num_gpus"

# max_pixels=2408448 # @1
max_pixels=4816896 # @1
# max_pixels=6422528  

# min_pixels=2408448
min_pixels=802816
# min_pixels=3136

n_times=1 # @2
# set test model
BASE_MODEL="saves/qwen2_5vl-7b/multi_step_sft_model" # @3
model_type="" # @4 
model_name=$BASE_MODEL/$model_type

LOG_PREFIX="result"  # @5

log_path="${LOG_PREFIX}/${model_type}-${max_pixels}-Pass@${n_times}_set_min${min_pixels}.json"
python "src/laser/eval/eval_screenspot_pro.py"  \
    --model_name_or_path $model_name  \
    --log_path $log_path \
    --worker_node $num_gpus \
    --screenspot_imgs "data/benchmark/ScreenSpot-pro/images"   \
    --screenspot_test "data/benchmark/ScreenSpot-pro/annotations"  \
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