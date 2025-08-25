START_TIME=$(date +%s)
# WORKSPACE='/public/home/ljt/hqp/release/Laser/data'
# CACHE='/public/home/ljt/hqp/release/Laser/cache/'
ulimit -u 65536
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Number of GPUs: $num_gpus"

# 预设max,min_pixels
max_pixels=4816896 # @1
min_pixels=802816


n_times=1 # @2
BASE_MODEL='/public/home/ljt/hqp/GUI-grounding/LLaMA-Factory/saves/qwen2_5vl-7b/sft/0727_multi_turn_0_3_18k_18k'
model_type="" # @4 
model_name=$BASE_MODEL/$model_type

# LOG_PREFIX="0728Results/screenspot_v2/dpo/0727_multi_turn_0_7_18k_18k"  # @5
LOG_PREFIX="${WORKSPACE}/evaluation"  # @5

log_path="${LOG_PREFIX}/${model_type}-MAX${max_pixels}-Pass@${n_times}-Min@${min_pixels}.json"
evaldata_path=$DATA_DIR/Benchmark/ScreenSpot-v2/data/data_correct.json
image_folder=$CACHE/image_folder


# 这里的一些路径需要更换
python "/public/home/ljt/hqp/GUI-grounding/src/pixel_grounding/eval/eval_screenspot_v2.py"  \
    --model_name_or_path $model_name  \
    --log_path $log_path \
    --worker_node $num_gpus \
    --img_folder $image_folder   \
    --annotation_file $evaldata_path  \
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
echo "**Eval** 总运行时间：$HOURS 小时 $MINUTES 分钟 $SECONDS 秒"