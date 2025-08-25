export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORKSPACE=$(pwd)
export CACHE=$WORKSPACE/cache
export DATA_DIR=$WORKSPACE/data

### 模型
MODEL_NAME=''

### 多轮Evovle
bash $WORKSPACE/scripts/multi_stage/sft1.sh
bash $WORKSPACE/scripts/multi_stage/dpo1.sh
bash $WORKSPACE/scripts/multi_stage/sft2.sh
bash $WORKSPACE/scripts/multi_stage/dpo2.sh

### eval
bash $WORKSPACE/scripts/eval/eval.sh
