MODEL_NAME=''
DATA_SET=''

###造一步crop数据



for i in {1..100000}; do
    python /public/home/ljt/hqp/GUI-grounding/src/pixel_grounding/generate_cot_focus_grounding_r1.py
    echo $i >> tmp/cot.txt
done


###挑选一步crop正确的数据

####基于一步crop 造 click cot 数据