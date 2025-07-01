python -u main.py \
    --train_data=../../../data/CHFS/fusion/train_2015.csv \
    --valid_data=../../../data/CHFS/fusion/val_2015.csv --test_data=../../../data/CHFS/fusion/test_2015.csv \
    --model_class=${1} \
    --learning_rate=${2} \
    --mlp_dim1=${3} \
    --skip_layer=${4} \
    --n_query=${5} \
    --debug=${6} \
    --ckpt_path=/storage/xylin/renkou/code/2015/qwen_sft/ckpt/Qwen2.5-0.5B-Instruct_3e-4lr \
    --tokenizer_path=Qwen2.5-0.5B \
    > ./log/${1}_${2}lr_${3}dim1_${4}dim2_${5}dimproj_log.txt #2>&1 &

# sh run.sh LLM_with_mlp_query 1e-3 128 0 1 True