for size in 1.5 3 7 
do
    torchrun --nproc_per_node=4 main_ddp.py \
        --model_class=${1} \
        --learning_rate=${2} \
        --mlp_dim1=${3} \
        --skip_layer=${4} \
        --n_query=${5} \
        --batch_size=16 \
        --ckpt_path=/storage/xylin/renkou/code/2015/qwen_sft/ckpt/Qwen2.5-${size}B-Instruct_3e-4lr \
        --tokenizer_path=Qwen2.5-0.5B \
        --lora True \
        > ./log/${1}_${2}lr_${3}mlp_${4}skip_${5}query_${size}B_log.txt
done

# sh run_ddp.sh LLM_with_mlp_query 1e-3 128 0 1