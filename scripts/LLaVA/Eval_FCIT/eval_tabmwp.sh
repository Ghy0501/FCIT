#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" # 
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='/mnt/haiyangguo/mywork/FCIT/CoIN/checkpoints/LLaVA/FCIT/multi_task/multask_llava_lora_ours/llava_lora_finetune_epoch_9'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    NUM_TASK=8
else
    NUM_TASK=$3
fi

RESULT_DIR="./results/FCIT/each_dataset/TabMWP"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_tabmwp \
        --model-path $MODELPATH \
        --model-base /mnt/haiyangguo/mywork/FCIT/pre_trained/llava-v1.5-7b \
        --question-file /mnt/ShareDB_6TB/datasets/FCIT_data/instructions/TabMWP/test.json \
        --image-folder /mnt/ShareDB_6TB/datasets/FCIT_data/datasets \
        --text-tower /mnt/ShareDB_6TB/models/clip-vit-large-patch14-336 \
        --num-task $NUM_TASK \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_iconqa \
    --annotation-file /mnt/ShareDB_6TB/datasets/FCIT_data/instructions/TabMWP/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

# python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \