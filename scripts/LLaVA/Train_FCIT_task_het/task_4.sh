################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

ALPHA=$1
OUTPUT_DIR="/mnt/cache/guohaiyang/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours"
PREVIOUS_TASK="/mnt/cache/guohaiyang/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9"

which python
/mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python -m pip install --upgrade pip
/mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

/mnt/cache/guohaiyang/miniconda3/envs/coin/bin/deepspeed --include localhost:0,1,2,3 --master_port 29600 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --expert_num 8 \
    --model_name_or_path /mnt/haiyangguo/mywork/FCIT/pre_trained/llava-v1.5-7b \
    --previous_task_model_path $PREVIOUS_TASK \
    --version $PROMPT_VERSION \
    --data_path /mnt/cache/xlpr_sharedata/FCIT_data/partitioned_data \
    --client_json_dir /mnt/ShareDB_6TB/datasets/FCIT_data/partitioned_data/Task-related \
    --image_folder /mnt/ShareDB_6TB/datasets/FCIT_data/datasets \
    --vision_tower /mnt/ShareDB_6TB/models/clip-vit-large-patch14-336 \
    --text_tower /mnt/ShareDB_6TB/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_text_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --cur_task 3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --num_clients 50 \
    --num_communication_rounds 10 \
    --fcit_type 'seq-dyn' \
    --client_selection_frac 0.1 \
    --fed_alg 'fedavg' \
    --alpha $ALPHA