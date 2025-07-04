# #!/bin/bash

NUM_TASK=4

if [ "$1" == "1" ]; then
    pip install -e .
    bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-1.0-task1 /mnt/ShareDB_6TB/datasets/FCIT_data/checkpoint/FCIT/cap-related-hom/1.0/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
elif [ "$1" == "2" ]; then
    pip install -e .
    bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-1.0-task2 /mnt/ShareDB_6TB/datasets/FCIT_data/checkpoint/FCIT/cap-related-hom/1.0/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
elif [ "$1" == "3" ]; then
    pip install -e .
    bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-1.0-task3 /mnt/ShareDB_6TB/datasets/FCIT_data/checkpoint/FCIT/cap-related-hom/1.0/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
else
    pip install -e .
    bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-1.0-task4 /mnt/ShareDB_6TB/datasets/FCIT_data/checkpoint/FCIT/cap-related-hom/1.0/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
fi