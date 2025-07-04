#!/bin/bash

ALPHA=$1
NUM_TASK=4

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_aokvqa.sh cap-hom-$ALPHA-task1 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh cap-hom-$ALPHA-task1 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_grounding.sh cap-hom-$ALPHA-task1 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-$ALPHA-task1 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_aokvqa.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_grounding.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh cap-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_aokvqa.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_grounding.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_tabmwp.sh cap-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_aokvqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_grounding.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_tabmwp.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_dvqa.sh cap-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/cap-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK