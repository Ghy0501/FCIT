#!/bin/bash

ALPHA=$1
NUM_TASK=8

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK



pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK


pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK


pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK



pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK


pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK


pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK


pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-het-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-het/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK