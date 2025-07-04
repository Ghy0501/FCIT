# #!/bin/bash

ALPHA=$1
NUM_TASK=8

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task1 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task1_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task2 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task2_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task3 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task3_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-hom-$ALPHA-task4 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task4_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-hom-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-hom-$ALPHA-task5 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task5_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-hom-$ALPHA-task6 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task6_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-hom-$ALPHA-task7 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task7_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK

pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_imagenetr.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_arxivqa.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_iconqa.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_clevr_math.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_ocrvqa.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_flickr30k.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_figureqa.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK
pip install -e .
bash scripts/LLaVA/Eval_FCIT/eval_super_clevr.sh task-hom-$ALPHA-task8 /your_path/checkpoints/FCIT/task-related-hom/$ALPHA/Task8_llava_lora_ours/llava_lora_finetune_epoch_9 $NUM_TASK