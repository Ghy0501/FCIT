$BETA=$1

pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_1.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_2.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_3.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_4.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_5.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_6.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_7.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_het/task_8.sh $BETA