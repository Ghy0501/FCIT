$BETA=$1

pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_1.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_2.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_3.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_4.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_5.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_6.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_7.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_task_hom/task_8.sh $BETA