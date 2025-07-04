$BETA=$1

pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_het/task_1.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_het/task_2.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_het/task_3.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_het/task_4.sh $BETA