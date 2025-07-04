$BETA=$1

pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_hom/task_1.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_hom/task_2.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_hom/task_3.sh $BETA
pip install -e .
sh scripts/LLaVA/Train_FCIT_cap_hom/task_4.sh $BETA