# This file is used to create a data module for LLaVA dataset
from .llava_dataset import make_supervised_data_module, FCIT_make_supervised_dataset
from .llava_dataset import LazySupervisedDataset,DataCollatorForSupervisedDataset

def create_LLaVA_data_module(*args):
    return make_supervised_data_module(*args)

def create_FCIT_LLaVA_dataset(*args):
    return FCIT_make_supervised_dataset(*args)