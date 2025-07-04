import os
import copy
from dataclasses import dataclass, field
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers
from pathlib import Path
import numpy as np
from collections import defaultdict

from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from PIL import Image, ImageFile
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token
from ETrain.utils.LLaVA import conversation as conversation_lib
from ETrain.Dataset.dataset import DataArguments
from ETrain.Dataset.LLaVA import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

FCIT_seq_data_path = [
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ArxivQA/train_4w.json',              #1
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/Flickr30k-cap/train_brief_4w.json',  #3
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ImageNet-R/train.json',              #4
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/OCRVQA/train_10w.json',              #5
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/IconQA/train.json',                  #2
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/CLEVR-Math/train_4w.json',           #6
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/FigureQA/train.json',                #7
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/super-CLEVR/train.json'              #8
]

FCIT_cap_data_path = [
    ['/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/AOKVQA/train.json',                  #1
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/IconQA/train.json',                  #2
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/Grounding/train_11w.json',           #3
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ImageNet-R/train.json'],             #4
    ['/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ArxivQA/train_4w.json',              #5
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/FigureQA/train.json',                #6
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/DVQA/train.json'],                   #7
    ['/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/CLEVR-Math/train_4w.json',           #8
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/super-CLEVR/train.json',             #9
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/TabMWP/train.json'],                 #10
    ['/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/Flickr30k-cap/train_brief_4w.json',  #11
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/OCRVQA/train_10w.json']              #12
]

FCIT_multi_task_data_path = [
    ['/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/AOKVQA/train.json',                  #1
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/IconQA/train.json',                  #2
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/Grounding/train_11w.json',           #3
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ImageNet-R/train.json',              #4
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/ArxivQA/train_4w.json',              #5
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/FigureQA/train.json',                #6
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/DVQA/train.json',                    #7
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/CLEVR-Math/train_4w.json',           #8
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/super-CLEVR/train.json',             #9
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/TabMWP/train.json',                  #10
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/Flickr30k-cap/train_brief_4w.json',  #11
     '/mnt/ShareDB_6TB/datasets/FCIT_data/instructions/OCRVQA/train_10w.json']              #12
]

class FCIT_data:
    def __init__(self, fcit_type='seq', alpha=1.0):
        self.type = fcit_type
        self.alpha = alpha

    def dirichlet_partition(self, data, client_nums, min_size=50):

        data_size = len(data)
        partitions = [[] for _ in range(client_nums)]
        
        while True:
            dirichlet_weights = np.random.dirichlet([self.alpha] * client_nums, size=1)[0]
            
            indices = np.arange(data_size)
            np.random.shuffle(indices) 
            
            start_idx = 0
            for client_idx, weight in enumerate(dirichlet_weights):
                client_data_size = int(weight * data_size)
                end_idx = start_idx + client_data_size
                partitions[client_idx].extend([data[i] for i in indices[start_idx:end_idx]])
                start_idx = end_idx
            
            if all(len(partition) >= min_size for partition in partitions):
                break
            
            partitions = [[] for _ in range(client_nums)]
        
        for client_idx, partition in enumerate(partitions):
            while len(partition) < min_size:
                for donor_idx in range(client_nums):
                    if len(partitions[donor_idx]) > min_size:
                        partitions[client_idx].append(partitions[donor_idx].pop())
                        if len(partition) >= min_size:
                            break

        return partitions


    def federated_partition_and_save(self, data_path, client_nums, output_dir, data_json, task_id, file_name):

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if data_path != None:
            dataset_name = f"Task{task_id + 1}"

            dataset_output_dir = Path(output_dir) / file_name / str(self.alpha) / dataset_name

            if dataset_output_dir.exists():
                print(f"Dataset '{dataset_name}' already exists. Skipping.")
            else:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    
                partitions = self.dirichlet_partition(data, client_nums, self.alpha)

                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                for client_idx, client_data in enumerate(partitions):
                    client_file_path = dataset_output_dir / f"client_{client_idx}.json"
                    with open(client_file_path, 'w') as f:
                        json.dump(client_data, f, indent=4)
                print(f"Dataset '{dataset_name}' processed and saved.")
        elif data_json != None:
            dataset_name = f"Task{task_id + 1}"

            dataset_output_dir = Path(output_dir) / file_name / str(self.alpha) / dataset_name

            if dataset_output_dir.exists():
                print(f"Dataset '{dataset_name}' already exists. Skipping.")
            else:
                partitions = self.dirichlet_partition(data_json, client_nums, self.alpha)

                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                for client_idx, client_data in enumerate(partitions):
                    client_file_path = dataset_output_dir / f"client_{client_idx}.json"
                    with open(client_file_path, 'w') as f:
                        json.dump(client_data, f, indent=4)
                print(f"Dataset '{dataset_name}' processed and saved.")
    
    def load_data(self, data_args, client_nums):

        if 'seq' in self.type:
            for i in range(len(FCIT_seq_data_path)):
                self.federated_partition_and_save(FCIT_seq_data_path[i], client_nums, data_args.client_json_dir, None, i, 'seq')
        elif 'cap' in self.type:
            for i in range(len(FCIT_cap_data_path)):
                cap_data_json = []
                for file_path in FCIT_cap_data_path[i]:
                    cap_data_json.extend(json.load(open(file_path, "r")))
                random.shuffle(cap_data_json)
                self.federated_partition_and_save(None, client_nums, data_args.client_json_dir, cap_data_json, i, 'cap')
        elif 'multask' in self.type:
            for i in range(len(FCIT_multi_task_data_path)):
                multi_data_json = []
                for file_path in FCIT_cap_data_path[i]:
                    multi_data_json.extend(json.load(open(file_path, "r")))
                random.shuffle(multi_data_json)
                self.federated_partition_and_save(None, client_nums, data_args.client_json_dir, multi_data_json, i, 'multi')

    # def load_data(self, tokenizer, data_args, local_rank, client_nums): 
    #     """
    #     Split the training set for each task in RING
    #     """
    #     if self.type == 'seq':
    #         data_modules_dict = {i: [] for i in range(len(FCIT_seq_data_path))}
    #         data_size_dict = {i: [] for i in range(len(FCIT_seq_data_path))}
    #         for i in range(len(FCIT_seq_data_path)):
    #             data_args.data_path = FCIT_seq_data_path[i]
    #             task_dataset = create_FCIT_LLaVA_dataset(tokenizer, data_args, local_rank)
    #             task_sub_dataset, task_data_size = task_dataset.split_for_clients(client_nums, data_args.alpha, True)
    #             for sub_dataset in task_sub_dataset:
    #                 data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #                 data_modules_dict[i].append(dict(train_dataset=sub_dataset, eval_dataset=None, data_collator=data_collator))
    #             for num in task_data_size:
    #                 data_size_dict[i].append(num)
    #         return data_modules_dict, data_size_dict
    #     elif self.type == 'seq-dyn':
    #         data_modules_dict = {i: [] for i in range(len(FCIT_seq_data_path))}
    #         data_size_dict = {i: [] for i in range(len(FCIT_seq_data_path))}
    #         for i in range(len(FCIT_seq_data_path)):
    #             data_args.data_path = FCIT_seq_data_path[i]
    #             task_dataset = create_FCIT_LLaVA_dataset(tokenizer, data_args, local_rank)
    #             task_sub_dataset, task_data_size = task_dataset.split_for_clients(client_nums, data_args.alpha, True)
    #             for sub_dataset in task_sub_dataset:
    #                 data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #                 data_modules_dict[i].append(dict(train_dataset=sub_dataset, eval_dataset=None, data_collator=data_collator))
    #             for num in task_data_size:
    #                 data_size_dict[i].append(num)
    #         return data_modules_dict, data_size_dict
    #     elif self.type == 'cap':
    #         data_modules_dict = {i: [] for i in range(len(FCIT_cap_data_path))}
    #         data_size_dict = {i: [] for i in range(len(FCIT_cap_data_path))}
    #         for i in range(len(FCIT_cap_data_path)):
    #             cap_data_json = []
    #             for file_path in FCIT_cap_data_path[i]:
    #                 cap_data_json.extend(json.load(open(file_path, "r")))
    #             random.shuffle(cap_data_json)
    #             task_dataset = create_FCIT_LLaVA_dataset(tokenizer, data_args, local_rank, cap_data_json)
    #             task_sub_dataset, task_data_size = task_dataset.split_for_clients(client_nums, data_args.alpha, True)
    #             for sub_dataset in task_sub_dataset:
    #                 data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #                 data_modules_dict[i].append(dict(train_dataset=sub_dataset, eval_dataset=None, data_collator=data_collator))
    #             for num in task_data_size:
    #                 data_size_dict[i].append(num)
    #         return data_modules_dict, data_size_dict
    #     elif self.type == 'cap-dyn':
    #         data_modules_dict = {i: [] for i in range(len(FCIT_cap_data_path))}
    #         data_size_dict = {i: [] for i in range(len(FCIT_cap_data_path))}
    #         for i in range(len(FCIT_cap_data_path)):
    #             cap_data_json = []
    #             for file_path in FCIT_cap_data_path[i]:
    #                 cap_data_json.extend(json.load(open(file_path, "r")))
    #             random.shuffle(cap_data_json)
    #             task_dataset = create_FCIT_LLaVA_dataset(tokenizer, data_args, local_rank, cap_data_json)
    #             task_sub_dataset, task_data_size = task_dataset.split_for_clients(client_nums, data_args.alpha, True)
    #             for sub_dataset in task_sub_dataset:
    #                 data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #                 data_modules_dict[i].append(dict(train_dataset=sub_dataset, eval_dataset=None, data_collator=data_collator))
    #             for num in task_data_size:
    #                 data_size_dict[i].append(num)
    #         return data_modules_dict, data_size_dict
    #     elif self.type == 'multask':
    #         data_modules_dict = {i: [] for i in range(len(FCIT_multi_task_data_path))}
    #         data_size_dict = {i: [] for i in range(len(FCIT_multi_task_data_path))}
    #         for i in range(len(FCIT_multi_task_data_path)):
    #             cap_data_json = []
    #             for file_path in FCIT_multi_task_data_path[i]:
    #                 cap_data_json.extend(json.load(open(file_path, "r")))
    #             random.shuffle(cap_data_json)
    #             task_dataset = create_FCIT_LLaVA_dataset(tokenizer, data_args, local_rank, cap_data_json)
    #             task_sub_dataset, task_data_size = task_dataset.split_for_clients(client_nums, data_args.alpha, True)
    #             for sub_dataset in task_sub_dataset:
    #                 data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    #                 data_modules_dict[i].append(dict(train_dataset=sub_dataset, eval_dataset=None, data_collator=data_collator))
    #             for num in task_data_size:
    #                 data_size_dict[i].append(num)
    #         return data_modules_dict, data_size_dict
    #     else:
    #         return 0