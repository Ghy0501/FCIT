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
import copy

from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from ETrain.Train.LLaVA.llava_trainer import LLaVATrainer, LLaVATrainerSCAFFOLD
from collections import OrderedDict

from ETrain.Models.LLaVA import *
from ETrain.Dataset import create_LLaVA_data_module
from ETrain.Dataset.dataset import DataArguments
from ETrain.Train.Base_trainer import *
from ETrain.Train.LLaVA.llava_trainer import load_model_from_previous_task
from CoIN.peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None


class GeneralClient:
    def __init__(self, args, client_id, model, tokenizer, set_for_current_task, idx, output_dir):
        self.args = args
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.client_ckpt = output_dir
        self.set_for_current_task = set_for_current_task
        self.idx = idx

    def prepare_client_dataset(self, data_args, FCIT_type, cur_task, tokenizer, local_rank, client_weight):
        
        if FCIT_type == 'seq':
            assert cur_task <= 12
            load_client_file = os.path.join(data_args.client_json_dir, FCIT_type, str(data_args.alpha), f"Task{cur_task + 1}", f"client_{self.client_id}.json")
            print(load_client_file)
            with open(load_client_file, 'r') as f:
                data = json.load(f)
            client_weight.append(len(data))
            data_args.data_path = load_client_file
            self.client_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)
        if FCIT_type == 'cap':
            assert cur_task <= 3
            load_client_file = os.path.join(data_args.client_json_dir, FCIT_type, str(data_args.alpha), f"Task{cur_task + 1}", f"client_{self.client_id}.json")
            print(load_client_file)
            with open(load_client_file, 'r') as f:
                data = json.load(f)
            client_weight.append(len(data))
            data_args.data_path = load_client_file
            self.client_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)
        if FCIT_type == 'multask':
            assert cur_task <= 0
            load_client_file = os.path.join(data_args.client_json_dir, FCIT_type, str(data_args.alpha), f"Task{cur_task + 1}", f"client_{self.client_id}.json")
            with open(load_client_file, 'r') as f:
                data = json.load(f)
            client_weight.append(len(data))
            data_args.data_path = load_client_file
            self.client_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)
        if 'dyn' in FCIT_type and self.set_for_current_task != None:
            FCIT_type = FCIT_type.split('-')[0] 
            load_client_file = os.path.join(data_args.client_json_dir, FCIT_type, str(data_args.alpha), f"Task{self.set_for_current_task[self.idx] + 1}", f"client_{self.client_id}.json")
            print(load_client_file)
            with open(load_client_file, 'r') as f:
                data = json.load(f)
            client_weight.append(len(data))
            data_args.data_path = load_client_file
            self.client_module = create_LLaVA_data_module(tokenizer, data_args, local_rank)

        return client_weight

    def build_local_trainer(self, global_dict, local_auxiliary, global_auxiliary):
        if self.args.fed_alg == 'scaffold':
            self.local_trainer = LLaVATrainerSCAFFOLD(model=self.model, global_state=global_dict, global_auxiliary=global_auxiliary, local_auxiliary=local_auxiliary, tokenizer=self.tokenizer, args=self.args, **self.client_module)
        else:
            self.local_trainer = LLaVATrainer(model=self.model, tokenizer=self.tokenizer, args=self.args, **self.client_module)

    def train(self,):
        self.local_trainer.train()
    
    def get_cluster_center(self, clusters, idx):
        for i, cluster in enumerate(clusters):
            if idx in cluster:
                return i
        return None 

    def initiate_local_training(self, epoch, clusters):
        self.model.set_cur_client(self.idx)
        if clusters != None:
            self.model.set_cur_lora_id_for_train(self.get_cluster_center(clusters, self.idx))
        # save init lora and mm_projector
        state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
        mm_projector_output_dir = os.path.join(self.client_ckpt, "llava_lora_finetune_epoch_{}".format(str(epoch)))
        if not os.path.exists(mm_projector_output_dir):
            os.makedirs(mm_projector_output_dir, exist_ok=True)
            self.model.config.save_pretrained(mm_projector_output_dir)
            self.model.save_pretrained(mm_projector_output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(mm_projector_output_dir, "init_mm_projector.bin"))


    def terminate_local_training(self, epoch, previously_selected_clients_set, init_clusters):
        if init_clusters is not None:
            lora_id = self.model.lora_id
            if lora_id >= len(init_clusters):
                init_clusters.extend([[-1]] * (lora_id - len(init_clusters) + 1))
            if init_clusters[lora_id] == [-1]:
                init_clusters[lora_id] = [self.idx]
            elif self.idx not in init_clusters[lora_id]:
                init_clusters[lora_id].append(self.idx)
        # save current client weights
        device = next(self.model.parameters()).device
        state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )
        single_output_dir = os.path.join(self.client_ckpt, "llava_lora_finetune_epoch_{}".format(str(epoch)), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(state_dict, single_output_dir + "/pytorch_model.bin")

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
        torch.save(non_lora_state_dict, os.path.join(single_output_dir, 'non_lora_trainables.bin'))

        # load init lora and mm_projector for next client
        mm_projector_output_dir = os.path.join(self.client_ckpt, "llava_lora_finetune_epoch_{}".format(str(epoch)), "init_mm_projector.bin")
        init_mm_projector = torch.load(mm_projector_output_dir, map_location=device)
        self.model.load_state_dict(init_mm_projector, strict=False)
        
        init_lora_path = os.path.join(self.client_ckpt, "llava_lora_finetune_epoch_{}".format(str(epoch)), "adapter_model.bin")
        adapters_weights = torch.load(init_lora_path, map_location=device)
        set_peft_model_state_dict(self.model, adapters_weights, adapter_name="default")
        
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})

        return self.model, previously_selected_clients_set, init_clusters