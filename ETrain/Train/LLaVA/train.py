# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from dataclasses import dataclass, field
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers
import subprocess
import numpy as np

from ETrain.utils.LLaVA.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from torch.utils.data import Dataset
from ETrain.Train.LLaVA.llava_trainer import LLaVATrainer

from ETrain.Models.LLaVA import *
from ETrain.Dataset import create_LLaVA_data_module
from ETrain.Dataset.dataset import DataArguments
from ETrain.Train.Base_trainer import *
from ETrain.Train.LLaVA.llava_trainer import load_model_from_previous_task
from ETrain.Fed_utils import *
from peft import get_peft_model, set_peft_model_state_dict, prepare_model_for_kbit_training
from CoIN.peft.utils import get_peft_model_state_dict


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_text_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    num_clients: int = 50
    num_communication_rounds: int = 10
    fcit_type: str = "seq"
    client_selection_frac: float = 0.1
    fed_alg: str = field(default="fedavg")
    prox_mu: Optional[float] = field(default=0.01)
    fedopt_tau: Optional[float] = field(default=1e-3)
    fedopt_eta: Optional[float] = field(default=1e-3)
    fedopt_beta1: Optional[float] = field(default=0.9)
    fedopt_beta2: Optional[float] = field(default=0.99)
    use_dyn_list: bool = True

cap_dyn_list = [[0,1,1,3,3],
                [0,2,2,2,3],
                [1,2,2,3,3],
                [0,0,0,1,1]]

seq_dyn_list = [[1, 2, 4, 5, 6],
                [3, 4, 5, 7, 7],
                [0, 2, 3, 4, 5],
                [0, 0, 1, 3, 7],
                [1, 2, 4, 4, 5],
                [0, 3, 5, 7, 7],
                [2, 2, 5, 7, 7],
                [5, 5, 5, 6, 6]]

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.text_tower,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    model, tokenizer = create_LLaVA_model(training_args, model_args, data_args, bnb_model_from_pretrained_args, compute_dtype, local_rank)

    model.set_cur_task(model_args.cur_task, model_args.expert_num)
    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)

    if model_args.previous_task_model_path is not None:
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)

    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    proxy_dict, opt_proxy_dict = get_proxy_dict(training_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(training_args, global_dict)

    print("Load and prepare Federated continual instruction tuning data...")
    FCIT_dataset = FCIT_data(fcit_type=training_args.fcit_type, alpha=data_args.alpha)
    FCIT_dataset.load_data(data_args, training_args.num_clients)
    previously_selected_clients_set = set()

    if 'dyn' in training_args.fcit_type:   #cap
        if training_args.use_dyn_list and 'cap' in training_args.fcit_type:
            set_for_current_task = cap_dyn_list[model_args.cur_task]
        elif training_args.use_dyn_list and 'seq' in training_args.fcit_type:
            set_for_current_task = seq_dyn_list[model_args.cur_task]
        else:
            np.random.seed(model_args.cur_task * 10)
            set_for_current_task = sorted(np.random.choice(np.arange(model_args.expert_num), size=int(training_args.num_clients * training_args.client_selection_frac), replace=True))
        print("dynamic domain distribution for this task:{}".format(set_for_current_task))
    else:
        set_for_current_task = None
    if not any(param.data.sum() != 0 for param in model.cluster_text_size):
        cluster_text_feature = None
        cluster_image_feature = None
        cluster_text_size = None
    else:
        cluster_text_feature = [param.detach().clone().to(training_args.device) for param in model.global_text_feature]
        cluster_image_feature = [param.detach().clone().to(training_args.device) for param in model.global_image_feature]
        cluster_text_size = [param.item() for param in model.cluster_text_size]
        print(cluster_text_size)
        
    clusters = None
    init_clusters = None

    if clusters is None and cluster_text_size is not None: 
        init_clusters = [[-1] for size in cluster_text_size if size != 0]
    for epoch in range(training_args.num_communication_rounds):
        print('Train on epoch:{}'.format(epoch))
        print("Conducting the client selection")
        selected_clients_set = client_selection(training_args.num_clients, training_args.client_selection_frac, "random", previously_selected_clients_set,
                                                other_info=(model_args.cur_task * 10) + epoch)
        print("Selected clients for this round:{}".format(selected_clients_set))
        client_weight = []
        model.set_fearure_dict(int(training_args.num_clients * training_args.client_selection_frac))
        for idx, client_id in enumerate(selected_clients_set):
            client = GeneralClient(training_args, client_id, model, tokenizer, set_for_current_task, idx, training_args.output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client_weight = client.prepare_client_dataset(data_args, training_args.fcit_type, model_args.cur_task, tokenizer, local_rank, client_weight)
            client.build_local_trainer(global_dict=global_dict, global_auxiliary=global_auxiliary, local_auxiliary=auxiliary_model_list[client_id])

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training(epoch, clusters)

            print("Training of Client_{}".format(client_id))
            client.train()

            if training_args.fed_alg == 'scaffold':
                auxiliary_model_list[client_id], auxiliary_delta_dict[client_id] = client.local_trainer.get_auxiliary_param()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, previously_selected_clients_set, init_clusters = client.terminate_local_training(epoch, previously_selected_clients_set, init_clusters)
            print(init_clusters)
            del client
            torch.cuda.empty_cache()

            remove_dir = training_args.output_dir
            subprocess.run(f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +", shell=True)
        
        # if clusters is None and init_clusters is not None:
        #     clusters = init_clusters
        cluster_text_feature, cluster_image_feature, cluster_text_size, clusters = process_text_feature(model, client_weight, cluster_text_feature, cluster_image_feature, cluster_text_size, clusters, init_clusters)
        print(cluster_text_size)
        print("Cluster result:{}".format(clusters))
        if epoch == 0 and model_args.cur_task == 0:
            lora_weight_id = [[i for i in range(int(training_args.num_clients * training_args.client_selection_frac))]]
        else:
            lora_weight_id = clusters
        print("Collecting the weights of clients and performing aggregation")
        print("Using FedAvg!")
        model = FedAvg(model,
                    selected_clients_set,
                    training_args.output_dir,
                    client_weight,
                    epoch,
                    clusters,
                    lora_weight_id,
                    )

        model.update_global_feature(cluster_text_feature, cluster_image_feature, cluster_text_size)

        if training_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                save_dir = os.path.join(training_args.output_dir, "llava_lora_finetune_epoch_{}".format(str(epoch)))
                model.config.save_pretrained(save_dir)
                model.save_pretrained(save_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(save_dir, 'non_lora_trainables.bin'))

        remove_dir = training_args.output_dir
        subprocess.run(f"find {remove_dir} -maxdepth 1 -type d -name 'llava_lora_finetune_epoch_{epoch-1}' -exec rm -rf {{}} +", shell=True)


if __name__ == "__main__":
    train()
