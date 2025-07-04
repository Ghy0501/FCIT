#    Copyright 2023 Haotian Liu
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


import os, sys
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ETrain.Models.LLaVA import *
from ETrain.utils.LLaVA.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

sys.path.append('/home/chencheng/Code/Slim_Train/')

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", num_task=8, text_tower=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            clip_tokenizer = AutoTokenizer.from_pretrained(
                text_tower,
                cache_dir=None,
                model_max_length=77,
                padding_side="right",
                use_fast=True,
            )

            model.set_clip_tokenizer(clip_tokenizer)
            model.set_tokenizer(tokenizer)
            model.set_eval(num_task)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            # if 'MOE' in model_name:
            from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            # else:
            #     from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        text_tower = model.get_text_tower()
        if not text_tower.is_loaded:
            text_tower.load_model()
        text_tower.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_and_merge_pretrained_model(model_paths, model_base, model_name, save_model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            
            model_path = model_paths[0] # loading mm_prejector, just for convenience 
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            
            print('Loading additional LLaVA weights...')        
            merged_non_lora_trainables = {}
            coef = 1/len(model_paths)
            # coef = 0.2

            for i, sub_model_path in enumerate(model_paths):
                assert os.path.exists(os.path.join(sub_model_path, 'non_lora_trainables.bin')), 'must load from local dir'
                non_lora_trainables = torch.load(os.path.join(sub_model_path, 'non_lora_trainables.bin'), map_location='cpu')
                
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                for name, param in non_lora_trainables.items():
                    if i==0:
                        merged_non_lora_trainables[name] = coef * param
                    else:
                        merged_non_lora_trainables[name] = merged_non_lora_trainables[name] + coef * param
            # model.load_state_dict(merged_non_lora_trainables, strict=False)
            
            torch.save(merged_non_lora_trainables, os.path.join(save_model_path,'non_lora_trainables.bin'))

            # # --------------------------------------################################################################################################ --------------------------------------
            # # load, merge and save lora weights v7: merge + prune + scaling, advanced implementation 
            # print('load, merge and save lora weights v7: merge + prune + scaling')
            # fuse_weight = [0.2 for _ in range(len(model_paths))]
            # concat_model_weights, merged_model_weights = {}, {}
            # for i, sub_model_path in enumerate(model_paths):
            #     lora_parameters = torch.load(os.path.join(sub_model_path, 'adapter_model.bin'), map_location='cpu')
            #     for name, param in lora_parameters.items():
            #         if i == 0:
            #             concat_model_weights[name] = param[None,:]
            #         else:
            #             concat_model_weights[name] = torch.concat((concat_model_weights[name],param[None,:]),dim=0)
            
            # mode = 'b from a'
            # assert mode in ['b from a', 'b'], 'not implemented yet.'
            # for name, concat_model_weight in concat_model_weights.items():
            #     T, d1, d2 = concat_model_weight.shape
            #     mask_ratio = 0.2

            #     kth_values, _ = concat_model_weight.reshape(-1, d1*d2).abs().kthvalue(int(T * (1-mask_ratio)), dim=0, keepdim=True)
            #     masks = concat_model_weight.reshape(-1,d1*d2).abs() >= kth_values
            #     masks = masks.reshape(T, d1, d2)
            #     trimed_model_weights = masks * concat_model_weight
                
            #     # model_weight_sign = torch.where(torch.sum(trimed_model_weights,dim=0)>0, torch.tensor(1.0, dtype=param.dtype,device=param.device), torch.tensor(-1.0, dtype=param.dtype,device=param.device))
            #     # assert (torch.sum(trimed_model_weights,dim=0).reshape(-1)==0).sum() == 0 , 'zero value exists.'
                
            #     for i, mask, trimed_param, param in zip(range(len(model_paths)),masks, trimed_model_weights,concat_model_weights[name]):
            #         if mode == 'b':
            #             if 'lora_B' in name:
            #                 # s_vector = []
            #                 # assert torch.isinf(torch.sum(mask,dim=1)).any() == False, 'contain inf tensor'
            #                 # s_vector = torch.sum(abs(param),dim=(0,1))/ torch.sum(abs(mask * param), dim=(0,1))
            #                 s_vector = torch.sum(abs(param),dim=0)/ torch.sum(abs(mask * param), dim=0)
            #                 # current_param = torch.matmul(s_vector, current_param)
            #                 if i==0:
            #                     merged_model_weights[name]=fuse_weight[i] * s_vector * trimed_param
            #                 else:
            #                     merged_model_weights[name]+=fuse_weight[i] * s_vector * trimed_param
            #             else:
            #                 if i==0:
            #                     merged_model_weights[name]=fuse_weight[i] * param
            #                 else:
            #                     merged_model_weights[name]+=fuse_weight[i] * param
            #         elif mode == 'b from a':
            #             if 'lora_A' in name:
            #                 # s_vector = []
            #                 # assert torch.isinf(torch.sum(mask,dim=1)).any() == False, 'contain inf tensor'
            #                 # s_vector = torch.sum(abs(param),dim=(0,1))/ torch.sum(abs(mask * param), dim=(0,1))
            #                 s_vector = torch.sum(abs(param),dim=1)/ torch.sum(abs(mask * param), dim=1)
            #                 # current_param = torch.matmul(s_vector, current_param)
            #                 if i==0:
            #                     merged_model_weights[name]=fuse_weight[i] * param
            #                 else:
            #                     merged_model_weights[name]+=fuse_weight[i] * param
            #             else:
            #                 if i==0:
            #                     merged_model_weights[name]=fuse_weight[i] * s_vector * trimed_param
            #                 else:
            #                     merged_model_weights[name]+=fuse_weight[i] * s_vector * trimed_param

            print("load, merge and save lora weights v1: simple merge")
            fuse_weight = [0.25 for _ in range(len(model_paths))]
            merged_model_weights = {}
            print(fuse_weight)
            for i, sub_model_path in enumerate(model_paths):
                print(sub_model_path)
                lora_parameters = torch.load(os.path.join(sub_model_path, 'non_lora_trainables.bin'), map_location='cpu')
                for name, param in lora_parameters.items():
                    if 'mm_projector' in name:
                        print(name)
                        if i==0:
                            merged_model_weights[name] = fuse_weight[i] * param
                        else:
                            merged_model_weights[name] = merged_model_weights[name] + fuse_weight[i] *param
                    else:
                        print(name)
                        merged_model_weights[name] = param

            # print("Load, merge and save LoRA weights v2: Concatenate along smaller dimension")

            # merged_model_weights = {}

            # # Load all LoRA weights at once
            # lora_weights_list = [
            #     torch.load(os.path.join(path, 'adapter_model.bin'), map_location='cpu') 
            #     for path in model_paths
            # ]

            # # Iterate over each parameter and concatenate along the smaller dimension
            # for name in lora_weights_list[0].keys():
            #     # Initialize the merged parameter list for the current name
            #     merged_param = None
            #     for i in range(len(model_paths)):
            #         param = lora_weights_list[i][name]
                    
            #         # If merged_param is None, initialize it with the current parameter
            #         if merged_param is None:
            #             merged_param = param
            #         else:
            #             # Compare the smaller dimension (0 or 1) and concatenate along that dimension
            #             if param.size(0) < param.size(1):
            #                 # Concatenate along dimension 0
            #                 merged_param = torch.cat([merged_param, param], dim=0)
            #             else:
            #                 # Concatenate along dimension 1
            #                 merged_param = torch.cat([merged_param, param], dim=1)

            #     # Store the merged parameter
            #     merged_model_weights[name] = merged_param

            # print("Weights concatenated and merged successfully.")

            model.config.save_pretrained(save_model_path)
            torch.save(merged_model_weights, os.path.join(save_model_path,'non_lora_trainables.bin'))
            print('finish saving merged model weights.')

            from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, save_model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
