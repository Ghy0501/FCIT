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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.training = False
        self.cur_task = 0
        self.lora_id = self.cur_task
        self.expert_num = 8
        self.global_text_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.global_image_feature = nn.ParameterList(
                [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.cluster_text_size = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)) for _ in range(8)]
            )
        self.lora_id_signal = False

    def set_cur_task(self, cur_task, expert_num):
        self.cur_task = cur_task
        self.expert_num = expert_num

    def update_global_feature(self, cluster_text_feature, cluster_image_feature, cluster_text_size):
        for i, feature in enumerate(cluster_text_feature):
            self.global_text_feature[i].data.copy_(feature.to(self.global_text_feature[i].dtype))
        for i, feature in enumerate(cluster_image_feature):
            self.global_image_feature[i].data.copy_(feature.to(self.global_image_feature[i].dtype))
        for i, num in enumerate(cluster_text_size):
            self.cluster_text_size[i].data.copy_(torch.tensor(num, dtype=self.cluster_text_size[i].dtype))
            
        for param in self.global_text_feature:
            param.requires_grad = True
        for param in self.global_image_feature:
            param.requires_grad = True
        for param in self.cluster_text_size:
            param.requires_grad = True
    
    def set_cur_client(self, client_idx):
        self.client_idx = client_idx
    
    def set_cur_lora_id_for_train(self, lora_id):
        self.lora_id_signal = True
        self.lora_id = lora_id

    def get_model(self):
        return self.model

    def set_fearure_dict(self, size):
        self.text_features_dict = {i: [] for i in range(size)}
        self.image_features_dict = {i: [] for i in range(size)}

    def set_eval(self, num_task):
        self.expert_num = num_task

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
