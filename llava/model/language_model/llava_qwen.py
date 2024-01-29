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
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from transformers.utils import ModelOutput

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
from .qwen.configuration_qwen import QWenConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from transformers import PretrainedConfig


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LlavaQWenConfig(QWenConfig):
    model_type = "llava_qwen"


class LlavaQWenModel(LlavaMetaModel, QWenModel):
    config_class = LlavaQWenConfig

    def __init__(self, config: QWenConfig):
        super(LlavaQWenModel, self).__init__(config)


class LlavaQWenForCausalLM(QWenLMHeadModel, LlavaMetaForCausalLM):
    config_class = LlavaQWenConfig

    def __init__(self, config):
        super(LlavaQWenForCausalLM, self).__init__(config)
        self.transformer = LlavaQWenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.mft_cnt = 0
        self.prev_image_features = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, num_dataset
    ):
        if past_key_values is None and torch.any(input_ids == self.get_vision_tower().vision_config['image_start_id']):
            bos_pos = torch.where(input_ids == self.get_vision_tower().vision_config['image_start_id'])
            eos_pos = torch.where(input_ids == self.get_vision_tower().vision_config['image_start_id'] + 1)
            assert (bos_pos[0] == eos_pos[0]).all()
            img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
            images = self.get_vision_tower()(images)
            images = self.get_model().mm_projector(images)
            assert images.shape[0] == len(images)
        else:
            images = None
            img_pos = None
        return img_pos, images

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        num_dataset: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast, LlavaCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #import pdb; pdb.set_trace()
        img_pos, images = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, num_dataset)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            images=images,
            img_pos=img_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            labels=labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
        

AutoConfig.register("llava_qwen", LlavaQWenConfig)
AutoModelForCausalLM.register(LlavaQWenConfig, LlavaQWenForCausalLM)
