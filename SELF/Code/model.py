import torch
import torch.nn as nn
from transformers import AutoModel, RobertaPreTrainedModel
from torch.cuda.amp import autocast

from roberta import RobertaModel
from KLloss import KLDivLoss
from typing import Any, Optional, Tuple
from transformers.activations import ACT2FN

class HierachicalFeatureRetainer(nn.Module):
    def __init__(self, config, aggred_layer_num=1):
        super().__init__()
        self.config = config
        # self.n_experts = aggred_layer_num
        self.n_experts = 13
        self.experts1 = nn.Parameter(torch.Tensor(self.n_experts, 2 * config.hidden_size, config.hidden_size),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform(self.experts1.data)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.experts2 = nn.Parameter(torch.Tensor(self.n_experts, config.hidden_size, config.hidden_size),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform(self.experts2.data)

        self.expert_router = nn.Linear((1 + self.n_experts) * config.hidden_size, self.n_experts + 1)

    def forward(self,
                last_hidden_state: torch.Tensor = None,
                hidden_states: Tuple = None,
                attention_mask: torch.Tensor = None, ):
        expert_router_input = torch.cat((last_hidden_state,
                                         hidden_states[0],
                                         hidden_states[1],
                                         hidden_states[2],
                                         hidden_states[3],
                                         hidden_states[4],
                                         hidden_states[5],
                                         hidden_states[6],
                                         hidden_states[7],
                                         hidden_states[8],
                                         hidden_states[9],
                                         hidden_states[10],
                                         hidden_states[11],
                                         hidden_states[12],
                                         ), dim=-1)

        # print(expert_router_input.shape)
        expert_weights = self.expert_router(expert_router_input)
        expert_weights = torch.softmax(expert_weights, dim=-1)
        bsz, tgt_len, embed_dim = last_hidden_state.size()

        hidden_states = torch.stack((hidden_states[0],
                                         hidden_states[1],
                                         hidden_states[2],
                                         hidden_states[3],
                                         hidden_states[4],
                                         hidden_states[5],
                                         hidden_states[6],
                                         hidden_states[7],
                                         hidden_states[8],
                                         hidden_states[9],
                                         hidden_states[10],
                                         hidden_states[11],
                                         hidden_states[12],
                                         ), dim=2)

        input_hidden_states = torch.cat((last_hidden_state.unsqueeze(2).expand(hidden_states.shape),
                                         hidden_states), dim=-1)
        experts_out = torch.einsum('bsni,nio->bsno', input_hidden_states, self.experts1)

        experts_out = self.activation_fn(experts_out)
        experts_out = torch.einsum('bsni,nio->bsno', experts_out, self.experts2)

        text_hidden_states = last_hidden_state.unsqueeze(2)
        experts_out = experts_out + text_hidden_states.expand(hidden_states.shape)

        experts_out = torch.cat([text_hidden_states, experts_out], dim=2)
        experts_out = torch.einsum('bsno,bsn->bsno', experts_out, expert_weights)
        TextFeature_fusion = torch.sum(experts_out, dim=2)

        return TextFeature_fusion, expert_weights

class REModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config=config)
        hidden_size = config.hidden_size  # 768
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(hidden_size, config.num_class)
        )

        self.KL_loss = KLDivLoss(T=1)
        config.hidden_act = 'gelu'
        self.expert = HierachicalFeatureRetainer(config)

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None, entity_mask=None, entity_label=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
            output_attentions=True,
            output_hidden_states=True
        )

        # output attention
        attentions = outputs['attentions']
        attention = attentions[-1]  # batch_size * 12 * sentence_len * sentence_len
        num = attention.shape[-1]  # sentence_len
        attention = torch.sum(attention, dim=1) / 12  # batch_size * sentence_len * sentence_len
        attention = torch.sum(attention, dim=1) / num  # batch_size * sentence_len

        # pooled_output = outputs[0]
        pooled_output, expert_weights = self.expert(outputs['last_hidden_state'], outputs['hidden_states'])
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            # multitask learning, KL_loss used to calculate attentional loss
            loss = self.loss_fnt(logits.float(), labels) + self.KL_loss(attention, entity_label)
            # loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs




