import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
from tqdm import tqdm

from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import *
from transformers.generation.utils import *


def init_weights(module, method='xavier'):
    if isinstance(module, nn.Linear):
        if method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        else:
            raise ValueError("Unsupported init method")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ==== MLP 模型定义 ====
class MLP(nn.Module):
    def __init__(self, input_dim, dim1=128, dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim1)
        self.bn1 = nn.BatchNorm1d(dim1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dim1, dim2)
        self.bn2 = nn.BatchNorm1d(dim2)
        self.dropout2 = nn.Dropout(0.3)
        # self.out = nn.Linear(dim2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return x
    
# ==== MLP 模型定义 ====
class MLP_out(nn.Module):
    def __init__(self, input_dim, dim1=128, dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim1)
        self.bn1 = nn.BatchNorm1d(dim1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dim1, dim2)
        self.bn2 = nn.BatchNorm1d(dim2)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(dim2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return self.out(x)

class LLM_with_mlp_query(nn.Module):
    '''
        input is input ids of text and the mlp rep -> on top of specific layer, a linear classifier for prediction
    '''
    def __init__(self, mlp_input_dim, mlp_dim1, llm, skip_layer=0):
        super().__init__()
        self.mlp = MLP(input_dim=mlp_input_dim, dim1=mlp_dim1, dim2=896)
        self.linear_out = nn.Linear(896, 2)
        # llm
        self.llm = llm
        self.skip_layer = skip_layer

    def get_llm_rep(self, **sample):
        hidden_states = self.llm(**sample, output_hidden_states=True)
        # hidden_states = output.hidden_states 
        # ipdb.set_trace()

        last_hidden_state = hidden_states[-self.skip_layer-1][:, -1, :] # (bs, dim)

        return last_hidden_state 
    
    def forward(self, x_tabular, input_ids, attention_mask):
        # ipdb.set_trace()
        mlp_rep = self.mlp(x_tabular)

        llm_rep = self.get_llm_rep(**{"input_ids":input_ids, "mlp_embeds":mlp_rep, "attention_mask":attention_mask})
        out = self.linear_out(llm_rep) # (bs, 2)
        return out

    
class MLP_only(nn.Module):
    '''
        directly concat llm feature with mlp input
    '''
    def __init__(self, mlp_input_dim, mlp_dim1, mlp_dim2, llm_proj_dim, llm_feat_tr, llm_feat_val, llm_feat_tst):
        super().__init__()
        mlp_input_dim = mlp_input_dim # + 896
        self.mlp = MLP_out(input_dim=mlp_input_dim, dim1=mlp_dim1, dim2=mlp_dim2)
        self.llm_feat_train = llm_feat_tr
        self.llm_feat_valid =  llm_feat_val
        self.llm_feat_test = llm_feat_tst

        # self.bn = nn.BatchNorm1d(mlp_dim2 + llm_proj_dim)

        self.linear = nn.Linear(896, llm_proj_dim)

    def forward(self, x, x_id, valid=False, test=False):
    
        out = self.mlp(x)
        
        return out

class Fusion_model_learnable(nn.Module):
    '''
        learnable fusion with real-time LLM feature extraction
    '''
    def __init__(self, mlp_input_dim, mlp_dim1, mlp_dim2, llm_proj_dim, llm_model):
        super().__init__()
        self.mlp = MLP(input_dim=mlp_input_dim, dim1=mlp_dim1, dim2=mlp_dim2)
        self.llm = llm_model
        
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
        
        self.linear_out = nn.Linear(mlp_dim2 + llm_proj_dim, 2)
        self.bn = nn.BatchNorm1d(mlp_dim2 + llm_proj_dim)
        self.linear = nn.Linear(896, llm_proj_dim)  # 896 is the LLM hidden dimension

    def forward(self, x_tabular, input_ids, attention_mask):
        # Process tabular data
        mlp_rep = self.mlp(x_tabular)
        
        # Extract LLM features in real-time
        with torch.no_grad():
            llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            llm_rep = llm_output.hidden_states[-1][:, -1, :]  # Get last token representation
        
        # Project LLM features
        llm_rep = self.linear(llm_rep)
        
        # Concatenate and fuse
        fusion_rep = torch.cat([mlp_rep, llm_rep], dim=1)
        fusion_rep = self.bn(fusion_rep)
        out = self.linear_out(fusion_rep)
        
        return out
    

class QwenWithReasoning(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        self.reasoning_steps = getattr(config, "reasoning_steps", 1)
        self.train_use_cache = getattr(config, "train_use_cache", True)

        self.yes_idx = 20412
        # self.yes_idx = torch.LongTensor([self.token_yes])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mlp_embeds: torch.FloatTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            mlp_embeds=mlp_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        

        return outputs.hidden_states
    
        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        # # positive_logits = logits[:, -1, self.yes_idx] # (bs,)
        # positive_logits = logits[:, -self.reasoning_steps-1:, :]  # (bs, vocab_size)
        # # positive_logits = logits[:, -1, :]  # (bs, vocab_size)

        # loss = None
        # if labels is not None:
        #     # labels = labels[:,-self.reasoning_steps:] if self.reasoning_steps>1 else labels[:, -1].unsqueeze(1)  #(bs,) if self
            
        #     labels = labels[:, -1].unsqueeze(1).repeat(1, (self.reasoning_steps + 1))
        #     loss = self.loss_function(logits=positive_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        #     # ipdb.set_trace()
        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=positive_logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

class Qwen2Model_Query(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.query_vector = nn.Embedding(config.reasoning_steps, self.config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mlp_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # concat mlp vector and query_vector 
        mlp_embeds = mlp_embeds.unsqueeze(1) # (bs, 1, dim)
        inputs_embeds = torch.cat([inputs_embeds, mlp_embeds, self.query_vector.weight.unsqueeze(0).repeat(inputs_embeds.shape[0],1,1)], dim=1) #(bs, seq_len+n_query+1, dim)
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()