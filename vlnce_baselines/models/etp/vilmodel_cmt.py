import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from vlnce_baselines.common.ops import create_transformer_encoder
from vlnce_baselines.common.ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad


logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

        # Dynamic graph related parameters
        self.use_dynamic_graph = getattr(config, 'use_dynamic_graph', False)
        if self.use_dynamic_graph:
            # Three learnable scalar weights: w1(geometric), w2(semantic), w3(instruction)
            # w2 and w3 are initialized close to 0, ensuring initial state equals original ETPNav
            # Note: w1 is fixed to 1, not a learnable parameter (geometric distance as foundation, unchangeable)
            # But still defined as Parameter (requires_grad=False) for recording and monitoring
            self.w1 = nn.Parameter(torch.ones(1), requires_grad=False)  # Geometric weight, fixed to 1 (as foundation)
            self.w2 = nn.Parameter(torch.tensor(0.2))  # Semantic weight, initialized to 0.1
            self.w3 = nn.Parameter(torch.tensor(0.2))  # Instruction weight, initialized to 0.1
            
            # Geometric Dropout configuration
            # During training, force geometric distance to 0 with a certain probability, forcing the model to rely on semantic and instruction information
            self.use_geo_dropout = getattr(config, 'use_geo_dropout', False)
            self.geo_dropout_prob = getattr(config, 'geo_dropout_prob', 0.3)  # Default 30% probability
            
            # Geometry-Conditioned Semantic Edge configuration
            # Strategy 2: Use geometric distance information as conditional input to semantic similarity MLP
            self.use_geo_conditioned_semantic = getattr(config, 'use_geo_conditioned_semantic', False)
            
            # Ablation study configuration: control whether to use Esem and Einst
            self.use_esem = getattr(config, 'use_esem', True)  # Default enable semantic edge
            self.use_einst = getattr(config, 'use_einst', True)  # Default enable instruction edge
            
            # Distance embedding layer: upscale 1D scalar distance to 64D, avoid being overwhelmed by visual feature noise
            # Only created when geometry-conditioned semantic edge is enabled
            if self.use_geo_conditioned_semantic:
                self.dist_embed = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.ReLU(),
                    nn.LayerNorm(64)  # Adding Norm works better
                )
            
            # Semantic similarity MLP: input two nodes' visual features (optional: + geometric distance embedding), output similarity (0~1)
            # If geometry-conditioned is enabled: input [2*D + 64] (two node features + 64D geometric distance embedding)
            # If not enabled: input [2*D] (two node features)
            semantic_input_dim = config.hidden_size * 2
            if self.use_geo_conditioned_semantic:
                semantic_input_dim += 64  # Add geometric distance embedding dimension (upscaled from 1D to 64D)
            self.semantic_sim_mlp = nn.Sequential(
                nn.Linear(semantic_input_dim, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                # nn.Tanh()  # Limit to [-1, 1]
            )

            # Instruction relevance MLP: input node features + instruction global features, output instruction relevance score (0~1)
            # Input: node features + instruction global features [D+D], output: scalar score
            self.instruction_rel_mlp = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                # nn.Tanh()  # Limit to [-1, 1]
            )

        # Node gating related parameters
        self.use_node_gating = getattr(config, 'use_node_gating', False)
        if self.use_node_gating:
            # Node gating MLP: input node features + instruction global features, output gating coefficient (0~1)
            # Input: node features + instruction global features [D+D], output: scalar gating coefficient
            self.node_gating_mlp = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()  # Ensure output is in [0, 1]
            )
            last_layer = self.node_gating_mlp[-2]
            nn.init.constant_(last_layer.weight,0.0)
            nn.init.constant_(last_layer.bias,-5.0)

    def _get_lang_global(self, lang_feats, lang_attention_mask=None):
        """
        Extract global features of instruction (Masked Mean)
        
        Args:
            lang_feats: (N, L_t, D) instruction features
            lang_attention_mask: language attention mask, may be in various formats
            
        Returns:
            lang_global: (N, D) global features of instruction
        """
        if lang_attention_mask is not None:
            # Handle different mask shapes
            # extend_neg_masks output: (N, 1, 1, L_t), padding positions are -10000.0, valid positions are 0.0
            # Or may be: (N, H, L_t, L_t)
            if lang_attention_mask.dim() == 4:
                # (N, H, L_t, L_t) or (N, 1, 1, L_t)
                if lang_attention_mask.shape[1] == 1 and lang_attention_mask.shape[2] == 1:
                    # (N, 1, 1, L_t) - output of extend_neg_masks
                    lang_mask = lang_attention_mask[:, 0, 0, :]  # (N, L_t)
                else:
                    # (N, H, L_t, L_t) - take first head and first query position
                    lang_mask = lang_attention_mask[:, 0, 0, :]  # (N, L_t)
            elif lang_attention_mask.dim() == 2:
                # (N, L_t) - original mask
                lang_mask = lang_attention_mask
            else:
                # Other shapes, try to extract
                lang_mask = lang_attention_mask.squeeze()
                if lang_mask.dim() > 1:
                    lang_mask = lang_mask[0] if lang_mask.shape[0] == 1 else lang_mask
            
            # Convert mask to bool type, True indicates valid token
            # extend_neg_masks: padding=-10000.0, valid=0.0
            # Original mask: padding=False/0, valid=True/1
            if lang_mask.dtype == torch.bool:
                valid_mask = lang_mask  # (N, L_t)
            else:
                # If output of extend_neg_masks, -10000.0 indicates padding, 0.0 indicates valid
                # If original mask, >=0 indicates valid
                if (lang_mask < -1000).any():  # May be output of extend_neg_masks
                    valid_mask = lang_mask >= -1000  # (N, L_t) - values greater than -1000 are valid
                else:
                    valid_mask = lang_mask >= 0  # (N, L_t) - non-negative values are valid
            
            # Masked Mean: only average valid tokens
            # Set features at padding positions to 0
            lang_feats_masked = lang_feats * valid_mask.unsqueeze(-1)  # (N, L_t, D)
            # Calculate number of valid tokens
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (N, 1)
            # Avoid division by zero
            valid_counts = torch.clamp(valid_counts, min=1.0)
            # Calculate masked mean
            lang_global = lang_feats_masked.sum(dim=1) / valid_counts  # (N, D)
        else:
            # If no mask, use ordinary mean (backward compatibility)
            lang_global = torch.mean(lang_feats, dim=1)  # (N, D)
        
        return lang_global

    def apply_node_gating(self, visn_feats, lang_feats, lang_attention_mask=None):
        """
        Apply node gating mechanism
        
        Args:
            visn_feats: (N, L_v, D) visual node features (output after Cross-Attn)
            lang_feats: (N, L_t, D) instruction features
            lang_attention_mask: language attention mask
            
        Returns:
            gated_visn_feats: (N, L_v, D) gated visual node features
        """
        N, L_v, D = visn_feats.shape
        
        # 1. Get global features of instruction
        lang_global = self._get_lang_global(lang_feats, lang_attention_mask)  # (N, D)
        
        # 2. Concatenate each node feature with instruction global features: (N, L_v, 2*D)
        node_gating_feats = torch.cat([
            visn_feats,  # (N, L_v, D)
            lang_global.unsqueeze(1).expand(N, L_v, D)  # (N, L_v, D)
        ], dim=-1)  # (N, L_v, 2*D)
        
        # 3. Get gating coefficient for each node through MLP: (N, L_v, 1) -> (N, L_v)
        gate_values = self.node_gating_mlp(node_gating_feats).squeeze(-1)  # (N, L_v)
        
        # 4. Apply residual gating: V_new = V_old + (V_old × Gate)
        # When Gate=0: V_new = V_old (keep original)
        # When Gate=1: V_new = 2×V_old (amplify by 2x)
        gate_values = gate_values.unsqueeze(-1)  # (N, L_v, 1) for broadcasting
        gated_visn_feats = visn_feats + (visn_feats * gate_values)  # (N, L_v, D)
        
        return gated_visn_feats

    def compute_dynamic_edges(self, visn_feats, lang_feats, graph_sprels, lang_attention_mask=None):
        """
        Compute dynamic graph edge weights
        
        Args:
            visn_feats: (N, L_v, D) visual node features
            lang_feats: (N, L_t, D) instruction features
            graph_sprels: (N, 1, L_v, L_v) original geometric distance matrix (real distance, used for MLP condition)
            lang_attention_mask: (N, L_t) language attention mask, True indicates valid token, False indicates padding
            
        Returns:
            dynamic_sprels: (N, 1, L_v, L_v) dynamic graph edge weight matrix
        """
        N, L_v, D = visn_feats.shape
        device = visn_feats.device
        
        # E_geo: original geometric distance matrix (N, 1, L_v, L_v)
        # Note: save original distance here for subsequent Dropout processing
        E_geo = graph_sprels
        
        # Save original geometric distance for MLP condition (even if Dropout is triggered, MLP should see real distance)
        geo_dist_for_mlp = graph_sprels
        
        # ----------------------------------------------------------------------
        # 1. Compute semantic similarity matrix Esem (N, L_v, L_v)
        # ----------------------------------------------------------------------
        # Use broadcasting to compute semantic similarity between all node pairs
        # visn_feats: (N, L_v, D) -> (N, L_v, 1, D)
        visn_feats_i = visn_feats.unsqueeze(2)  # (N, L_v, 1, D)
        # visn_feats: (N, L_v, D) -> (N, 1, L_v, D)
        visn_feats_j = visn_feats.unsqueeze(1)  # (N, 1, L_v, D)
        # Broadcast to (N, L_v, L_v, D)
        visn_feats_i_expanded = visn_feats_i.expand(N, L_v, L_v, D)
        visn_feats_j_expanded = visn_feats_j.expand(N, L_v, L_v, D)
        # Concatenate: (N, L_v, L_v, 2*D)
        pair_feats = torch.cat([visn_feats_i_expanded, visn_feats_j_expanded], dim=-1)
        
        # Strategy 2: Geometry-conditioned semantic edge - use geometric distance information as conditional input
        # Important: use real geometric distance (geo_dist_for_mlp), even if Dropout is triggered, use real distance
        if hasattr(self, 'use_geo_conditioned_semantic') and self.use_geo_conditioned_semantic:
            # geo_dist_for_mlp: (N, 1, L_v, L_v) -> (N, L_v, L_v, 1)
            geo_dist = geo_dist_for_mlp.squeeze(1).unsqueeze(-1)  # (N, L_v, L_v, 1)
            
            # [Key optimization] First upscale through distance embedding layer, then concatenate
            # Upscale 1D scalar distance to 64D, avoid being overwhelmed by 1536D visual feature noise
            geo_embed = self.dist_embed(geo_dist)  # (N, L_v, L_v, 64)
            
            # Concatenate to visual features: (N, L_v, L_v, 2*D + 64)
            pair_feats = torch.cat([pair_feats, geo_embed], dim=-1)
        
        # Get similarity through MLP: (N, L_v, L_v, 1) -> (N, L_v, L_v)
        sim_logits = self.semantic_sim_mlp(pair_feats).squeeze(-1)
        # Ensure symmetry (optional, but helps stability)
        Esem = torch.tanh(sim_logits)
        # Add dimension to match graph_sprels shape: (N, 1, L_v, L_v)
        Esem = Esem.unsqueeze(1)
        
        # ----------------------------------------------------------------------
        # 2. Compute instruction relevance matrix Einst (N, 1, L_v, L_v) - optimized version
        # ----------------------------------------------------------------------
        # Optimization: first compute each node's relevance score to instruction (O(N)), then generate edge weights through matrix multiplication (O(N^2))
        # Instead of computing for each node pair (O(N^2))
        
        # 2.1 Compute global features of instruction (Masked Mean) - use helper function
        lang_global = self._get_lang_global(lang_feats, lang_attention_mask)  # (N, D)
        
        # 2.2 Compute each node's relevance score to instruction (O(N) operation)
        # Concatenate each node feature with instruction global features: (N, L_v, 2*D)
        node_inst_feats = torch.cat([
            visn_feats,  # (N, L_v, D)
            lang_global.unsqueeze(1).expand(N, L_v, D)  # (N, L_v, D)
        ], dim=-1)  # (N, L_v, 2*D)
        
        # Get each node's instruction relevance score through MLP: (N, L_v, 1) -> (N, L_v)
        node_inst_logits = self.instruction_rel_mlp(node_inst_feats).squeeze(-1)  # (N, L_v)
        node_inst_scores = torch.tanh(node_inst_logits)
        
        # 2.3 Generate edge weight matrix through matrix multiplication (O(N^2) operation, but more efficient than before)
        # Multiply two nodes' scores to get edge's instruction relevance: (N, L_v) × (N, L_v) -> (N, L_v, L_v)
        # node_inst_scores: (N, L_v) -> (N, L_v, 1)
        node_inst_scores_i = node_inst_scores.unsqueeze(2)  # (N, L_v, 1)
        # node_inst_scores: (N, L_v) -> (N, 1, L_v)
        node_inst_scores_j = node_inst_scores.unsqueeze(1)  # (N, 1, L_v)
        # Broadcast multiply: (N, L_v, 1) * (N, 1, L_v) -> (N, L_v, L_v)
        Einst = node_inst_scores_i * node_inst_scores_j  # (N, L_v, L_v)
        # Add dimension: (N, 1, L_v, L_v)
        Einst = Einst.unsqueeze(1)
        
        # ----------------------------------------------------------------------
        # 3. Strategy 1: Geometric Dropout - handled inside function
        # ----------------------------------------------------------------------
        # During training, set E_geo to 0 with a certain probability, forcing the model to rely on semantic and instruction information
        # Note: Dropout only acts on final fused E_geo, does not affect MLP's conditional input (real distance)
        if hasattr(self, 'use_geo_dropout') and self.use_geo_dropout:
            if self.training and torch.rand(1).item() < self.geo_dropout_prob:
                # Simulate sensor failure, set E_geo to 0, force model to look at semantic and instruction
                E_geo = torch.zeros_like(E_geo)
        
        # ----------------------------------------------------------------------
        # 4. Fusion: Edynamic = E_geo + (w2 * Esem) + (w3 * Einst)
        # ----------------------------------------------------------------------
        # Use additive enhancement, E_geo as foundation, Esem and Einst as fine-tuning
        # If Dropout is triggered, E_geo is 0, model is forced to rely on Esem and Einst
        # Ablation study: can control whether to include Esem and Einst through configuration
        Edynamic = E_geo
        if self.use_esem:
            Edynamic = Edynamic + (self.w2 * Esem)
        if self.use_einst:
            Edynamic = Edynamic + (self.w3 * Einst)
        
        return Edynamic

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        # Node gating: apply after Cross-Attn and before Self-Attn
        if self.use_node_gating:
            visn_att_output = self.apply_node_gating(
                visn_att_output, lang_feats, lang_attention_mask
            )

        if graph_sprels is not None:
            # If dynamic graph is enabled, compute dynamic graph edge weights
            if self.use_dynamic_graph:
                # Note: always pass original graph_sprels (real distance)
                # Dropout logic is handled inside compute_dynamic_edges, ensuring MLP can see real distance
                graph_sprels = self.compute_dynamic_edges(
                    visn_feats, lang_feats, graph_sprels, lang_attention_mask
                )
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        if config.use_depth_embedding:
            self.dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size)
            self.dep_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.dep_linear = self.dep_layer_norm = None

        # if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
        #     self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
        #     self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # else:
        #     self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable
        self.nav_type_embedding = nn.Embedding(2, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
       
class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        self.global_encoder = GlobalMapEncoder(config)
        self.global_sap_head = NextActionPrediction(self.config.hidden_size, 0.1)
        
        self.init_weights()
        
        if config.fix_lang_embedding:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
    
    def forward_txt(self, txt_ids, txt_masks):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama(
        self, rgb_fts, dep_fts, loc_fts, nav_types, view_lens
    ):
        device = rgb_fts.device

        rgb_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(rgb_fts)
        )
        if self.img_embeddings.dep_linear is not None:
            dep_embeds = self.img_embeddings.dep_layer_norm(
                self.img_embeddings.dep_linear(dep_fts)
            )
            img_embeds = rgb_embeds + dep_embeds
        else:
            img_embeds = rgb_embeds

        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_lens = view_lens
        pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks

    def forward_navigation(
        self, txt_embeds, txt_masks, 
        gmap_vpids, gmap_step_ids, 
        gmap_img_fts, gmap_pos_fts, 
        gmap_masks, gmap_visited_masks, gmap_pair_dists,
    ):
        # global branch
        gmap_embeds = gmap_img_fts + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2)
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))

        outs = {
            'gmap_embeds': gmap_embeds,
            'global_logits': global_logits,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
            )
