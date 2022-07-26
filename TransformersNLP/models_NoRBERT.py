'''

Implementation of Top NoRBERT (Noisy Regulagarized BERT), which applies a GM-VAE between the Encoder and the output layer
from BERT, and Deep NoRBERT, which applies a GM-VAE between the Encoder layers from BERT.
It substitutes the files NoRBERT and DeepNoRBERT due to library updates.

Author: Aurora Cobo Aguilera
Date: 14th May 2021
Updated: 21th May 2021
'''

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, BertEmbeddings, BertPooler, BertAttention, BertIntermediate, BertOutput, BertModel
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
import math
from transformers.utils import logging
from models_BERT_multiMLHead import BertOnlyMLMmultiHead

logger = logging.get_logger(__name__)

# ACA
class TopNoRBERT(BertPreTrainedModel):
    def __init__(self, config, freeze_bert_encoder=True, GMVAE_model=None, K_select=-1):
        '''
                Class that define the TopNoRBERT model as the concatenation of BERT (Transformer Encoder) with a classification
                layer to compute the [MASK] tokens and build the reconstruction of the sentence.
                It is based on the classes of the library transformers from Hugging face.
                Parameters:
                    - config: Is a class from the library transformers.
                    - freeze_bert_encoder: Flag to freeze parameters of BERT (Encoder part), not to train them. Just train
                    the classification layer.
                    - GMVAE_model: GMVAE model to be applied between BERT and the top classification layer.
        '''
        super().__init__(config)

        # Instantiating BERT model object
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.gmvae = GMVAE_model        # ACA

        self.K_select = K_select

        self.init_weights()

        # ACA
        # Freeze bert layers
        if freeze_bert_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

        print('NoRBERT initialized with freezed bert parameters: {}; and a GMVAE model: {}'.format(freeze_bert_encoder, (GMVAE_model != None)))


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch x length x hidden_size)

        # ACA: Apply GMVAE reconstruction
        if self.gmvae != None:
            sequence_output_GMVAE = sequence_output.clone()
            if self.gmvae.contextual_length == 0:
                if self.gmvae.cuda:
                    sequence_output_GMVAE[attention_mask == 1] = torch.Tensor(self.gmvae.batch_reconstruction((sequence_output[attention_mask == 1]).cuda(), K_select=self.K_select)).cuda()
                    # sequence_output_GMVAE[attention_mask == 1] = self.gmvae.batch_reconstruction((sequence_output[attention_mask == 1]).cuda(), K_select=self.K_select)

                else:
                    sequence_output_GMVAE[attention_mask == 1] = torch.Tensor(self.gmvae.batch_reconstruction(sequence_output[attention_mask == 1], K_select=self.K_select))
            else:   # Contextual NoRBERT
                # Prepare inputs concatenating context vectors
                attention_mask_extended = attention_mask.repeat(sequence_output_GMVAE.size(-1), 1, 1).permute(1, 2, 0)
                top_hidden_state_masked = attention_mask_extended.cpu() * sequence_output_GMVAE.cpu()
                # Concatenate norbert_args.contextual_tokens 0s before and after regarding the number of contextual tokens
                padding_contextual = torch.zeros(top_hidden_state_masked.size()[0], self.gmvae.contextual_length, top_hidden_state_masked.size()[2])
                top_hidden_state_padded = torch.cat((torch.cat((padding_contextual, top_hidden_state_masked), dim=1), padding_contextual), dim=1)
                # Make each sentence a batch
                window_size = self.gmvae.contextual_length * 2 + 1
                stride = 1
                # REconstruct per batch, considering a batch, all the tokens from a sentence
                for batch_index, batch in enumerate(top_hidden_state_padded):

                    contextual_top_hidden_state = [batch[i:i + window_size, :] for i in range(0, top_hidden_state_padded.size(1) - window_size + 1, stride) if (batch[i + self.gmvae.contextual_length:i + window_size, :]).sum() != 0]
                    contextual_top_hidden_state_stack = torch.stack(contextual_top_hidden_state)
                    contextual_top_hidden_state_flatten = torch.flatten(contextual_top_hidden_state_stack, 1)

                    contextual_top_hidden_state_reconstructed = self.gmvae.batch_reconstruction(contextual_top_hidden_state_flatten)

                    sequence_output_GMVAE[batch_index, :contextual_top_hidden_state_reconstructed.shape[0], :] = torch.Tensor(contextual_top_hidden_state_reconstructed).cuda()

            prediction_scores = self.cls(sequence_output_GMVAE)
        else:
            prediction_scores = self.cls(sequence_output)


        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

        return MaskedLMOutput2(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            deep_hidden_state=sequence_output,  # ACA

        )

# ACA
class DeepNoRBERT(BertPreTrainedModel):

    def __init__(self, config, freeze_bert_lowerEncoder=True, GMVAE_model=None, save_deep_hidden_state=False, layer_deep=None, double_flag=False, K_select=-1, multiHead=False):
        '''
                Class that define the DeepNoRBERT model as the concatenation of BERT (Transformer Encoder) with a classification
                layer to compute the [MASK] tokens and build the reconstruction of the sentence.
                It is based on the classes of the library transformers from Hugging face.
                Parameters:
                    - config: Is a class from the library transformers.
                    - freeze_bert_lowerEncoder: Flag to freeze the parameters of BERT lower than the position where the GMVAE is applied,
                    not to train them. Just train the classification layer and the top encoder layers.
                    - GMVAE_model: GMVAE model to be applied between the encoder layers of BERT.
                    - save_deep_hidden_state: Flag to save the (low) deep hidden states. Not used for training.
                    - low_deep: Configure GMVAE in the layer 9 of the encoder. If False, do it deep, in the middle layer,
                     layer 6 (if there are 12).
        '''
        super().__init__(config)

        # Instantiating BERT model object
        self.bert = BertModel_(config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag, add_pooling_layer=False, K_select=K_select)        # ACA

        if multiHead:
            self.cls = BertOnlyMLMmultiHead(config)
        else:
            self.cls = BertOnlyMLMHead(config)

        self.init_weights()

        # ACA
        # Freeze bert layers below the place where the GMVAE is applied
        if freeze_bert_lowerEncoder:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            if double_flag == False:
                if layer_deep != None:
                    for nlayer in range(layer_deep-1):
                        for p in self.bert.encoder.layer[nlayer].parameters():
                            p.requires_grad = False

                    for p in self.bert.encoder.layer[layer_deep-1].attention.parameters():
                        p.requires_grad = False
                else:
                    for nlayer in range(math.ceil(config.num_hidden_layers / 2)-1):
                        for p in self.bert.encoder.layer[nlayer].parameters():
                            p.requires_grad = False

                    for p in self.bert.encoder.layer[math.ceil(config.num_hidden_layers / 2)-1].attention.parameters():
                        p.requires_grad = False
            else:
                if len(layer_deep) != 2:
                    raise ValueError("You should select 2 layers to apply a DGM in Double Deep NoRBERT")
                else:
                    freezer = max(layer_deep)
                    for nlayer in range(freezer - 1):
                        for p in self.bert.encoder.layer[nlayer].parameters():
                            p.requires_grad = False

                    for p in self.bert.encoder.layer[freezer - 1].attention.parameters():
                        p.requires_grad = False


        if layer_deep != None:
            print('Customized {} Layer Deep NoRBERT initialized with freezed lower bert parameters: {}; and a GMVAE model: {}'.format(layer_deep, freeze_bert_lowerEncoder, (GMVAE_model != None)))
        else:
            print('1/2 Deep NoRBERT initialized with freezed lower bert parameters: {}; and a GMVAE model: {}'.format(freeze_bert_lowerEncoder, (GMVAE_model != None)))


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output           # (masked_lm_loss), prediction_scores, (hidden_states), (attentions), ((low) deep hidden state)

        return MaskedLMOutput2(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            deep_hidden_state=outputs.deep_hidden_state,  # ACA

        )



class BertModel_(BertPreTrainedModel):
    """
        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in `Attention is
        all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
        To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
        set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
        argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
        input to the forward pass.
        """

    def __init__(self, config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag, add_pooling_layer=True, K_select=-1):       # ACA
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag, K_select=K_select)            # ACA
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]   # add hidden_states and attentions if they are here, and deep hidden state

        return BaseModelOutputWithPoolingAndCrossAttentions2(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            deep_hidden_state=encoder_outputs.deep_hidden_state,        # ACA

        )


class BertEncoder(nn.Module):
    def __init__(self, config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag, K_select=-1):       # ACA
        super().__init__()
        self.config = config
        self.layer_deep = layer_deep

        # ACA
        if double_flag:
            layers = []
            for nlayer in range(config.num_hidden_layers):
                if nlayer + 1 == layer_deep[0]:
                    layers.append(BertLayer(config, GMVAE_model[0], save_deep_hidden_state, K_select=K_select))
                elif nlayer + 1 == layer_deep[1]:
                    layers.append(BertLayer(config, GMVAE_model[1], save_deep_hidden_state, K_select=K_select))
                else:
                    layers.append(BertLayer(config, None))
            self.layer = nn.ModuleList(layers)
        else:
            if layer_deep == None:
                self.layer = nn.ModuleList([BertLayer(config, GMVAE_model, save_deep_hidden_state, K_select=K_select) if nlayer + 1 == math.ceil(config.num_hidden_layers / 2) else BertLayer(config, None) for nlayer in range(config.num_hidden_layers)])
            else:
                self.layer = nn.ModuleList([BertLayer(config, GMVAE_model, save_deep_hidden_state, K_select=K_select) if nlayer+1 == layer_deep else BertLayer(config, None) for nlayer in range(config.num_hidden_layers)])

        self.save_deep_hidden_state = save_deep_hidden_state

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        deep_hidden_state = None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask,)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions,)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            if self.save_deep_hidden_state and i+1 == self.layer_deep:             # ACA
                deep_hidden_state = layer_outputs[-1]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-2],)
            else:
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)



        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                    deep_hidden_state,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions2(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            deep_hidden_state=deep_hidden_state,
        )



class BertLayer(nn.Module):
    def __init__(self, config, GMVAE_model, save_deep_hidden_state=False, K_select=-1):          # ACA
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.gmvae = GMVAE_model                                                # ACA
        self.save_deep_hidden_state = save_deep_hidden_state                    # ACA

        self.K_select = K_select

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value,)
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(self, "crossattention"), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        #ACA
        # Apply GMVAE reconstruction
        if self.gmvae != None:
            attention_output_GMVAE = attention_output.clone()
            if self.gmvae.cuda:
                att_out = (attention_output[(attention_mask == 0)[:, 0, 0, :]]).cuda()
                gmvae_att = self.gmvae.batch_reconstruction(att_out, K_select=self.K_select)
                attention_output_GMVAE[(attention_mask == 0)[:, 0, 0, :]] = torch.Tensor(gmvae_att).cuda()
                # attention_output_GMVAE[(attention_mask == 0)[:, 0, 0, :]] = torch.Tensor(gmvae_att.cpu()).cuda()
            else:
                attention_output_GMVAE[(attention_mask == 0)[:, 0, 0, :]] = torch.Tensor(
                    self.gmvae.batch_reconstruction(attention_output[(attention_mask == 0)[:, 0, 0, :]], K_select=self.K_select))
            if self.gmvae.cuda:
                attention_output_GMVAE = attention_output_GMVAE.cuda()

            layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output_GMVAE)   # It changes from previous version than was not normalized with attention_output_GMVAE #TODO
        else:
            layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output) # Original


        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        #ACA: Save deep hidden state
        if self.save_deep_hidden_state:
            outputs = outputs + (attention_output,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BaseModelOutputWithPastAndCrossAttentions2(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    deep_hidden_state: Optional[Tuple[torch.FloatTensor]] = None        # ACA


class BaseModelOutputWithPoolingAndCrossAttentions2(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    deep_hidden_state: Optional[Tuple[torch.FloatTensor]] = None        # ACA

class MaskedLMOutput2(ModelOutput):
    """
    Base class for masked language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    deep_hidden_state: Optional[Tuple[torch.FloatTensor]] = None        # ACA
