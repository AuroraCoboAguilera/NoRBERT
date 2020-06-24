'''

Implementation of Deep NoRBERT (Noisy Regulagarized BERT), which applies a GM-VAE between the Encoder layers
from BERT.

Author: Aurora Cobo Aguilera
Date: 30 April 2020
'''

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, BertEmbeddings, BertPooler, BertAttention, BertIntermediate, BertOutput
import math

class DeepNoRBERT(BertPreTrainedModel):

    def __init__(self, config, freeze_bert_lowerEncoder=True, GMVAE_model=None, save_deep_hidden_state=False, layer_deep=None, double_flag=False):
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
        self.bert = BertModel(config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

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


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions), ((low) deep hidden state)

class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag)
        self.pooler = BertPooler(config)

        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
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
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:
        ]  # add hidden_states and attentions if they are here, and deep hidden state
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), ((low) deep_hidden_state)


class BertEncoder(nn.Module):
    def __init__(self, config, GMVAE_model, save_deep_hidden_state, layer_deep, double_flag):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        if double_flag:
            layers = []
            for nlayer in range(config.num_hidden_layers):
                if nlayer + 1 == layer_deep[0]:
                    layers.append(BertLayer(config, GMVAE_model[0], save_deep_hidden_state))
                elif nlayer + 1 == layer_deep[1]:
                    layers.append(BertLayer(config, GMVAE_model[1], save_deep_hidden_state))
                else:
                    layers.append(BertLayer(config, None))
            self.layer = nn.ModuleList(layers)
        else:
            if layer_deep == None:
                self.layer = nn.ModuleList([BertLayer(config, GMVAE_model, save_deep_hidden_state) if nlayer + 1 == math.ceil(config.num_hidden_layers / 2) else BertLayer(config, None) for nlayer in range(config.num_hidden_layers)])
            else:
                self.layer = nn.ModuleList([BertLayer(config, GMVAE_model, save_deep_hidden_state) if nlayer+1 == layer_deep else BertLayer(config, None) for nlayer in range(config.num_hidden_layers)])


        self.save_deep_hidden_state = save_deep_hidden_state
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

                if self.save_deep_hidden_state and len(layer_outputs) == 3:
                    deep_hidden_state = layer_outputs[2]
            else:
                if self.save_deep_hidden_state and len(layer_outputs) == 2:
                    deep_hidden_state = layer_outputs[1]


        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if self.save_deep_hidden_state:
            outputs = outputs + (deep_hidden_state,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertLayer(nn.Module):
    def __init__(self, config, GMVAE_model, save_deep_hidden_state=False):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.gmvae = GMVAE_model

        self.save_deep_hidden_state = save_deep_hidden_state

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # Apply GMVAE reconstruction
        if self.gmvae != None:
            attention_output_GMVAE = attention_output.clone()
            attention_output_GMVAE[(attention_mask == 0)[:, 0, 0, :]] = torch.Tensor(
                self.gmvae.batch_reconstruction(attention_output[(attention_mask == 0)[:, 0, 0, :]]))
            if self.gmvae.cuda:
                attention_output_GMVAE = attention_output_GMVAE.cuda()

            intermediate_output = self.intermediate(attention_output_GMVAE)
        else:
            intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        # Save deep hidden state
        if self.save_deep_hidden_state:
            outputs = outputs + (attention_output,)
        return outputs











