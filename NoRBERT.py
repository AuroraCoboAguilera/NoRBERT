'''

Implementation of NoRBERT (Noisy Regulagarized BERT), which applies a GM-VAE between the Encoder and the output layer
from BERT.

Author: Aurora Cobo Aguilera
Date: 7 April 2020
'''

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead

class NoRBERT(BertPreTrainedModel):

    def __init__(self, config, freeze_bert_encoder=True, GMVAE_model=None):
        '''
                Class that define the NoRBERT model as the concatenation of BERT (Transformer Encoder) with a classification
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

        self.gmvae = GMVAE_model

        self.init_weights()

        # Freeze bert layers
        if freeze_bert_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

        print('NoRBERT initialized with freezed bert parameters: {}; and a GMVAE model: {}'.format(freeze_bert_encoder, (GMVAE_model != None)))


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

        # Apply GMVAE reconstruction
        if self.gmvae != None:
            sequence_output_GMVAE = sequence_output.clone()
            sequence_output_GMVAE[attention_mask == 1] = torch.Tensor(self.gmvae.batch_reconstruction(sequence_output[attention_mask == 1]))
            if self.gmvae.cuda:
                sequence_output_GMVAE = sequence_output_GMVAE.cuda()

            prediction_scores = self.cls(sequence_output_GMVAE)

        else:
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

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
