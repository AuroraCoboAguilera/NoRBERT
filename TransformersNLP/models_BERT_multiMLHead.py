from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLMHeadModel, BertPredictionHeadTransform
import torch
from torch import nn

class BertLMPredictionmultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.decoder2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder1(hidden_states)
        hidden_states = self.decoder2(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMmultiHead(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = BertLMPredictionmultiHead(config)



class BertLMmultiHeadModel(BertLMHeadModel):

    def __init__(self, config):
        super().__init__(config)

        self.cls = BertOnlyMLMmultiHead(config)

        # Initialize weights and apply final processing
        self.post_init()
