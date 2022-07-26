'''

Implementation of Top NoRxlmRoBERTa (Noisy Regulagarized xlm RoBERTa), which applies a GM-VAE between the Encoder and the output layer
from ROBERTA, and Deep NoRxlmRoBERTa, which applies a GM-VAE between the Encoder layers from ROBERTA.
It substitutes the files NoRxlmRoBERTa and DeepNoRxlmRoBERTa due to library updates.

Author: Aurora Cobo Aguilera
Date: 19th January 2022
Updated: 19th January 2022
'''


from transformers.utils import logging
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from models_NoRoBERTa import TopNoRoBERTa, DeepNoRoBERTa

logger = logging.get_logger(__name__)

# ACA
class TopNoRxlmRoBERTa(TopNoRoBERTa):
    """
        This class overrides [`TopNoRoBERTa`]. Please check the superclass for the appropriate
        documentation alongside usage examples.
        """
    config_class = XLMRobertaConfig

# ACA
class DeepNoRxlmRoBERTa(DeepNoRoBERTa):
    """
        This class overrides [`DeepNoRoBERTa`]. Please check the superclass for the appropriate
        documentation alongside usage examples.
        """
    config_class = XLMRobertaConfig