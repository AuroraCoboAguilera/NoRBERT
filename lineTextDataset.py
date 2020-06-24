import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import random

class LineTextDataset(Dataset):

    def __init__(self, filename, filename_true, maxlen=100, model_name='bert-base-uncased', change_unk_token=False, replace_mask_token=False, eval_mode=False, mlm_generator=False, mlm_probability=0.15, seed=42, return_tokens_true=False):
        '''
        Class to define a dataset that read some text file.
        Read the file, use a Tokenizer loaded from model_name.
        Parameters:
            - change_unk_token: When the dataset file have <UNK> and we need to convert them to BERT format [MASK].
            - eval_mode: When we just want the tokens obtained by the encoder of he Tokenizer, but not the masked labels
            whose computation slow the process and we can avoid it.
            - mlm_generator: Activate it when we want to generate new [MASK] tokens as BERT do in the mlm mode.
        '''
        self.eval_mode = eval_mode
        self.mlm_generator = mlm_generator
        self.mlm_probability = mlm_probability
        self.return_tokens_true = return_tokens_true

        #Store the contents of the file in a pandas dataframe
        self.change_unk_token = change_unk_token
        self.replace_mask_token = replace_mask_token
        self.df = pd.read_csv(filename, delimiter='\n', header=None)
        self.df_true = pd.read_csv(filename_true, delimiter='\n', header=None)

        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)#, mask_token='<UNK>')

        self.maxlen = maxlen

        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        random.seed(self.seed+index)

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 0]

        # Replace <UNK> token by [MASK] or whatever mask token is being used by BERT model tokenizer.
        if self.change_unk_token:
            sentence = sentence.replace('<UNK>', self.tokenizer.mask_token)

        # Replace [MASK] tokens with a random token from the vocabulary
        # if self.replace_mask_token:
        #     sentence_replaced = ' '.join([])

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True, return_tensors='pt')

        mask_token_index = (tokens['input_ids'] == self.tokenizer.mask_token_id)[0]

        # Replace [MASK] tokens with a random token from the vocabulary
        if self.replace_mask_token:
            tokens_replaced = [token if token != self.tokenizer.mask_token_id else random.randint(0, self.tokenizer.vocab_size-1) for token in tokens['input_ids'].squeeze()]
            tokens_replaced = torch.tensor(tokens_replaced).unsqueeze(0)
            sep_index = np.where(tokens_replaced.squeeze().numpy() == self.tokenizer.sep_token_id)[0][0]
            sentence_replaced = self.tokenizer.decode(tokens_replaced.squeeze()[1:sep_index])


            tokens['input_ids'] = tokens_replaced
            sentence = sentence_replaced

        # Generate masked_lm_labels for training
        if not self.eval_mode:
            sentence_true = self.df_true.loc[index, 0]
            if self.mlm_generator:

                masked_lm_labels = tokens['input_ids'].clone()

                # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
                probability_matrix = torch.full(masked_lm_labels.shape, self.mlm_probability)
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in masked_lm_labels.tolist()
                ]
                probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
                if self.tokenizer._pad_token is not None:
                    padding_mask = masked_lm_labels.eq(self.tokenizer.pad_token_id)
                    probability_matrix.masked_fill_(padding_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                masked_lm_labels[~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(torch.full(masked_lm_labels.shape, 0.8)).bool() & masked_indices
                tokens['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(
                    torch.full(masked_lm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), masked_lm_labels.shape, dtype=torch.long)
                tokens['input_ids'][indices_random] = random_words[indices_random]

                mask_token_index = (tokens['input_ids'] == self.tokenizer.mask_token_id)[0]

            else:
                tokens_true = self.tokenizer.encode_plus(sentence_true, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True, return_tensors='pt')


                masked_lm_labels = tokens_true['input_ids'][0].clone()
                masked_lm_labels[~mask_token_index] = torch.tensor(-100)  # We only compute loss on masked tokens

        if not self.eval_mode:
            if self.return_tokens_true:
                return tokens, masked_lm_labels, mask_token_index, sentence, sentence_true, tokens_true
            else:
                return tokens, masked_lm_labels, mask_token_index, sentence, sentence_true
        else:
            return tokens, 0, 0, sentence, 0

    def decode(self, token):
        return self.tokenizer.decode([token])

    def generate_missing(self, src_sent):
        for ind in range(len(src_sent)):
            if torch.rand() < self.probMiss:
                src_sent[ind] = self.tokenizer.mask_token
        return src_sent