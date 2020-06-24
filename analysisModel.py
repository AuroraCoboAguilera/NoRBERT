from torch.utils.data import DataLoader
from lineTextDataset import LineTextDataset
from NoRBERT import NoRBERT
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os, sys
import matplotlib.pyplot as plt


import h5py


add_path = '../GMVAE'
sys.path.append(os.path.abspath(add_path))

from GMVAE import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


dataset = 11
NoRBERT_flag = False
model_path = 'bert-base-uncased'
model_name = 'bert_base_snli'

dataset = 11
NoRBERT_flag = True
model_path = './models/NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-4700'
model_name = 'NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-4700'
nor_path = '../GMVAE/experiments/bert_base_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
step_restore = 200


maxlen=100

# Creating instances of training and validation set
if dataset == 1:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train.txt', filename_true='./dataset/multi30k/multi30k_train.txt', maxlen=maxlen, model_name=model_path)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test.txt', filename_true='./dataset/multi30k/multi30k_test.txt', maxlen=maxlen, model_name=model_path)
    data_name = 'multi30k'
elif dataset == 13:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    data_name = 'multi30k_unk'
elif dataset == 14:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk06.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk06.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    data_name = 'multi30k_unk06'
elif dataset == 11:
    train_set = LineTextDataset(filename='./dataset/data_snli/train_unk.txt', filename_true='./dataset/data_snli/train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    val_set = LineTextDataset(filename='./dataset/data_snli/test_unk.txt', filename_true='./dataset/data_snli/test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    data_name = 'snli_unk'
elif dataset == 15:
    train_set = LineTextDataset(filename='./dataset/data_snli/train_unk06.txt', filename_true='./dataset/data_snli/train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    val_set = LineTextDataset(filename='./dataset/data_snli/test_unk06.txt', filename_true='./dataset/data_snli/test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True)
    data_name = 'snli_unk06'



# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=1, num_workers=5, shuffle=False)
val_loader = DataLoader(val_set, batch_size=1, num_workers=5, shuffle=False)


# -------------------------------------
# Load the pre-trained GMVAE model
# -------------------------------------

if NoRBERT_flag:
    print('Loading GMVAE model from checkpoint in %s...' % (nor_path))
    checkpoint = torch.load(nor_path, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    config.step_restore = step_restore
    config.cuda = 0
    GMVAE_model = GMVAE(config, seq2seq=True)
    GMVAE_model.restore_model(add_path=add_path)

# Load the pre-trained BERT model
if NoRBERT_flag:
    model = NoRBERT.from_pretrained(model_path, GMVAE_model=GMVAE_model)
else:
    model = BertForMaskedLM.from_pretrained(model_path)

# Configure results directory
results_dir = './results/dataset{}/{}/'.format(dataset, model_name)
directory, filename = os.path.split(os.path.abspath(results_dir))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)




for it, (tokens, _, mask_token_index, sentence, sentence_true) in enumerate(val_loader):

    input_ids = tokens['input_ids'].squeeze(1)
    token_type_ids = tokens['token_type_ids'].squeeze(1)
    attention_mask = tokens['attention_mask'].squeeze(1)
    mask_token_index = mask_token_index.squeeze(0)

    # Converting these to cuda tensors
    # seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

    # Evaluation mode
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        token_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    token_logits = token_logits[0]

    # Predicted tokens of the complete sentence
    predicted_tokens = torch.argmax(token_logits, dim=2).squeeze()
    predicted_sentence = ' '.join([train_set.decode(token) for token in predicted_tokens])

    # Token corresponding to [MASK] tokens
    mask_token_logits = token_logits[0, mask_token_index, :]

    # if it == 0:
    #     plt.figure(figsize=(14, 8))
    #     plt.stem(np.arange(len(mask_token_logits[0, :])), np.sort(mask_token_logits[0, :])[::-1])
    #     plt.title(sentence, fontsize=12)
    #     plt.savefig(results_dir + 'maskTokenDistribution1.pdf')
    #     plt.show()
    # if it == 4:
    #     plt.figure(figsize=(14, 8))
    #     plt.stem(np.arange(len(mask_token_logits[0, :])), np.sort(mask_token_logits[0, :])[::-1])
    #     plt.title(sentence, fontsize=12)
    #     plt.savefig(results_dir + 'maskTokenDistribution2.pdf')
    #     plt.show()
    #     plt.figure(figsize=(14, 8))
    #     plt.stem(np.arange(len(mask_token_logits[1, :])), np.sort(mask_token_logits[0, :])[::-1])
    #     plt.title(sentence, fontsize=12)
    #     plt.savefig(results_dir + 'maskTokenDistribution3.pdf')
    #     plt.show()

    if it == 5:
        sorted_tokens_id = np.argsort(mask_token_logits[0, :]).numpy()[::-1]
        sorted_logits = np.sort(mask_token_logits[0, :])[::-1]
        sorted_tokens = [train_set.decode(token) for token in sorted_tokens_id]

        plt.figure(figsize=(14, 8))
        plt.stem(np.arange(50), sorted_logits[:50])
        plt.xticks(np.arange(50), sorted_tokens[:50], rotation='vertical')
        plt.title(sentence[0]+'\n'+predicted_sentence[1:predicted_sentence[1:].find('.')+2], fontsize=12)
        plt.savefig(results_dir + 'maskTokenDistribution{}b.eps'.format(it))
        plt.show()
    if it == 0 or it == 4 or it == 11:
        sorted_tokens_id = np.argsort(mask_token_logits[0, :]).numpy()[::-1]
        sorted_logits = np.sort(mask_token_logits[0, :])[::-1]
        sorted_tokens = [train_set.decode(token) for token in sorted_tokens_id]
        plt.figure(figsize=(14, 8))
        plt.stem(np.arange(50), sorted_logits[:50])
        plt.xticks(np.arange(50), sorted_tokens[:50], rotation='vertical')
        plt.title(sentence[0]+'\n'+predicted_sentence[1:predicted_sentence[1:].find('.')+2], fontsize=12)
        plt.savefig(results_dir + 'maskTokenDistribution{}b.eps'.format(it))
        plt.show()
        sorted_tokens_id = np.argsort(mask_token_logits[1, :]).numpy()[::-1]
        sorted_logits = np.sort(mask_token_logits[1, :])[::-1]
        sorted_tokens = [train_set.decode(token) for token in sorted_tokens_id]
        plt.figure(figsize=(14, 8))
        plt.stem(np.arange(50), sorted_logits[:50])
        plt.xticks(np.arange(50), sorted_tokens[:50], rotation='vertical')
        plt.title(sentence[0]+'\n'+predicted_sentence[1:predicted_sentence[1:].find('.')+2], fontsize=12)
        plt.savefig(results_dir + 'maskTokenDistribution{}b2.eps'.format(it))
        plt.show()

    if it == 11:
        break







