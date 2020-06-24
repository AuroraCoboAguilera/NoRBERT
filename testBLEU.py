from torch.utils.data import DataLoader
from lineTextDataset import LineTextDataset
from NoRBERT import NoRBERT
from DeepNoRBERT import DeepNoRBERT
from transformers import BertForMaskedLM
import torch
import os, sys
import random
import math
from collections import Counter
import numpy as np
import warnings

from torchnlp.metrics import get_moses_multi_bleu
from nltk.translate.bleu_score import sentence_bleu

add_path = '../GMVAE'
sys.path.append(os.path.abspath(add_path))

from GMVAE import *

seed = 42
random.seed(seed)

var = -1
replace_mask_token = False
option=-1
double_flag = False

# dataset = 1
# NoRBERT_flag = False
# model_path = './models/bert_base_retrained_multi30k'
# model_name = 'bert_base_retrained_multi30k'

# dataset = 13#14
# NoRBERT_flag = False
# model_path = 'bert-base-uncased'
# model_name = 'bert_base'

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_base_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_retrained_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_retrained_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150

# dataset = 14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_retrained_dataset14_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = 'NoRBERT_retrained_dataset14_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_retrained_dataset14/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 110

# dataset = 13#14
# NoRBERT_flag = False
# model_path = './models/bert_retrained_multi30k'
# model_name = 'bert_retrained_multi30k'

# dataset = 11#15
# NoRBERT_flag = False
# model_path = 'bert-base-uncased'
# model_name = 'bert_base_snli'

# dataset = 11#15
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-1150'
# model_name = 'NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-1150'
# nor_path = '../GMVAE/experiments/bert_base_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 200

# dataset = 11#15
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-4700'
# model_name = 'NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-4700'
# nor_path = '../GMVAE/experiments/bert_base_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 200

# dataset = 11#15
# NoRBERT_flag = True
# model_path = './models/NoRBERT_retrained_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-4800'
# model_name = 'NoRBERT_retrained_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-4800'
# nor_path = '../GMVAE/experiments/bert_retrained_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 300

# dataset = 11#15
# NoRBERT_flag = True
# model_path = './models/NoRBERT_retrained_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-2500'
# model_name = 'NoRBERT_retrained_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-2500'
# nor_path = '../GMVAE/experiments/bert_retrained_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 300

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/LowDeepNoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05_1'
# model_name = 'LowDeepNoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05_1'
# nor_path = '../GMVAE/experiments/bert_retrained_dataset1_lowDeep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 950

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/LowDeepNoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'LowDeepNoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_retrained_dataset1_lowDeep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 950

# dataset = 13#14
# NoRBERT_flag = False
# model_path = 'bert-base-uncased'
# model_name = 'bert_base'
# replace_mask_token = True

# dataset = 13#13#14
# NoRBERT_flag = False
# model_path = './models/bert_retrained_multi30k'
# model_name = 'bert_retrained_multi30k'
# replace_mask_token = False

# dataset = 14#12#13#14
# NoRBERT_flag = False
# model_path = './models/bert_retrained_multi30k_8epochs'
# model_name = 'bert_retrained_multi30k_8epochs'
# replace_mask_token = False

# dataset = 14#13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_base_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150
# replace_mask_token = True#False#True

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_retrained_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_retrained_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_retrained_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150
# replace_mask_token = True

# dataset = 11#15
# NoRBERT_flag = False
# model_path = 'bert-base-uncased'
# model_name = 'bert_base'
# replace_mask_token = True

# dataset = 11#15#11
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05/checkpoint-4700'
# model_name = 'NoRBERT_base_datasetsnli_GMVAE_1_001_150_50_20_1500_6_03_1e-05_checkpoint-4700'
# nor_path = '../GMVAE/experiments/bert_base_snli_50000/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 200
# replace_mask_token = False#True

# dataset = 15#11#15
# NoRBERT_flag = False
# model_path = './models/bert_retrained_snli'
# model_name = 'bert_retrained_snli'
# replace_mask_token = False#True


# dataset = 14#12#14#13
# NoRBERT_flag = True
# model_path = './models/11DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '11DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_11deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#False#True
# option = 11

# dataset = 14#14#13#14
# NoRBERT_flag = True
# model_path = './models/9DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '9DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_9deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#False#True
# option = 9

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05_var1e-4'
# nor_path = '../GMVAE/experiments/bert_base_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150
# replace_mask_token = False
# var = 1e-4

# dataset = 14#12#13#14
# NoRBERT_flag = True
# model_path = './models/3DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '3DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_3deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#False#True
# option = 3

# dataset = 13#14#13
# NoRBERT_flag = True
# model_path = './models/11DeepNoRBERT_retrained_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '11DeepNoRBERT_retrained_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_retrained_multi30k_11deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#False#True
# option = 11

# dataset = 14#12#14#13
# NoRBERT_flag = True
# model_path = './models/12DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '12DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_12deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#False#True
# option = 12

# dataset = 13#14#13
# NoRBERT_flag = True
# model_path = './models/12DeepNoRBERT_base_multi30k_GMVAE_1_01_150_50_20_1500_6_03_1e-05'
# model_name = '12DeepNoRBERT_base_multi30k_GMVAE_1_01_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_12deep/checkpoints/GMVAE_1_01_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#True#False#True
# option = 12

# dataset = 14#14#14#13
# NoRBERT_flag = True
# model_path = './models/1DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '1DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_1deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#True#False#True
# option = 1

# dataset = 14#12#14#13
# NoRBERT_flag = True
# model_path = './models/2DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '2DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_2deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#True#False#True
# option = 2

# dataset = 14#12#14#13
# NoRBERT_flag = True
# model_path = './models/1_9DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '1_9DeepNoRBERT_base_multi30k_GMVAE_1_01_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_1deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# nor_path2 = '../GMVAE/experiments/bert_base_multi30k_9deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# step_restore2 = 500
# replace_mask_token = True#True#False#True
# option = [1, 9]
# double_flag = True

dataset = 12#12#14#13
NoRBERT_flag = True
model_path = './models/1_3DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
model_name = '1_3DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
nor_path = '../GMVAE/experiments/bert_base_multi30k_1deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
nor_path2 = '../GMVAE/experiments/bert_base_multi30k_3deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
step_restore = 500
step_restore2 = 500
replace_mask_token = False#True#False#True
option = [1, 3]
double_flag = True



maxlen=100

# Creating instances of training and validation set
if dataset == 1:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train.txt', filename_true='./dataset/multi30k/multi30k_train.txt', maxlen=maxlen, model_name=model_path, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test.txt', filename_true='./dataset/multi30k/multi30k_test.txt', maxlen=maxlen, model_name=model_path, return_tokens_true=True)
    data_name = 'multi30k'
elif dataset == 13:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    data_name = 'multi30k_unk'
elif dataset == 14:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk06.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk06.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    data_name = 'multi30k_unk06'
elif dataset == 11:
    train_set = LineTextDataset(filename='./dataset/data_snli/train_unk.txt', filename_true='./dataset/data_snli/train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/data_snli/test_unk.txt', filename_true='./dataset/data_snli/test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    data_name = 'snli_unk'
elif dataset == 15:
    train_set = LineTextDataset(filename='./dataset/data_snli/train_unk06.txt', filename_true='./dataset/data_snli/train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/data_snli/test_unk06.txt', filename_true='./dataset/data_snli/test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    data_name = 'snli_unk06'
elif dataset == 12:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk04.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk04.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, replace_mask_token=replace_mask_token, seed=seed, return_tokens_true=True)
    data_name = 'multi30k_unk04'


# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=1, num_workers=5, shuffle=False)
val_loader = DataLoader(val_set, batch_size=1, num_workers=5, shuffle=False)

# -------------------------------------
# BLEU score code from scratch
# https://stackoverflow.com/questions/56968434/bleu-score-in-python-from-scratch
# -------------------------------------
def n_gram_generator(sentence, masked_sentence=None, n= 2,n_gram= False):
    '''
    N-Gram generator with parameters sentence
    n is for number of n_grams
    The n_gram parameter removes repeating n_grams
    '''
    sentence = sentence.lower() # converting to lower case
    sent_arr = np.array(sentence.split()) # split to string arrays
    if masked_sentence != None:
        sent_masked_arr = np.array(masked_sentence.lower().split())  # split to string arrays
    length = len(sent_arr)

    word_list = []
    for i in range(length+1):
        if i < n:
            continue
        word_range = list(range(i-n,i))
        s_list = sent_arr[word_range]
        if masked_sentence != None:
            s_masked_list = sent_masked_arr[word_range]
        if masked_sentence != None:
            if '[mask]' in s_masked_list:
                string = ' '.join(s_list) # converting list to strings
                word_list.append(string) # append to word_list
                if n_gram:
                    word_list = list(set(word_list))
        else:
            string = ' '.join(s_list) # converting list to strings
            word_list.append(string) # append to word_list
            if n_gram:
                word_list = list(set(word_list))
    return word_list

def bleu_score_masked(original, machine_translated, original_masked=None, epsilon=0.001):
    '''
    Bleu score function given a orginal and a machine translated sentences
    '''
    mt_length = len(machine_translated.split())
    o_length = len(original.split())

    # Brevity Penalty
    if mt_length>o_length:
        BP=1
    else:
        penality=1-(mt_length/o_length)
        BP=np.exp(penality)

    # Clipped precision
    clipped_precision_score = []
    for i in range(1, 5):
        original_n_gram = Counter(n_gram_generator(original, original_masked, n=i))
        machine_n_gram = Counter(n_gram_generator(machine_translated, n=i))

        c = sum(machine_n_gram.values())
        for j in machine_n_gram:
            if j in original_n_gram:
                if machine_n_gram[j] > original_n_gram[j]:
                    machine_n_gram[j] = original_n_gram[j]
            else:
                machine_n_gram[j] = 0

        #print (sum(machine_n_gram.values()), c)
        clipped_precision_score.append((sum(machine_n_gram.values()) + epsilon)/(c + epsilon))

    #print (clipped_precision_score)

    weights =[0.25]*4

    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    s = BP * math.exp(math.fsum(s))
    return s

# original = "It is a guide to action which ensures that the military alwasy obeys the command of the party"
# original_masked = "It is a [MASK] to action which [MASK] that the military [MASK] obeys the command of the party"
# machine_translated = "It is the guiding principle which guarantees the military forces alwasy being under the command of the party"

# print(bleu_score_masked(original, machine_translated, original_masked=original_masked))
# print(sentence_bleu([original.split()], machine_translated.split()))

# -------------------------------------
# Load the pre-trained GMVAE model
# -------------------------------------

if NoRBERT_flag:
    print('Loading GMVAE model from checkpoint in %s...' % (nor_path))
    checkpoint = torch.load(nor_path, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    config.step_restore = step_restore
    config.cuda = 0
    if var != -1:
        config.sigma = var
    GMVAE_model = GMVAE(config, seq2seq=True)
    GMVAE_model.restore_model(add_path=add_path)

    if double_flag:
        print('Loading GMVAE model from checkpoint in %s...' % (nor_path2))
        checkpoint = torch.load(nor_path2, map_location=lambda storage, loc: storage)
        config = checkpoint['config']
        config.step_restore = step_restore2
        config.cuda = 0
        GMVAE_model2 = GMVAE(config, seq2seq=True)
        GMVAE_model2.restore_model(add_path=add_path)

# Load the pre-trained BERT model
if NoRBERT_flag:
    if double_flag:
        model = DeepNoRBERT.from_pretrained(model_path, GMVAE_model=[GMVAE_model, GMVAE_model2], layer_deep=option, double_flag=True)
    else:
        if option == -1:
            model = NoRBERT.from_pretrained(model_path, GMVAE_model=GMVAE_model)
        else:
            model = DeepNoRBERT.from_pretrained(model_path, GMVAE_model=GMVAE_model, layer_deep=option)
else:
    model = BertForMaskedLM.from_pretrained(model_path)

# Configure results directory
results_dir = './results/dataset{}/{}/'.format(dataset, model_name)
directory, filename = os.path.split(os.path.abspath(results_dir))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


###########################################################################################
# Compute statistic in the whole test dataset, not just the 50 first sentences
###########################################################################################

# Evaluate and save test samples
if replace_mask_token:
    # fileTotal = open(results_dir + 'TestingTotal_replaced_bleu.txt', 'w')
    # fileTotal = open(results_dir + 'TestingTotal_replaced_bleu2.txt', 'w')
    fileTotal = open(results_dir + 'TestingTotal_replaced_bleuMasked.txt', 'w')
else:
    # fileTotal = open(results_dir + 'TestingTotal_bleu.txt', 'w')
    # fileTotal = open(results_dir + 'TestingTotal_bleu2.txt', 'w')
    fileTotal = open(results_dir + 'TestingTotal_bleuMasked.txt', 'w')

val_loader2 = DataLoader(val_set, batch_size=1, num_workers=5, shuffle=False)

count_different_true = 0
count_total = 0
count_different_true_mask = 0
count_total_mask = 0
total_masked_tokens = 0
changed_masked_tokens = 0
total_bleu_score = 0
for it, (tokens, _, mask_token_index, sentence, sentence_true, tokens_true) in enumerate(val_loader2):

    input_ids = tokens['input_ids'].squeeze(1)
    token_type_ids = tokens['token_type_ids'].squeeze(1)
    attention_mask = tokens['attention_mask'].squeeze(1)
    mask_token_index = mask_token_index.squeeze(0)
    input_ids_true = tokens_true['input_ids'].squeeze(1)

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
    predicted_sentence = train_set.tokenizer.decode(predicted_tokens[1:attention_mask.squeeze().sum()-1])
    # With nltk library
    # bleu_score = sentence_bleu([sentence_true[0].lower().replace('.', '').split()], predicted_sentence.replace('.', '').split())
    # Compute BLEU score with the official BLEU perl script
    # bleu_score = get_moses_multi_bleu([predicted_sentence.replace('.', '')], [sentence_true[0].replace('.', '')], lowercase=True)  # RETURNS: 47.9
    # My masked smoothed version
    bleu_score = bleu_score_masked(sentence_true[0].replace('.', ''), predicted_sentence.replace('.', ''), original_masked=sentence[0].replace('.', ''))
    # print(bleu_score*100)
    # print(sentence_true[0].replace('.', ''))
    # print(predicted_sentence.replace('.', ''))
    # print(sentence[0].replace('.', ''))
    if bleu_score is not None:
        total_bleu_score += bleu_score*100
    else:
        warnings.warn("BLEU score of one sentence ignored")

    if (it+1) % 50 == 0:
        print("Iteration {}/{} completed at testing. ".format(it + 1, len(val_loader2)))



fileTotal.write('BLEU Score in test dataset: {:.2f}\n'.format(total_bleu_score/(it+1)))
fileTotal.close()