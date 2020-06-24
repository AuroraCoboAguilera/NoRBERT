from torch.utils.data import DataLoader
from lineTextDataset import LineTextDataset
from NoRBERT import NoRBERT
from DeepNoRBERT import DeepNoRBERT
from transformers import BertForMaskedLM
import torch
import os, sys
import random

import h5py


add_path = '../GMVAE'
sys.path.append(os.path.abspath(add_path))

from GMVAE import *

seed = 42
random.seed(seed)

eval1 = True
eval2 = True
var = -1
replace_mask_token = False
option=-1

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

dataset = 14#13#14
NoRBERT_flag = False
model_path = './models/bert_retrained_multi30k_8epochs'
model_name = 'bert_retrained_multi30k_8epochs'
replace_mask_token = False

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


# dataset = 13#14#13
# NoRBERT_flag = True
# model_path = './models/11DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '11DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_11deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = True#False#True
# option = 11

# dataset = 14#13#14
# NoRBERT_flag = True
# model_path = './models/9DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '9DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_9deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = True#False#True
# option = 9

# dataset = 13#14
# NoRBERT_flag = True
# model_path = './models/NoRBERT_base_dataset1_GMVAE_1_001_150_50_20_1500_6_03_1e-05'
# model_name = 'NoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_1e-05_var1e-4'
# nor_path = '../GMVAE/experiments/bert_base_dataset1/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 150
# replace_mask_token = False
# var = 1e-4

# dataset = 14#13#14
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
# replace_mask_token = True#False#True
# option = 11

# dataset = 14#14#13
# NoRBERT_flag = True
# model_path = './models/12DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '12DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_12deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = True#True#False#True
# option = 12

# dataset = 14#14#13
# NoRBERT_flag = True
# model_path = './models/12DeepNoRBERT_base_multi30k_GMVAE_1_01_150_50_20_1500_6_03_1e-05'
# model_name = '12DeepNoRBERT_base_multi30k_GMVAE_1_01_150_50_20_1500_6_03_1e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_12deep/checkpoints/GMVAE_1_01_150_50_20_1500_6_03_1e-05/config.pt'
# step_restore = 500
# replace_mask_token = False#True#False#True
# option = 12

# dataset = 13#14#13
# NoRBERT_flag = True
# model_path = './models/1DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '1DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_1deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = True#True#False#True
# option = 1

# dataset = 14#14#13
# NoRBERT_flag = True
# model_path = './models/2DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# model_name = '2DeepNoRBERT_base_multi30k_GMVAE_1_001_150_50_20_1500_6_03_5e-05'
# nor_path = '../GMVAE/experiments/bert_base_multi30k_2deep/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt'
# step_restore = 500
# replace_mask_token = True#True#False#True
# option = 2


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

# Load the pre-trained BERT model
if NoRBERT_flag:
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

if eval1:
    ###########################################################################################
    # Evaluate and save training samples (50 first samples)
    ###########################################################################################

    if replace_mask_token:
        file = open(results_dir + 'Training_replaced.txt', 'w')
        file2 = open(results_dir + 'Training_replaced_count.txt', 'w')
    else:
        file = open(results_dir + 'Training.txt', 'w')


    total_masked_tokens = 0
    changed_masked_tokens = 0
    for it, (tokens, masked_lm_labels, mask_token_index, sentence, sentence_true, _) in enumerate(train_loader):

        input_ids = tokens['input_ids'].squeeze(1)
        token_type_ids = tokens['token_type_ids'].squeeze(1)
        attention_mask = tokens['attention_mask'].squeeze(1)
        mask_token_index = mask_token_index.squeeze(0)
        masked_lm_labels = masked_lm_labels.squeeze(0)

        # Converting these to cuda tensors
        #seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

        # Evaluation mode
        model.eval()

        # Predict all tokens
        with torch.no_grad():
            token_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        token_logits = token_logits[0]

        # Predicted tokens of the complete sentence
        predicted_tokens = torch.argmax(token_logits, dim=2).squeeze()
        # predicted_sentence = ' '.join([train_set.decode(token) for token in predicted_tokens[1:attention_mask.squeeze().sum()]])

        predicted_sentence = train_set.tokenizer.decode(predicted_tokens[1:attention_mask.squeeze().sum()-1])

        # Token corresponding to [MASK] tokens
        mask_token_logits = token_logits[0, mask_token_index, :]

        # Top 5 tokens predicted to [MASK] tokens
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1)[1]

        if replace_mask_token:
            # Compute statistics about replaced tokens, how many are changed
            predicted_masked_tokens = predicted_tokens[mask_token_index]
            source_masked_tokens = input_ids.squeeze()[mask_token_index]
            total_masked_tokens += len(source_masked_tokens)
            changed_masked_tokens += sum(source_masked_tokens != predicted_masked_tokens)

        print('Source: ' + sentence[0])
        file.write('Source: ' + sentence[0] + '\n')
        print('Target: ' + sentence_true[0])
        file.write('Target: ' + sentence_true[0] + '\n')
        if NoRBERT_flag:
            print('NoRBERT: ' + predicted_sentence)
            file.write('NoRBERT: ' + predicted_sentence + '\n')
        else:
            print('BERT: ' + predicted_sentence)
            file.write('BERT: ' + predicted_sentence + '\n')

        for masked_token in top_5_tokens:
            decoded_mask_tokens = [train_set.decode(token) for token in masked_token]
            print(decoded_mask_tokens)
            file.write(str(decoded_mask_tokens) + '\n')


        if (it + 1) % 10 == 0:
            print("Iteration {} complete at training. ".format(it + 1))

        if it == 50:
            break

    file.close()
    if replace_mask_token:
        file2.write('Replaced disturbed tokens: {}/{}: {:.2f}%\n'.format(changed_masked_tokens, total_masked_tokens, float(changed_masked_tokens)/total_masked_tokens*100))
        file2.close()

    ###########################################################################################
    # Evaluate and save test samples (50 first samples)
    ###########################################################################################

    if replace_mask_token:
        file = open(results_dir + 'Testing_replaced.txt', 'w')
        file2 = open(results_dir + 'Testing_replaced_count.txt', 'w')
    else:
        file = open(results_dir + 'Testing.txt', 'w')


    total_masked_tokens = 0
    changed_masked_tokens = 0
    for it, (tokens, _, mask_token_index, sentence, sentence_true, _) in enumerate(val_loader):

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
        # predicted_sentence = ' '.join([train_set.decode(token) for token in predicted_tokens[1:attention_mask.squeeze().sum()]])
        predicted_sentence = train_set.tokenizer.decode(predicted_tokens[1:attention_mask.squeeze().sum()-1])

        # Token corresponding to [MASK] tokens or to replaced tokens
        mask_token_logits = token_logits[0, mask_token_index, :]

        # Top 5 tokens predicted to [MASK] tokens
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1)[1]

        if replace_mask_token:
            # Compute statistics about replaced tokens, how many are changed
            predicted_masked_tokens = predicted_tokens[mask_token_index]
            source_masked_tokens = input_ids.squeeze()[mask_token_index]
            total_masked_tokens += len(source_masked_tokens)
            changed_masked_tokens += sum(source_masked_tokens != predicted_masked_tokens)

        print('Source: ' + sentence[0])
        file.write('Source: ' + sentence[0] + '\n')
        print('Target: ' + sentence_true[0])
        file.write('Target: ' + sentence_true[0] + '\n')
        if NoRBERT_flag:
            print('NoRBERT: ' + predicted_sentence)
            file.write('NoRBERT: ' + predicted_sentence + '\n')
        else:
            print('BERT: ' + predicted_sentence)
            file.write('BERT: ' + predicted_sentence + '\n')

        for masked_token in top_5_tokens:
            decoded_mask_tokens = [train_set.decode(token) for token in masked_token]
            print(decoded_mask_tokens)
            file.write(str(decoded_mask_tokens) + '\n')


        if (it + 1) % 10 == 0:
            print("Iteration {} complete at testing. ".format(it + 1))

        if it == 50:
            break

    file.close()
    if replace_mask_token:
        file2.write('Replaced disturbed tokens: {}/{}: {:.2f}%\n'.format(changed_masked_tokens, total_masked_tokens, float(changed_masked_tokens)/total_masked_tokens*100))
        file2.close()

if eval2:

    ###########################################################################################
    # Compute statistic in the whole test dataset, not just the 50 first sentences
    ###########################################################################################

    # Evaluate and save test samples
    if replace_mask_token:
        fileTotal = open(results_dir + 'TestingTotal_replaced_count.txt', 'w')
    else:
        fileTotal = open(results_dir + 'TestingTotal_count.txt', 'w')

    val_loader2 = DataLoader(val_set, batch_size=50, num_workers=5, shuffle=False)

    count_different_true = 0
    count_total = 0
    count_different_true_mask = 0
    count_total_mask = 0
    total_masked_tokens = 0
    changed_masked_tokens = 0
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
        # predicted_sentence = ' '.join([train_set.decode(token) for token in predicted_tokens[1:attention_mask.squeeze().sum()]])
        # predicted_sentence = train_set.tokenizer.decode(predicted_tokens[1:attention_mask.squeeze().sum()-1])

        if replace_mask_token:
            # Compute statistics about replaced tokens, how many are changed
            predicted_masked_tokens = predicted_tokens[mask_token_index]
            source_masked_tokens = input_ids.squeeze()[mask_token_index]
            total_masked_tokens += len(source_masked_tokens)
            changed_masked_tokens += sum(source_masked_tokens != predicted_masked_tokens)


        # Discard sentences where the [MASK] word correspond to more than 1 tokens and the lengths of the sentences reconstructed are different from the true one
        sentencesToCount = torch.sum(tokens_true['attention_mask'].squeeze(1) == attention_mask, dim=1) == attention_mask.shape[1]

        attention_mask[range(input_ids.shape[0]), torch.sum(attention_mask, dim=1) - 1] = 0
        attention_mask[:, 0] = 0

        predicted_tokens_masked = predicted_tokens * attention_mask
        true_tokens_masked = tokens_true['input_ids'].squeeze(1) * attention_mask

        count_different_true += sum(torch.sum((predicted_tokens_masked != true_tokens_masked), dim=1)[sentencesToCount])
        count_total += sum(torch.sum(attention_mask, dim=1)[sentencesToCount])

        predicted_maskTokens_masked = predicted_tokens * mask_token_index
        true_maskTokens_masked = tokens_true['input_ids'].squeeze(1) * mask_token_index

        count_different_true_mask += sum(torch.sum((predicted_maskTokens_masked != true_maskTokens_masked), dim=1)[sentencesToCount])
        count_total_mask += sum(torch.sum(mask_token_index, dim=1)[sentencesToCount])

        print("Iteration {}/{} completed at testing. ".format(it + 1, len(val_loader2)))


    fileTotal.write('Different tokens from ground truth: {}/{}: {:.2f}%\n'.format(count_different_true, count_total, float(count_different_true)/float(count_total)*100))
    fileTotal.write('Different (just [MASK] or disrupted) tokens from ground truth: {}/{}: {:.2f}%\n'.format(count_different_true_mask, count_total_mask, float(count_different_true_mask)/float(count_total_mask)*100))
    if replace_mask_token:
        fileTotal.write('Replaced disturbed tokens: {}/{}: {:.2f}%\n'.format(changed_masked_tokens, total_masked_tokens, float(changed_masked_tokens)/float(total_masked_tokens)*100))
    fileTotal.close()