from torch.utils.data import DataLoader
from lineTextDataset import LineTextDataset
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os

import h5py

os.environ['KMP_DUPLICATE_LIB_OK']='True'



dataset = 1


#model_path = 'bert-base-uncased'
#model_name = 'BERT_base'

#model_path = './models/prueba1'
#model_name = 'BERT_prueba1'

#model_path = './models/bert_base_dataset14'
#model_name = 'bert_base_retrained_dataset14'

#dataset = 1
#model_path = './models/bert_base_retrained_multi30k'
#model_name = 'bert_base_retrained_multi30k'

# dataset = 'snli'
# model_path = 'bert-base-uncased'
# model_name = 'bert_base_snli_50000'

dataset = 'snli'
model_path = './models/bert_base_retrained_snli/checkpoint-92500'
model_name = 'bert_retrained_snli_50000'

maxlen = 100
maxsentences = 50000

# Creating instances of training and validation set
if dataset == 1:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train.txt', filename_true='./dataset/multi30k/multi30k_train.txt', maxlen=maxlen, model_name=model_path, eval_mode=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test.txt', filename_true='./dataset/multi30k/multi30k_test.txt', maxlen=maxlen, model_name=model_path, eval_mode=True)
    data_name = 'multi30k'
elif dataset == 13:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, eval_mode=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, eval_mode=True)
    data_name = 'multi30k_unk'
elif dataset == 14:
    train_set = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk06.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, eval_mode=True)
    val_set = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk06.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=model_path, change_unk_token=True, eval_mode=True)
    data_name = 'multi30k_unk06'
elif dataset == 'snli':
    train_set = LineTextDataset(filename='./dataset/data_snli/train.txt', filename_true='./dataset/data_snli/train.txt', maxlen=maxlen, model_name=model_path, eval_mode=True)
    val_set = LineTextDataset(filename='./dataset/data_snli/test.txt', filename_true='./dataset/data_snli/test.txt', maxlen=maxlen, model_name=model_path, eval_mode=True)
    data_name = 'snli'


# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=50, num_workers=5, shuffle=True)
val_loader = DataLoader(val_set, batch_size=50, num_workers=5, shuffle=False)


#configuration = BertConfig()


# Load the pre-trained BERT model
model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)

# Configure results directory
results_dir = './results/dataset{}/{}/'.format(dataset, model_name)
directory, filename = os.path.split(os.path.abspath(results_dir))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

maxit = np.floor(maxsentences / 50)
# Evaluate and save training samples
for it, (tokens, _, _, _, _) in enumerate(train_loader):

    input_ids = tokens['input_ids'].squeeze()
    token_type_ids = tokens['token_type_ids'].squeeze()
    attention_mask = tokens['attention_mask'].squeeze()

    # Converting these to cuda tensors
    #seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

    # Evaluation mode
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        token_logits, hidden_state, attentions = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #token_logits = token_logits[0]
    if it == 0:
        #save_last_hidden_state = np.copy(hidden_state[-1].cpu())
        save_last_hidden_state = np.copy(hidden_state[-1][attention_mask == 1].cpu())
    else:
        save_last_hidden_state = np.concatenate((save_last_hidden_state, hidden_state[-1][attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 10 == 0:
        print("Iteration {}/{} completed at training. ".format(it + 1, maxit))

    if it+1 == maxit:
        break

print('{} iterations completed'.format(it+1))

# creating a HDF5 file
decoded_file_name = results_dir + 'lastHiddenState_{}_train.hdf5'.format(maxsentences)
print('Saving file of shape: ' + str(save_last_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=save_last_hidden_state)



# Evaluate and save test samples

for it, (tokens, _, _, _, _) in enumerate(val_loader):

    input_ids = tokens['input_ids'].squeeze()
    token_type_ids = tokens['token_type_ids'].squeeze()
    attention_mask = tokens['attention_mask'].squeeze()

    # Converting these to cuda tensors
    # seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

    # Evaluation mode
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        token_logits, hidden_state, attentions = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #token_logits = token_logits[0]

    if it == 0:
        #save_last_hidden_state = np.copy(hidden_state[-1].cpu())
        save_last_hidden_state = np.copy(hidden_state[-1][attention_mask == 1].cpu())

    else:
        #save_last_hidden_state = np.concatenate((save_last_hidden_state, hidden_state[-1].cpu()), axis=0)
        save_last_hidden_state = np.concatenate((save_last_hidden_state, hidden_state[-1][attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 10 == 0:
        print("Iteration {}/{} completed at testing. ".format(it + 1, len(val_loader)))


# creating a HDF5 file
decoded_file_name = results_dir + 'lastHiddenState_{}_test.hdf5'.format(len(val_loader)*50)
print('Saving file of shape: ' + str(save_last_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=save_last_hidden_state)