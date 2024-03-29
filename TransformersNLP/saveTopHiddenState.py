'''

This is a script to save the TOP hidden vectors from the Transformer to be used as training samples in the Generative model.

size_vae_dataset: -1 if using the complete dataset, if not an integer with the number of sentences to consider in the training set. The validation is completely computed.

Author: Aurora Cobo Aguilera
Date: 2021/05/21
Update: 2021/05/21

'''

from torch.utils.data import DataLoader
import numpy as np
from utils import *
from transformers import HfArgumentParser, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, AutoModelForMaskedLM
import h5py
import torch
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device, num_workers = set_device()

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, NoRBERTArguments))
model_args, data_args, training_args, norbert_args = parser.parse_args_into_dataclasses()


# Configure results directory
results_dir = training_args.output_dir
create_dir(results_dir)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

# Configure and load dataset
train_dataset = get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, num_workers=num_workers, shuffle=False)

maxsentences = norbert_args.size_vae_dataset
layer_deep = norbert_args.layer_deep
model_path = model_args.model_name_or_path


if len(train_dataset) < maxsentences or maxsentences == -1:
    maxsentences = len(train_dataset)


# Load the pre-trained Transformer model
model = AutoModelForMaskedLM.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)

# Change model to cpu/gpu
model = model.to(device)

# Evaluation mode
model.eval()


print('Process to save {}/{} sentences from training dataset'.format(maxsentences, len(train_dataset)))

maxit = int(np.floor(maxsentences / training_args.per_device_train_batch_size))
# Evaluate and save training samples
for it, tokens in enumerate(train_loader):
    # Converting these to cuda tensors if possible
    input_ids = tokens['input_ids'].to(device)
    labels = tokens['labels'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Predict all tokens
    with torch.no_grad():
        masked_lm_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        top_hidden_state = masked_lm_output.hidden_states[-1]

    if it == 0:
        save_last_hidden_state = np.copy(top_hidden_state[attention_mask == 1].cpu())
    else:
        save_last_hidden_state = np.concatenate((save_last_hidden_state, top_hidden_state[attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 25 == 0:
        print("Iteration {}/{} completed at training. ".format(it + 1, maxit))

    if it+1 == maxit:
        break

print('{} iterations completed'.format(it+1))

# creating a HDF5 file
decoded_file_name = results_dir + 'topHiddenState_{}_train.hdf5'.format(maxsentences)
print('Saving file of shape: ' + str(save_last_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=save_last_hidden_state)


print('Process to save {} sentences from validation dataset'.format(len(eval_dataset)))

# Evaluate and save test samples
for it, tokens in enumerate(val_loader):
    # Converting these to cuda tensors if possible
    input_ids = tokens['input_ids'].to(device)
    labels = tokens['labels'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Predict all tokens
    with torch.no_grad():
        masked_lm_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        top_hidden_state = masked_lm_output.hidden_states[-1]

    if it == 0:
        save_last_hidden_state = np.copy(top_hidden_state[attention_mask == 1].cpu())
    else:
        save_last_hidden_state = np.concatenate((save_last_hidden_state, top_hidden_state[attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 25 == 0:
        print("Iteration {}/{} completed at testing. ".format(it + 1, len(val_loader)))


# creating a HDF5 file
decoded_file_name = results_dir + 'topHiddenState_{}_test.hdf5'.format(len(eval_dataset))
print('Saving file of shape: ' + str(save_last_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=save_last_hidden_state)
