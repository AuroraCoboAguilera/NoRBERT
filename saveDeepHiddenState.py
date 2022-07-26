'''

This is a script to save the DEEP hidden vectors from the Transformer to be used as training samples in the Generative model.
ONLY FOR BERT MODELS

layer_deep: The depth of the layer to same the hidden vectors from
size_vae_dataset: -1 if using the complete dataset, if not an integer with the number of sentences to consider in the training set. The validation is completely computed.

Author: Aurora Cobo Aguilera
Date: 2021/05/14
Update: 2022/02/18

'''

# EXAMPLES with an own file and with a GLUE dataset.
# CUDA_VISIBLE_DEVICES=3 python saveDeepHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/bert-base-uncased_sst2_10epochs/"
# CUDA_VISIBLE_DEVICES=3 python saveDeepHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_trec_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./results/bert-base-uncased_trec_10epochs/"
# CUDA_VISIBLE_DEVICES=0 python saveDeepHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./results/bert-base-uncased_multi30k_10epochs/"
# CUDA_VISIBLE_DEVICES=1 python saveDeepHiddenState.py --model_type "roberta" --model_name_or_path "./models/roberta-base_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/roberta-base_sst2_10epochs/"
# CUDA_VISIBLE_DEVICES=3 python saveDeepHiddenState.py --model_type "roberta" --model_name_or_path "./models/roberta-base_trec_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./results/roberta-base_trec_10epochs/"
# CUDA_VISIBLE_DEVICES=1 python saveDeepHiddenState.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=16 --per_device_eval_batch_size 16 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/xlm-roberta-base_sst2_10epochs/"
# CUDA_VISIBLE_DEVICES=3 python saveDeepHiddenState.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_trec_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./results/xlm-roberta-base_trec_10epochs/"
# CUDA_VISIBLE_DEVICES=3 python saveDeepHiddenState.py --model_type "roberta" --model_name_or_path "./models/roberta-base_multi30k_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./results/roberta-base_multi30k_10epochs/"
# CUDA_VISIBLE_DEVICES=1 python saveDeepHiddenState.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_multi30k_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./results/xlm-roberta-base_multi30k_10epochs/"


from models_NoRBERT import DeepNoRBERT
from models_NoRoBERTa import DeepNoRoBERTa
from models_NoRxlmRoBERTa import DeepNoRxlmRoBERTa
from utils import *
from transformers import HfArgumentParser, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
import h5py

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


# Load the pre-trained BERT model
if model_args.model_type == "bert":
    model = DeepNoRBERT.from_pretrained(model_path, GMVAE_model=None, save_deep_hidden_state=True, layer_deep=layer_deep, from_tf=bool(".ckpt" in model_path), output_hidden_states=True, output_attentions=True)
elif model_args.model_type == "roberta":
    model = DeepNoRoBERTa.from_pretrained(model_path, GMVAE_model=None, save_deep_hidden_state=True, layer_deep=layer_deep, from_tf=bool(".ckpt" in model_path), output_hidden_states=True, output_attentions=True)
elif model_args.model_type == "xlm-roberta":
    model = DeepNoRxlmRoBERTa.from_pretrained(model_path, GMVAE_model=None, save_deep_hidden_state=True, layer_deep=layer_deep, from_tf=bool(".ckpt" in model_path), output_hidden_states=True, output_attentions=True)

# Change model to cpu/gpu
model = model.to(device)

# Evaluation mode
model.eval()

print('Process to save {}/{} from training dataset'.format(maxsentences, len(train_dataset)))

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
        deep_hidden_state = masked_lm_output.deep_hidden_state
    if it == 0:
        saved_deep_hidden_state = np.copy(deep_hidden_state[attention_mask == 1].cpu())
    else:
        saved_deep_hidden_state = np.concatenate((saved_deep_hidden_state, deep_hidden_state[attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 25 == 0:
        print("Iteration {}/{} completed at training. ".format(it + 1, maxit))

    if it+1 == maxit:
        break

print('{} iterations completed'.format(it+1))

# creating a HDF5 file
if layer_deep != None:
    decoded_file_name = results_dir + '{}deepHiddenState_{}_train.hdf5'.format(layer_deep, maxsentences)
else:
    decoded_file_name = results_dir + 'deepHiddenState_{}_train.hdf5'.format(maxsentences)
print('Saving file of shape: ' + str(saved_deep_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=saved_deep_hidden_state)


print('Process to save {} from validation dataset'.format(len(eval_dataset)))

# Evaluate and save test samples
for it, tokens in enumerate(val_loader):
    # Converting these to cuda tensors if possible
    input_ids = tokens['input_ids'].to(device)
    labels = tokens['labels'].to(device)
    attention_mask = tokens['attention_mask'].to(device)


    # Predict all tokens
    with torch.no_grad():
        masked_lm_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        deep_hidden_state = masked_lm_output.deep_hidden_state

    if it == 0:
        saved_deep_hidden_state = np.copy(deep_hidden_state[attention_mask == 1].cpu())
    else:
        saved_deep_hidden_state = np.concatenate((saved_deep_hidden_state, deep_hidden_state[attention_mask == 1].cpu()), axis=0)

    if (it + 1) % 25 == 0:
        print("Iteration {}/{} completed at testing. ".format(it + 1, len(val_loader)))


# creating a HDF5 file
if layer_deep != None:
    decoded_file_name = results_dir + '{}deepHiddenState_{}_test.hdf5'.format(layer_deep, len(eval_dataset))
else:
    decoded_file_name = results_dir + 'deepHiddenState_{}_test.hdf5'.format(len(eval_dataset))
print('Saving file of shape: ' + str(saved_deep_hidden_state.shape))
with h5py.File(decoded_file_name, 'w') as f:
    dset = f.create_dataset("default", data=saved_deep_hidden_state)