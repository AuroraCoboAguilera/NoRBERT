"""
This a script used for training NoRBERT once the GMVAE has been trained with some hidden states from BERT.
It works for base model BERT/RoBERTa/xlm-RoBERTa

Author: Aurora Cobo Aguilera
Date: May 2021
Updated: January 2022
"""

# Libraries
from models_NoRBERT import TopNoRBERT, DeepNoRBERT
from models_NoRoBERTa import TopNoRoBERTa, DeepNoRoBERTa
from models_NoRxlmRoBERTa import TopNoRxlmRoBERTa, DeepNoRxlmRoBERTa
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EvalPrediction,
)
import torch
import sys
import os
import logging
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_metric
from utils import ModelArguments, DataTrainingArguments, get_dataset, saving_sentences, set_device, NoRBERTArguments, Trainer_bleu, plot_dict
import math
import numpy as np


# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_75epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_sst2_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/bert-base-uncased_trec_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_75epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_trec_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/bert-base-uncased_multi30k_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_75epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_multi30k_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/roberta-base_sst2_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/roberta-base_sst2_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/roberta-base_trec_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/roberta-base_trec_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1--logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=1 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/roberta-base_multi30k_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/roberta-base_multi30k_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=1 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/xlm-roberta-base_trec_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_trec_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/xlm-roberta-base_sst2_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_sst2_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/xlm-roberta-base_multi30k_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_30epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_multi30k_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'

# CUDA_VISIBLE_DEVICES=1 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --nor_path "../GMVAE/experiments/bert-base-uncased_sst2_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 4000 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy "epoch" --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/bert-base-uncased_trec_10epochs_1deep_GMVAE_1_1e-05_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_trec_10epochs_1deep/checkpoints/GMVAE_1_1e-05_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 5000 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy "epoch" --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_30epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_multi30k_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 2000 --num_train_epochs 30 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy "epoch" --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/roberta-base_trec_10epochs_1deep_GMVAE_1_1e-06_150_50_20_1500_6_03_5e-05_5000steps_50epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/roberta-base_trec_10epochs_1deep/checkpoints/GMVAE_1_1e-06_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 5000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --nor_path "../GMVAE/experiments/roberta-base_sst2_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 4000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=1 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/roberta-base_multi30k_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 2000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_trec_10epochs/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/xlm-roberta-base_trec_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_trec_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 5000 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/xlm-roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_multi30k_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 2000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/xlm-roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --nor_path "../GMVAE/experiments/xlm-roberta-base_sst2_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 4000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch' --evaluation_strategy "epoch" --ignore_checkpoint

#TODO
# CUDA_VISIBLE_DEVICES=2 python3.8 train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --nor_path "../GMVAE/experiments/bert-base-uncased_sst2_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 4000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy "epoch" --evaluation_strategy "epoch" --ignore_checkpoint --multihead
# CUDA_VISIBLE_DEVICES=3 python3.8 train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_multi30k_10epochs_1deep/checkpoints/GMVAE_1_00001_150_50_20_1500_6_03_5e-05/config.pt" --option 1 --step_restore 2000 --num_train_epochs 50 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy "epoch" --evaluation_strategy "epoch" --ignore_checkpoint --multihead


# EVAL MODE
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint_eval_mlm02/" --mlm --mlm_probability 0.2 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_trec_10epochs_1deep_GMVAE_1_1e-05_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/bert-base-uncased_trec_10epochs_1deep_GMVAE_1_1e-05_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=3 python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_2000steps_30epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_2000steps_30epochs_lr5e-5_ignoreCheckpoint_eval_mlm06/" --mlm --mlm_probability 0.6 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_trec_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_5000steps_30epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/roberta-base_trec_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_5000steps_30epochs_lr5e-5_ignoreCheckpoint_eval_mlm02/" --mlm --mlm_probability 0.2 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint_eval_mlm02/" --mlm --mlm_probability 0.2 --per_device_train_batch_size=32 --per_device_eval_batch_size 32 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "roberta" --model_name_or_path "./models/roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint_eval_mlm02/" --mlm --mlm_probability 0.2 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_trec_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/TREC/train.csv" --eval_data_file "./dataset/TREC/dev.csv" --output_dir "./models/xlm-roberta-base_trec_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_5000steps_75epochs_lr5e-5_ignoreCheckpoint_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=2 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/xlm-roberta-base_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode
# CUDA_VISIBLE_DEVICES=0 python train_NoRBERT.py --model_type "xlm-roberta" --model_name_or_path "./models/xlm-roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_75epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/xlm-roberta-base_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_75epochs_lr5e-5_ignoreCheckpoint_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=8 --per_device_eval_batch_size 8 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode

#TODO
# CUDA_VISIBLE_DEVICES=1 python3.8 train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_4000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode --multihead
# CUDA_VISIBLE_DEVICES=2 python3.8 train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead/" --line_by_line --train_data_file "./dataset/multi30k/multi30k_train.txt" --eval_data_file "./dataset/multi30k/multi30k_test.txt" --output_dir "./models/bert-base-uncased_multi30k_10epochs_1deep_GMVAE_1_00001_150_50_20_1500_6_03_5e-05_2000steps_50epochs_lr5e-5_ignoreCheckpoint_multiHead_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode --multihead


# CGMVAE Model
add_path = '../GMVAE'
sys.path.append(os.path.abspath(add_path))

from GMVAE import *

logger = logging.getLogger(__name__)


device, num_workers = set_device()

def main():
    # See all possible arguments in src/transformers/training_args.py and ./utils.py or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, NoRBERTArguments))
    model_args, data_args, training_args, norbert_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN, )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   training_args.local_rank, training_args.device, training_args.n_gpu, bool(training_args.local_rank != -1), training_args.fp16, )

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # ---------------------------------------------------------------------------------------------------------------
    # Load the pre-trained GMVAE model
    # ---------------------------------------------------------------------------------------------------------------
    GMVAE_model = None
    if not(norbert_args.eval_mode):         # We do not use the GMVAE reconstruction in eval mode
        logger.info('Loading GMVAE model from checkpoint in %s...' % (norbert_args.nor_path))
        checkpoint = torch.load(norbert_args.nor_path, map_location=lambda storage, loc: storage)
        config = checkpoint['config']
        config.step_restore = norbert_args.step_restore
        if device == torch.device("cpu"):
            config.cuda = 0
        GMVAE_model = GMVAE(config, seq2seq=True)
        GMVAE_model.restore_model(add_path=add_path)

        print('CONTEXTUAL LENGTH')
        print(GMVAE_model.contextual_length)

        if norbert_args.double_flag:
            logger.info('Loading GMVAE model from checkpoint in %s...' % (norbert_args.nor_path2))
            checkpoint = torch.load(norbert_args.nor_path2, map_location=lambda storage, loc: storage)
            config = checkpoint['config']
            config.step_restore = norbert_args.step_restore2
            if device == torch.device("cpu"):
                config.cuda = 0
            GMVAE_model2 = GMVAE(config, seq2seq=True)
            GMVAE_model2.restore_model(add_path=add_path)

    # ---------------------------------------------------------------------------------------------------------------
    # Load the pre-trained BERT model in the corresponding layers of TopNoRBERT_ or DeepNoRBERT
    # ---------------------------------------------------------------------------------------------------------------
    if norbert_args.double_flag:
        option_list = [int(item) for item in norbert_args.option.split(' ')]
        if model_args.model_type == "bert":
            model = DeepNoRBERT.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=[GMVAE_model, GMVAE_model2], layer_deep=option_list, double_flag=True, from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select, multiHead=model_args.multihead)
        elif model_args.model_type == "roberta":
            model = DeepNoRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=[GMVAE_model, GMVAE_model2], layer_deep=option_list, double_flag=True, from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
        elif model_args.model_type == "xlm-roberta":
            model = DeepNoRxlmRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=[GMVAE_model, GMVAE_model2], layer_deep=option_list, double_flag=True, from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
    else:
        if norbert_args.option == -1:
            if model_args.model_type == "bert":
                model = TopNoRBERT.from_pretrained(model_args.model_name_or_path, freeze_bert_encoder=True, GMVAE_model=GMVAE_model,  from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
            elif model_args.model_type == "roberta":
                model = TopNoRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_encoder=True, GMVAE_model=GMVAE_model, from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
            elif model_args.model_type == "xlm-roberta":
                model = TopNoRxlmRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_encoder=True, GMVAE_model=GMVAE_model, from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
        else:
            if model_args.model_type == "bert":
                model = DeepNoRBERT.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=GMVAE_model, layer_deep=int(norbert_args.option), from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select, multiHead=model_args.multihead)
            elif model_args.model_type == "roberta":
                model = DeepNoRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=GMVAE_model, layer_deep=int(norbert_args.option), from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)
            elif model_args.model_type == "xlm-roberta":
                model = DeepNoRxlmRoBERTa.from_pretrained(model_args.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=GMVAE_model, layer_deep=int(norbert_args.option), from_tf=bool(".ckpt" in model_args.model_name_or_path), K_select=norbert_args.K_select)

    model.to(device)

    logger.info("NoR()BERT() parameters %s", norbert_args)

    # ---------------------------------------------------------------------------------------------------------------
    # Load tokenizer and dataset
    # ---------------------------------------------------------------------------------------------------------------
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Configure and load dataset and data collator
    train_dataset = get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir, return_all=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)


    # ---------------------------------------------------------------------------------------------------------------
    # Look for checkpoints
    # ---------------------------------------------------------------------------------------------------------------
    # Load last checkpoint in case it exists
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ---------------------------------------------------------------------------------------------------------------
    # Configure evaluation metrics
    # ---------------------------------------------------------------------------------------------------------------
    # Get the metric function
    metric = load_metric("bleu")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field), tokenizer, output_dir, metric_key_prefix and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction, tokenizer, output_dir, metric_key_prefix, all_labels_masked):
        # compute bleu score
        result = metric.compute(predictions=p.predictions, references=p.label_ids)
        result['precisions'] = np.mean(result['precisions'])    # ACA: I obtained an error in logger of tensorflow to save a list of 4 values as an scalar so I aplly the mean
        if norbert_args.n_sentences == -1:
            N = len(p.label_ids)
        else:
            N = norbert_args.n_sentences
        # Save some/all (with -1) examples of text
        saving_sentences(output_dir, tokenizer, p.label_ids, p.predictions, name=metric_key_prefix, number=N, all_labels_masked=all_labels_masked)

        return result


    # Initialize our Trainer
    trainer = Trainer_bleu(     #ACA: Trainer modified to compute bleu score and save sentences in the evaluation step
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        if norbert_args.ignore_checkpoint:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["hidden_states", "attentions", "deep_hidden_state"])
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # ACA: plot loss
            # Keep track of train and evaluate loss.
            loss_history = {'train_loss': [], 'eval_loss': []}


            # Loop through each log history.
            for log_history in trainer.state.log_history:

                if 'loss' in log_history.keys():
                    # Deal with training loss.
                    loss_history['train_loss'].append(log_history['loss'])

                if 'eval_loss' in log_history.keys():
                    # Deal with eval loss.
                    loss_history['eval_loss'].append(log_history['eval_loss'])

            if training_args.logging_strategy == 'epoch':
                step_size = 1
                use_xlabel = 'Epochs'
            else:
                step_size = training_args.logging_steps
                use_xlabel = 'Iterations'

            # Plot Losses.
            plot_dict(loss_history, start_step=step_size,
                      step_size=step_size, use_title='Loss',
                      use_xlabel=use_xlabel, use_ylabel='Values', magnify=2,
                      path=os.path.join(training_args.output_dir, "trainer_state_loss.pdf"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = ['train', 'validation']
        eval_datasets = [train_dataset, eval_dataset]

        for dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=dataset, ignore_keys=["hidden_states", "attentions", "deep_hidden_state"], metric_key_prefix=task)

            perplexity = math.exp(eval_result[task + "_loss"])
            eval_result["perplexity"] = perplexity

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

if __name__ == "__main__":
    main()