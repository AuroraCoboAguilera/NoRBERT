This is the project to pretrain and finetune both the baseline and NoRBERT on any of its variants, with BERT, RoBERTa and XLM-R. The main scripts are run_language_modeling_bleu.py and train_NoRBERT.py, both for training the models.

An example to use those scripts is:

For training:

python run_language_modeling_bleu.py --model_type "bert" --model_name_or_path "bert-base-uncased" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_100epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --num_train_epochs 100 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --logging_strategy "epoch" --evaluation_strategy "epoch"

python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_top_GMVAE_1_001_150_50_20_1500_6_03_5e-05_75epochs_lr5e-5/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --nor_path "../GMVAE/experiments/bert-base-uncased_sst2_10epochs_top/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_5e-05/config.pt" --option -1 --step_restore 500 --num_train_epochs 75 --learning_rate 5e-5 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --n_sentences -1 --logging_strategy 'epoch'

For evaluation:

python run_language_modeling_bleu.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3

python train_NoRBERT.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_1deep_GMVAE_1_0001_150_50_20_1500_6_03_5e-05_4000steps_30epochs_lr5e-5_ignoreCheckpoint_eval_mlm02/" --mlm --mlm_probability 0.2 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3 --n_sentences -1 --eval_mode


For saving hidden vectors:

python saveDeepHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/bert-base-uncased_sst2_10epochs/"

python saveTopHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/bert-base-uncased_sst2_10epochs/"

