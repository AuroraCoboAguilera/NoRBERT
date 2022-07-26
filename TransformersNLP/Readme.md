This is the project to pretrain and finetune both the baseline and NoRBERT on any of its variants, with BERT, RoBERTa and XLM-R. The main scripts are run_language_modeling_bleu.py and train_NoRBERT.py, both for training the models.

An example to use those scripts is:

For training:

python run_language_modeling_bleu.py --model_type "bert" --model_name_or_path "bert-base-uncased" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_100epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --num_train_epochs 100 --do_train --do_eval --save_total_limit 1 --eval_accumulation_steps 3 --logging_strategy "epoch" --evaluation_strategy "epoch"

For evaluation:

python run_language_modeling_bleu.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./models/bert-base-uncased_sst2_10epochs_eval_mlm015/" --mlm --mlm_probability 0.15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --do_eval --eval_accumulation_steps 3

For saving hidden vectors:
python saveDeepHiddenState.py --model_type "bert" --model_name_or_path "./models/bert-base-uncased_sst2_10epochs/" --mlm --mlm_probability 0.15 --per_device_train_batch_size=64 --per_device_eval_batch_size 64 --block_size 150 --layer_deep 1 --line_by_line --train_data_file "./dataset/SST2/train.csv" --eval_data_file "./dataset/SST2/dev.csv" --output_dir "./results/bert-base-uncased_sst2_10epochs/"
