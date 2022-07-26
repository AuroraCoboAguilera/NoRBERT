"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

"""
This a script modified for the library Hugging face and adapted to my project. It is used for pre-training without the Regularizer.

Author: Aurora Cobo Aguilera
Date: May 2021
"""

# Libraries
import logging
import math
import os
import random
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from utils import ModelArguments, DataTrainingArguments, get_dataset, Trainer_bleu, saving_sentences, plot_dict
from datasets import load_metric
from models_BERT_multiMLHead import BertLMmultiHeadModel



logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py and ./utils.py or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval and data_args.task_name == None:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file or remove the --do_eval argument.")
    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   training_args.local_rank, training_args.device, training_args.n_gpu, bool(training_args.local_rank != -1), training_args.fp16,)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # ---------------------------------------------------------------------------------------------------------------
    # Load pretrained model and tokenizer
    # ---------------------------------------------------------------------------------------------------------------
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it, and load it from here, using --tokenizer_name")

    if model_args.multihead:
        model = BertLMmultiHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        if model_args.model_name_or_path:
            model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm flag (masked language modeling).")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets
    train_dataset = (get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir))
    eval_dataset = (get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir) if training_args.do_eval else None)

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


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
        N = len(p.label_ids)
        # Save some examples of text
        saving_sentences(output_dir, tokenizer, p.label_ids, p.predictions, name=metric_key_prefix, number=N, all_labels_masked=all_labels_masked)

        return result

    # Initialize our Trainer
    trainer = Trainer_bleu(     #ACA: Trainer modified to compute bleu score and save sentences in the evaluation step
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
        if ModelArguments.ignore_checkpt:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["hidden_states", "attentions", "deep_hidden_state"])
        metrics = train_result.metrics

        trainer.save_model()    # Saves the tokenizer too for easy upload
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        # For convenience, we also re-save the tokenizer to the same directory,
        if trainer.is_world_process_zero():     # ACA
            tokenizer.save_pretrained(training_args.output_dir)
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


    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
