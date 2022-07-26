'''
This is a python file that implements some funtions that are useful along all classes from the project.

Author: Aurora Cobo Aguilera
Date: 2021/05/14

'''

import numpy as np
import torch
import os
from dataclasses import dataclass, field
from glob import glob
from torch.utils.data import ConcatDataset
from datasets import load_dataset, load_metric
from typing import Callable, Dict, List, Optional, Tuple, Union, NamedTuple

from transformers import (LineByLineTextDataset,  #TODO:             lines = [line.replace('<UNK>', '[MASK]') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    MODEL_WITH_LM_HEAD_MAPPING,
    Trainer,
    TrainingArguments,
    EvalPrediction
    )

from torch.utils.data.dataset import Dataset, IterableDataset
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_utils import denumpify_detensorize
from transformers.utils import logging
import collections
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_numpify,
    nested_truncate)

if is_torch_tpu_available():
    import torch_xla.distributed.parallel_loader as pl


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
import random
import matplotlib.pyplot as plt
import warnings

logger = logging.get_logger(__name__)


#ACA
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def set_device():
    """Return the device (cpu/gpu) and number of workers regarding the availability of resources"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_workers = 6
    else:
        device = torch.device("cpu")
        num_workers = 0
    print("Device: " + str(device))
    return device, num_workers



def create_dir(name_dir):
    """Creates a directory if it does not exist."""
    directory, filename = os.path.split(os.path.abspath(name_dir))
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

@dataclass
class NoRBERTArguments:
    """
    Arguments for the configuration of NoRBERT.
    """
    # Used in savedDeepHiddenState/saveTopHiddenState
    layer_deep: int = field(
        default=None,
        metadata={"help": "Depth of the layer where to apply the GMVAE or save the embedding to train the GMVAE"}
    )
    size_vae_dataset: int = field(
        default=-1, metadata={"help": "Maximum number of sentences from which saving the embeddings to train the GMVAE."}
    )

    # Used in train_NoRBERT
    option: int = field(
        default=-1, metadata={"help": "The option of the model to train, TopNoRBERT_ (-1), Deep NoRBERT(layer to apply it), Double Deep NoRBERT (list of two layers)."}
    )
    double_flag: bool = field(
        default=False,
        metadata={"help": "Whether to use Double Deep NoRBERT."},
    )
    nor_path: str = field(
        default=None,
        metadata={
            "help": "path to config CGMVAE checkpoint."
        },
    )
    nor_path2: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to config CGMVAE checkpoint. It refers to the second layer if it is applied in 2 (double_flag=True)."
        },
    )
    step_restore: int = field(
        default=100, metadata={"help": "The step to be restored in the GMVAE model."}
    )
    step_restore2: int = field(
        default=100, metadata={"help": "The step to be restored in the second GMVAE model (double_flag=True)."}
    )
    n_sentences: int = field(
        default=50,
        metadata={
            "help": "Number of sentences to generate and save. Use -1 for all sentences in the datasets"
        },
    )
    ignore_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to checkpoint even if there is one to load."},
    )
    K_select: int = field(
        default=-1, metadata={"help": "Component in the GMVAE to be selected. If -1, this information is not used in the reconstruction. [0, K)."}
    )
    variance: float = field(
        default=0.1,
        metadata={
            "help": "sigma of the noise injection."
        },
    )
    contextual_tokens: int = field(
        default=5,
        metadata={
            "help": "Number of contextual tokens to save before and after the current one. Total dim = (2 x contextual_tokens + 1) x hidden"
        },
    )
    eval_mode: bool = field(
        default=False,
        metadata={"help": "Whether to use eval mode and deactivate GMVAE reconstruction."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    ignore_checkpt: bool = field(
        default=False,
        metadata={"help": "Whether to checkpoint even if there is one to load."},
    )
    multihead: bool = field(
        default=False,
        metadata={"help": "Whether to use 3 layers in the classification head. Only for BERT."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # ACA: Added parameter for training in a GLUE task. If keep None, not used any benchmark
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )



def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    return_all: bool = False,

):
    def _dataset(file_path, ref_path=None, return_all=False):
        # ACA: To integrate the glue benchmarks
        if args.task_name is not None:
            datasets = load_dataset("glue", args.task_name)
            # datasets = load_dataset("glue", args.task_name, split=['train[:10%]', 'test[:10%]']) #TODO

            sentence1_key, sentence2_key = task_to_keys[args.task_name]

            def preprocess_function(examples):
                # Tokenize the texts
                sents = (
                    (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
                )
                result = tokenizer(*sents, max_length=args.block_size, add_special_tokens=True, truncation=True)

                return result

            datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
            train_dataset = datasets["train"]
            eval_dataset = datasets["validation_matched" if args.task_name == "mnli" else "validation"]

            # train_dataset = datasets[0]
            # eval_dataset = datasets[1]
            # train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
            # eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

            if return_all:
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label', 'idx', 'sentence'])
                eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label', 'idx', 'sentence'])
            else:
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
                eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            if evaluate:
                return eval_dataset

            else:
                return train_dataset
        else:
            if args.line_by_line:
                if ref_path is not None:
                    if not args.whole_word_mask or not args.mlm:
                        raise ValueError("You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                    return LineByLineWithRefDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, ref_path=ref_path,)

                return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
            else:
                return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache, cache_dir=cache_dir,)

    if evaluate:
        return _dataset(args.eval_data_file, args.eval_ref_file, return_all=return_all)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f, return_all=return_all) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file, args.train_ref_file, return_all=return_all)


class Trainer_bleu(Trainer):

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        from_index = 1

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )


        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        labels_masked_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_labels_masked = None

        # Will be useful when we have an iterable dataset so don't know its length.

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            labels = inputs['input_ids'] * (inputs['labels'] == -100) + inputs['labels'] * (inputs['labels'] != -100)        # ACA: labels are the true ids of the tokens
            labels_masked = torch.clone(inputs['labels'])

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if labels_masked is not None:
                labels_masked = self._pad_across_processes(labels_masked)
                labels_masked = self._nested_gather(labels_masked)
                labels_masked_host = labels_masked if labels_masked_host is None else nested_concat(labels_masked_host, labels_masked, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)


            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None and labels_host is not None:
                    logits = nested_numpify(preds_host)
                    labels = nested_numpify(labels_host)
                    labels_masked = nested_numpify(labels_masked_host)
                    logits_ = np.argmax(logits, axis=2)
                    # Replace [UNK] by next best prediction, and check that is not [SEP]
                    indexs = np.argwhere(logits_ == self.tokenizer.unk_token_id)
                    for i in range(len(indexs)):
                        argsort_logit = (-logits[indexs[i, 0], indexs[i, 1]]).argsort()
                        next_best_logit = argsort_logit[1]
                        if next_best_logit == self.tokenizer.sep_token_id:
                            next_best_logit = argsort_logit[2]
                        logits_[indexs[i, 0], indexs[i, 1]] = next_best_logit
                    # Replace [SEP] by next best prediction, and check that is not [UNK]
                    indexs = np.argwhere(logits_ == self.tokenizer.sep_token_id)
                    for i in range(len(indexs)):
                        argsort_logit = (-logits[indexs[i, 0], indexs[i, 1]]).argsort()
                        next_best_logit = argsort_logit[1]
                        if next_best_logit == self.tokenizer.unk_token_id:
                            next_best_logit = argsort_logit[2]
                        logits_[indexs[i, 0], indexs[i, 1]] = next_best_logit
                    len_sentences = (labels_host == self.tokenizer.sep_token_id).nonzero(as_tuple=False)[:, 1]
                    len_sentences[len_sentences < from_index] = from_index
                    # To cut the length of the reconstructed sentences: ACA
                    # for ind, leng in enumerate(len_sentences):
                    #     logits_[ind, leng:] = self.tokenizer.sep_token_id
                    #     labels[ind, leng:] = self.tokenizer.sep_token_id
                    #     labels_masked[ind, leng:] = self.tokenizer.sep_token_id
                    logits = [self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]]) for i, item in enumerate(logits_)]
                    # logits = self.tokenizer.batch_decode(logits_[:, 3:], skip_special_tokens=False)          # ACA
                    # logits = [self.tokenizer.tokenize(item)[:len_sentences[i]-3] for i, item in enumerate(logits)]        #TODO: Skip 2 first items: class + , and skip CLS token
                    # all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    all_preds = logits if all_preds is None else all_preds + logits              # ACA
                # if labels_host is not None:
                    #     labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)  # ACA
                    labels = [[self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]])] for i, item in enumerate(labels)]
                    # labels = self.tokenizer.batch_decode(labels[:, 3:], skip_special_tokens=False)          # ACA
                    # labels = [[item] for item in labels]
                    # labels = [[self.tokenizer.tokenize(item)[:len_sentences[i]-3]] for i, item in enumerate(labels)]
                    # all_labels = (labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100))
                    all_labels = labels if all_labels is None else all_labels + labels              # ACA
                # if labels_masked_host is not None: #ACA
                    labels_masked[labels_masked == -100] = self.tokenizer.mask_token_id
                    labels_masked = [self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]]) for i, item in enumerate(labels_masked)]
                    # labels_masked = self.tokenizer.batch_decode(labels_masked[:, 3:], skip_special_tokens=False)
                    # labels_masked = [self.tokenizer.tokenize(item)[:len_sentences[i]-3] for i, item in enumerate(labels_masked)]
                    all_labels_masked = labels_masked if all_labels_masked is None else all_labels_masked + labels_masked

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, labels_masked_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None and labels_host is not None and labels_masked_host is not None:
            logits = nested_numpify(preds_host)
            labels = nested_numpify(labels_host)
            labels_masked = nested_numpify(labels_masked_host)
            logits_ = np.argmax(logits, axis=2)
            # Replace [UNK] by next best prediction, and check that is not [SEP]
            indexs = np.argwhere(logits_ == self.tokenizer.unk_token_id)
            for i in range(len(indexs)):
                argsort_logit = (-logits[indexs[i, 0], indexs[i, 1]]).argsort()
                next_best_logit = argsort_logit[1]
                if next_best_logit == self.tokenizer.sep_token_id:
                    next_best_logit = argsort_logit[2]
                logits_[indexs[i, 0], indexs[i, 1]] = next_best_logit
            # Replace [SEP] by next best prediction, and check that is not [UNK]
            indexs = np.argwhere(logits_ == self.tokenizer.sep_token_id)
            for i in range(len(indexs)):
                argsort_logit = (-logits[indexs[i, 0], indexs[i, 1]]).argsort()
                next_best_logit = argsort_logit[1]
                if next_best_logit == self.tokenizer.unk_token_id:
                    next_best_logit = argsort_logit[2]
                logits_[indexs[i, 0], indexs[i, 1]] = next_best_logit
            len_sentences = (labels_host == self.tokenizer.sep_token_id).nonzero(as_tuple=False)[:, 1]
            len_sentences[len_sentences < 3] = 3
            # To cut the length of the reconstructed sentences: ACA
            # for ind, leng in enumerate(len_sentences):
            #     logits_[ind, leng:] = self.tokenizer.sep_token_id
            #     labels[ind, leng:] = self.tokenizer.sep_token_id
            #     labels_masked[ind, leng:] = self.tokenizer.sep_token_id
            logits = [self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]]) for i, item in enumerate(logits_)]
            # logits = self.tokenizer.batch_decode(logits_[:, 3:], skip_special_tokens=False)  # ACA: Skip 2 first items: class + , and skip CLS token
            # logits = [self.tokenizer.tokenize(item)[:len_sentences[i]-3] for i, item in enumerate(logits) ]  # ACA
            # logits = self.tokenizer.batch_decode(logits, skip_special_tokens=True)  # ACA
            # logits = [self.tokenizer.tokenize(item) for item in logits]
            # all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            all_preds = logits if all_preds is None else all_preds + logits  # ACA
        # if labels_host is not None:
        #     labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)  # ACA
            labels = [[self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]])] for i, item in enumerate(labels)]
            # labels = self.tokenizer.batch_decode(labels[:, 3:], skip_special_tokens=False)  # ACA
            # labels = [[item] for item in labels]
            # labels = [[self.tokenizer.tokenize(item)[:len_sentences[i]-3]] for i, item in enumerate(labels)]
            # all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            all_labels = labels if all_labels is None else all_labels + labels  # ACA
        # if labels_masked_host is not None:
            labels_masked[labels_masked == -100] = self.tokenizer.mask_token_id
            labels_masked = [self.tokenizer.convert_ids_to_tokens(item[from_index:len_sentences[i]]) for i, item in enumerate(labels_masked)]
            # labels_masked = self.tokenizer.batch_decode(labels_masked[:, 3:], skip_special_tokens=False)
            # labels_masked = [self.tokenizer.tokenize(item)[:len_sentences[i]-3] for i, item in enumerate(labels_masked)]
            all_labels_masked = labels_masked if all_labels_masked is None else all_labels_masked + labels_masked

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_labels_masked is not None:
            all_labels_masked = nested_truncate(all_labels_masked, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), self.tokenizer, self.args.output_dir, metric_key_prefix, all_labels_masked)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

def saving_sentences(output_dir, tokenizer, true_tokens, predicted_tokens, name='', number=50, all_labels_masked=None):
    file = open(output_dir + '/text_reconstruction_' + name + '_' + str(number) +'.txt', 'w', encoding="utf-8")
    file2 = open(output_dir + '/samples_reconstructed_' + name + '_' + str(number) +'.txt', 'w', encoding="utf-8")
    file_ = open(output_dir + '/text_reconstruction_onlyMask_' + name + '_' + str(number) + '.txt', 'w', encoding="utf-8")
    file2_ = open(output_dir + '/samples_reconstructed_onlyMask_' + name + '_' + str(number) + '.txt', 'w', encoding="utf-8")

    indexs = random.sample(range(0, len(true_tokens)), number)
    for ind in indexs:
        if len(true_tokens[ind][0]) > 0:
            true_sentence = tokenizer.convert_tokens_to_string(true_tokens[ind][0])
            predicted_sentence = tokenizer.convert_tokens_to_string(predicted_tokens[ind])

            # true_sentence.replace(tokenizer.sep_token, ' ')
            # predicted_sentence.replace(tokenizer.sep_token, ' ')

            file.write('True: ' + true_sentence + '\n')
            file_.write('True: ' + true_sentence + '\n')
            # print('True: ' + true_sentence + '\n')
            # print('Tokens: ' + ' '.join(true_tokens[ind][0]) + '\n')
            file.write('Pred: ' + predicted_sentence + '\n')
            file2.write(predicted_sentence + '\n')

            predicted_tokens_onlyMask = true_tokens[ind][0].copy()
            indexs_masked = [i for i, val in enumerate(all_labels_masked[ind]) if val==tokenizer.mask_token]
            if len(predicted_tokens[ind]) == len(all_labels_masked[ind]):
                for index in indexs_masked:
                    predicted_token_new = predicted_tokens[ind][index]
                    predicted_tokens_onlyMask[index] = predicted_token_new
                predicted_sentence_onlyMasked = tokenizer.convert_tokens_to_string(predicted_tokens_onlyMask)
                predicted_sentence_onlyMasked.replace(tokenizer.sep_token, ' ')

                file_.write('Pred: ' + predicted_sentence_onlyMasked + '\n')
                file2_.write(predicted_sentence_onlyMasked + '\n')
            else:
                print('BAD\n')

    file.close()
    file2.close()
    file_.close()
    file2_.close()



def plot_dict(dict_arrays, start_step=0, step_size=1, use_title=None, points_values=False, points_round=3,
              use_xlabel=None, use_xticks=True, use_rotation_xticks=0, xticks_labels=None, use_ylabel=None,
              style_sheet='ggplot', use_grid=True, use_linestyles=None, font_size=None, width=3, height=1, magnify=1.2,
              use_dpi=50, path=None, show_plot=True):
    r"""
    Create plot from a single array of values.
    Arguments:
        dict_arrays (:obj:`dict([list])`):
            Dictionary of arrays that will get plotted. The keys in dictionary are used as labels and the values as
            arrays that get plotted.
        start_step (:obj:`int`, `optional`, defaults to :obj:`0`):
            Starting value of plot.This argument is optional and it has a default value attributed inside
            the function.
        step_size (:obj:`int`, `optional`, defaults to :obj:`q`):
            Steps shows on x-axis. Change if each steps is different than 1.This argument is optional and it has a
            default value attributed inside the function.
        use_title (:obj:`int`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed
            inside the function.
        points_values (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Display each point value on the plot. This argument is optional and it has a default value attributed
            inside the function.
        points_round (:obj:`int`, `optional`, defaults to :obj:`1`):
            Round decimal valus for points values. This argument is optional and it has a default value attributed
            inside the function.
        use_xlabel (:obj:`str`, `optional`):
            Label to use for x-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.
        use_xticks (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Display x-axis tick values (the values at each point). This argument is optional and it has a default
            value attributed inside the function.
        use_ylabel (:obj:`str`, `optional`):
            Label to use for y-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.
        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.
        use_grid (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Show grid on plot or not. This argument is optional and it has a default value attributed inside
            the function.
        use_linestyles (:obj:`str`, `optional`, defaults to :obj:`-`):
            Style to use on line from ['-', '--', '-.', ':']. This argument is optional and it has a default
            value attributed inside the function.
        font_size (:obj:`int` or `float`, `optional`):
            Font size to use across the plot. By default this function will adjust font size depending on `magnify`
            value. If this value is set, it will ignore the `magnify` recommended font size. The title font size is by
            default `1.8` greater than font-size. This argument is optional and it will have a `None` value attributed
            inside the function.
        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.
        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Height length of plot in inches. This argument is optional and it has a default value attributed inside
            the function.
        magnify (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            Ratio increase of both with and height keeping the same ratio size. This argument is optional and it has a
            default value attributed inside the function.
        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Print resolution is measured in dots per inch (or “DPI”). This argument is optional and it has a default
            value attributed inside the function.
        path (:obj:`str`, `optional`):
            Path and file name of plot saved as image. If want to save in current path just pass in the file name.
            This argument is optional and it will have a None value attributed inside the function.
        show_plot (:obj:`bool`, `optional`, defaults to :obj:`1`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.
    Raises:
        ValueError: If `dict_arrays` is not of type `dictionary`.
        ValueError: If `dict_arrays` doesn't have string keys.
        ValueError: If `dict_arrays` doesn't have array values.
        ValueError: If `style_sheet` is not valid.
        ValueError: If `use_linestyle` is not valid.
        ValueError: If `points_values`of type list don't have same length as `dict_arrays`.
        DeprecationWarning: If `magnify` is se to values that don't belong to [0, 1] values.
        ValueError: If `font_size` is not `None` and smaller or equal to 0.
    """

    # Check if `dict_arrays` is the correct format.
    if not isinstance(dict_arrays, dict):
        # Raise value error.
        raise ValueError("`dict_arrays` needs to be a dictionary of values!")

    # Check each label
    for label, array in dict_arrays.items():
        # Check if format is correct.
        if not isinstance(label, str):
            # Raise value error.
            raise ValueError("`dict_arrays` needs string keys!")
        if not isinstance(array, list) or isinstance(array, np.ndarray):
            # Raise value error.
            raise ValueError("`dict_arrays` needs lists values!")

    # Make sure style sheet is correct.
    if style_sheet in plt.style.available:
        # Set style of plot
        plt.style.use(style_sheet)
    else:
        # Style is not correct.
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))

    # Make sure `magnify` is in right range.
    if magnify > 1 or magnify <= 0:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify` needs to have value in [0,1]! `{magnify}` will be converted to `0.1` as default.',
                      DeprecationWarning)
        # Convert to regular value 0.1.
        magnify = 0.1

    # all linestyles.
    linestyles = ['-', '--', '-.', ':']

    # Make sure `font_size` is set right.
    if (font_size is not None) and (font_size <= 0):
        # Raise value error -  is not correct.
        raise ValueError(f'`font_size` needs to be positive number! Invalid value {font_size}')

    # Font size select custom or adjusted on `magnify` value.
    font_size = font_size if font_size is not None else np.interp(magnify, [0.1, 1], [10.5, 50])

    # Font variables dictionary. Keep it in this format for future updates.
    font_dict = dict(
        family='DejaVu Sans',
        color='black',
        weight='normal',
        size=font_size,
    )

    # If single style value is passed, use it on all arrays.
    if use_linestyles is None:
        use_linestyles = ['-'] * len(dict_arrays)

    else:
        # Check if linestyle is set right.
        for use_linestyle in use_linestyles:
            if use_linestyle not in linestyles:
                # Raise error.
                raise ValueError("`linestyle=%s` is not in the styles: %s!" % (str(use_linestyle), str(linestyles)))

    # Check `points_value` type - it can be bool or list(bool).
    if isinstance(points_values, bool):
        # Convert to list.
        points_values = [points_values] * len(dict_arrays)
    elif isinstance(points_values, list) and (len(points_values) != len(dict_arrays)):
        raise ValueError('`points_values` of type `list` must have same length as dictionary!')

    # Single plot figure.
    #plt.subplot(1, 2, 1)
    plt.figure()

    # Use maximum length of steps. In case each arrya has different lengths.
    max_steps = []

    # Plot each array.
    for index, (use_label, array) in enumerate(dict_arrays.items()):
        # Set steps plotted on x-axis - we can use step if 1 unit has different value.
        if start_step > 0:
            # Offset all steps by start_step.
            steps = np.array(range(0, len(array))) * step_size + start_step
            max_steps = steps if len(max_steps) < len(steps) else max_steps
        else:
            steps = np.array(range(1, len(array) + 1)) * step_size
            max_steps = steps if len(max_steps) < len(steps) else max_steps

        # Plot array as a single line.
        plt.plot(steps, array, linestyle=use_linestyles[index], label=use_label)

        # Plots points values.
        if points_values[index]:
            # Loop through each point and plot the label.
            for x, y in zip(steps, array):
                # Add text label to plot.
                plt.text(x, y, str(round(y, points_round)), fontdict=font_dict)

    # Set horizontal axis name.
    plt.xlabel(use_xlabel, fontdict=font_dict)

    # Use x ticks with steps or labels.
    plt.xticks(max_steps, xticks_labels, rotation=use_rotation_xticks) if use_xticks else None

    # Set vertical axis name.
    plt.ylabel(use_ylabel, fontdict=font_dict)

    # Adjust both axis labels font size at same time.
    plt.tick_params(labelsize=font_dict['size'])

    # Place legend best position.
    plt.legend(loc='best', fontsize=font_dict['size'])

    # Adjust font for title.
    font_dict['size'] *= 1.8

    # Set title of figure.
    plt.title(use_title, fontdict=font_dict)

    # Rescale `magnify` to be used on inches.
    magnify *= 15

    # Display grid depending on `use_grid`.
    plt.grid(use_grid)

    # Make figure nice.
    plt.tight_layout()

    # Get figure object from plot.
    fig = plt.gcf()

    # Get size of figure.
    figsize = fig.get_size_inches()

    # Change size depending on height and width variables.
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]

    # Set the new figure size with magnify.
    fig.set_size_inches(figsize)

    # There is an error when DPI and plot size are too large!
    try:
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    except ValueError:
        # Deprecation warning from last time.
        warnings.warn(f'`magnify={magnify // 15}` is to big in combination'
                      f' with `use_dpi={use_dpi}`! Try using lower values for'
                      f' `magnify` and/or `use_dpi`. Image was saved in {path}'
                      f' with `use_dpi=50 and `magnify={magnify // 15}`!', Warning)
        # Set DPI to smaller value and warn user to use smaller magnify or smaller dpi.
        use_dpi = 50
        # Save figure to image if path is set.
        fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None

    # Show plot.
    plt.show() if show_plot is True else None

    return
