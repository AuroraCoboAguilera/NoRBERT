from torch.utils.data import DataLoader
from lineTextDataset import LineTextDataset
from NoRBERT import NoRBERT
from DeepNoRBERT import DeepNoRBERT
import torch
import numpy as np
import sys, os
import logging
import glob
import re
import shutil
from typing import Dict, List, Tuple
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from tqdm import tqdm, trange
import random

# CGMVAE Model
add_path = '../GMVAE'
sys.path.append(os.path.abspath(add_path))

from GMVAE import *

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def evaluate(args, model, eval_dataset, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=5, shuffle=False)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for (tokens, masked_lm_labels, mask_token_index, sentence, sentence_true) in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = tokens['input_ids'].squeeze(1)
        token_type_ids = tokens['token_type_ids'].squeeze(1)
        attention_mask = tokens['attention_mask'].squeeze(1)

        # Converting these to cuda tensors
        input_ids = input_ids.to(opt.device)
        token_type_ids = token_type_ids.to(opt.device)
        attention_mask = attention_mask.to(opt.device)
        masked_lm_labels = masked_lm_labels.to(opt.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)

            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

# Load arguments and configuration
parser = argparse.ArgumentParser()
parser.add_argument("--option", default=-1, help="The option of the model to train, NoRBERT (-1), Deep NoRBERT(layer to apply it), Double Deep NoRBERT (list of two layers)",)
parser.add_argument("--double_flag", action="store_true", help="Whether to use Double Deep NoRBERT")

parser.add_argument("--model_name_or_path", type=str, default='./models/NoRBERT_dataset13_GMVAE_1_001_150_50_20_1500_6_03_1e-05', help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",)
parser.add_argument('--dataset', default=13, help='number of the dataset with its configuration to be loaded')
parser.add_argument("--output_dir", type=str, default='./models/NoRBERT_dataset13_GMVAE_1_001_150_50_20_1500_6_03_1e-05_2', help="The output directory where the model predictions and checkpoints will be written.",)
parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nor_path', help='path to config CGMVAE checkpoint', default='../GMVAE/experiments/bert_base_dataset13/checkpoints/GMVAE_1_001_150_50_20_1500_6_03_1e-05/config.pt')
parser.add_argument('--nor_path2', help='path to config CGMVAE checkpoint',type=str, default=None)
parser.add_argument('--step_restore', type=int, default=100)
parser.add_argument('--step_restore2', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument('--gpu_id', type=int, default=0, help="GPU id to use")

parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

opt = parser.parse_args()

if opt.should_continue:
    sorted_checkpoints = _sorted_checkpoints(opt)
    if len(sorted_checkpoints) == 0:
        raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
    else:
        opt.model_name_or_path = sorted_checkpoints[-1]
if (os.path.exists(opt.output_dir) and os.listdir(opt.output_dir)):
    logger.warning("Output directory ({}) already exists and is not empty. Proccedings overcome.".format(opt.output_dir))

# Setup CUDA, GPU & distributed training
if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cuda", opt.gpu_id)
else:
    opt.device = torch.device("cpu")


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.warning(
    "gpu_id: %s, device: %s",
    opt.gpu_id,
    opt.device,
)




maxlen = 100

# Creating instances of training and validation set
if opt.dataset == '1':
    train_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_train.txt', filename_true='./dataset/multi30k/multi30k_train.txt', maxlen=maxlen, model_name=opt.model_name_or_path, mlm_generator=True)
    val_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_test.txt', filename_true='./dataset/multi30k/multi30k_test.txt', maxlen=maxlen, model_name=opt.model_name_or_path, mlm_generator=True)
    data_name = 'multi30k'
elif opt.dataset == '13':
    train_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=opt.model_name_or_path, change_unk_token=True)
    val_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=opt.model_name_or_path, change_unk_token=True)
    data_name = 'multi30k_unk'
elif opt.dataset == '14':
    train_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_train_unk06.txt', filename_true='./dataset/multi30k/multi30k_train_true.txt', maxlen=maxlen, model_name=opt.model_name_or_path, change_unk_token=True)
    val_dataset = LineTextDataset(filename='./dataset/multi30k/multi30k_test_unk06.txt', filename_true='./dataset/multi30k/multi30k_test_true.txt', maxlen=maxlen, model_name=opt.model_name_or_path, change_unk_token=True)
    data_name = 'multi30k_unk06'
elif opt.dataset == 'snli':
    train_dataset = LineTextDataset(filename='./dataset/data_snli/train.txt', filename_true='./dataset/data_snli/train.txt', maxlen=maxlen, model_name=opt.model_name_or_path, mlm_generator=True)
    val_dataset = LineTextDataset(filename='./dataset/data_snli/test.txt', filename_true='./dataset/data_snli/test.txt', maxlen=maxlen, model_name=opt.model_name_or_path, mlm_generator=True)
    data_name = 'snli'

print(opt)
# Creating instances of training dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=5, shuffle=True)


# Set seed
set_seed(opt.seed)

# -------------------------------------
# Load the pre-trained GMVAE model
# -------------------------------------

logger.info('Loading GMVAE model from checkpoint in %s...' % (opt.nor_path))
checkpoint = torch.load(opt.nor_path, map_location=lambda storage, loc: storage)
config = checkpoint['config']
config.step_restore = opt.step_restore
if opt.device == torch.device("cpu"):
    config.cuda = 0
GMVAE_model = GMVAE(config, seq2seq=True)
GMVAE_model.restore_model(add_path=add_path)

if opt.double_flag:
    logger.info('Loading GMVAE model from checkpoint in %s...' % (opt.nor_path2))
    checkpoint = torch.load(opt.nor_path, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    config.step_restore = opt.step_restore2
    if opt.device == torch.device("cpu"):
        config.cuda = 0
    GMVAE_model2 = GMVAE(config, seq2seq=True)
    GMVAE_model2.restore_model(add_path=add_path)

# -------------------------------------
# Load the pre-trained BERT model in the corresponding layers of NoRBERT or DeepNoRBERT
# -------------------------------------
if opt.double_flag:
    option_list = [int(item) for item in opt.option.split(' ')]
    model = DeepNoRBERT.from_pretrained(opt.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=[GMVAE_model, GMVAE_model2], layer_deep=option_list, double_flag=True, from_tf=bool(".ckpt" in opt.model_name_or_path))
else:
    if opt.option == -1:
        model = NoRBERT.from_pretrained(opt.model_name_or_path, freeze_bert_encoder=True, GMVAE_model=GMVAE_model,  from_tf=bool(".ckpt" in opt.model_name_or_path))
    else:
        model = DeepNoRBERT.from_pretrained(opt.model_name_or_path, freeze_bert_lowerEncoder=True, GMVAE_model=GMVAE_model, layer_deep=int(opt.option), from_tf=bool(".ckpt" in opt.model_name_or_path))
model.to(opt.device)




logger.info("Training/evaluation parameters %s", opt)


# TRAINING
tb_writer = SummaryWriter()

if opt.max_steps > 0:
    t_total = opt.max_steps
    opt.num_epochs = opt.max_steps // (len(train_dataloader) // opt.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // opt.gradient_accumulation_steps * opt.num_epochs

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": opt.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=opt.warmup_steps, num_training_steps=t_total
)


# Check if saved optimizer or scheduler states exist
if (
    opt.model_name_or_path
    and os.path.isfile(os.path.join(opt.model_name_or_path, "optimizer.pt"))
    and os.path.isfile(os.path.join(opt.model_name_or_path, "scheduler.pt"))
):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(opt.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(opt.model_name_or_path, "scheduler.pt")))

# Train!
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_dataset))
logger.info("  Num Epochs = %d", opt.num_epochs)
logger.info(
    "  Total train batch size (w. accumulation) = %d",
    opt.batch_size
    * opt.gradient_accumulation_steps,
)
logger.info("  Gradient Accumulation steps = %d", opt.gradient_accumulation_steps)
logger.info("  Total optimization steps = %d", t_total)

global_step = 0
epochs_trained = 0
steps_trained_in_current_epoch = 0
# Check if continuing training from a checkpoint
if opt.model_name_or_path and os.path.exists(opt.model_name_or_path):
    try:
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = opt.model_name_or_path.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(train_dataloader) // opt.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // opt.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except ValueError:
        logger.info("  Starting fine-tuning.")

tr_loss, logging_loss = 0.0, 0.0

model.zero_grad()
train_iterator = trange(epochs_trained, int(opt.num_epochs), desc="Epoch")
set_seed(opt.seed)  # Added here for reproducibility

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, (tokens, masked_lm_labels, mask_token_index, sentence, sentence_true) in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        input_ids = tokens['input_ids'].squeeze(1)
        token_type_ids = tokens['token_type_ids'].squeeze(1)
        attention_mask = tokens['attention_mask'].squeeze(1)

        # Converting these to cuda tensors
        input_ids = input_ids.to(opt.device)
        token_type_ids = token_type_ids.to(opt.device)
        attention_mask = attention_mask.to(opt.device)
        masked_lm_labels = masked_lm_labels.to(opt.device)

        model.train()

        # Obtaining the loss from the model
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc) (loss, token_logits, hidden_state, attentions)

        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps

        # Backpropagating the gradients
        loss.backward()

        tr_loss += loss.item()
        if (step + 1) % opt.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if opt.logging_steps > 0 and global_step % opt.logging_steps == 0:
                # Log metrics
                results = evaluate(opt, model, val_dataset)
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / opt.logging_steps, global_step)
                logging_loss = tr_loss

            if opt.save_steps > 0 and global_step % opt.save_steps == 0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(opt.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                train_dataset.tokenizer.save_pretrained(output_dir)

                torch.save(opt, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                _rotate_checkpoints(opt, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if opt.max_steps > 0 and global_step > opt.max_steps:
            epoch_iterator.close()
            break
    if opt.max_steps > 0 and global_step > opt.max_steps:
        train_iterator.close()
        break

tb_writer.close()

logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
# Create output directory if needed
os.makedirs(opt.output_dir, exist_ok=True)


logger.info("Saving model checkpoint to %s", opt.output_dir)
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = (
    model.module if hasattr(model, "module") else model
)  # Take care of distributed/parallel training
model_to_save.save_pretrained(opt.output_dir)
train_dataset.tokenizer.save_pretrained(opt.output_dir)

# Good practice: save your training arguments together with the trained model
torch.save(opt, os.path.join(opt.output_dir, "training_args.bin"))

# Evaluation
results = {}
checkpoints = [opt.output_dir]
checkpoints = list(
    os.path.dirname(c) for c in sorted(glob.glob(opt.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
logger.info("Evaluate the following checkpoints: %s", checkpoints)
for checkpoint in checkpoints:
    global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

    model = NoRBERT.from_pretrained(checkpoint, GMVAE_model=GMVAE_model)
    model.to(opt.device)
    result = evaluate(opt, model, val_dataset, prefix=prefix)
    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    results.update(result)

