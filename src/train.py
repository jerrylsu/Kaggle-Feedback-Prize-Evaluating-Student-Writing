import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import warnings
import numpy as np
import pandas as pd
import torch
import math
import bitsandbytes as bnb    # https://github.com/facebookresearch/bitsandbytes
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, get_scheduler

from args import args
from dataset import prepare_training_data, target_id_map, num_labels, FeedbackDataset #, EarlyStopping
from model_bigbird import BigBirdForTokenClassification
from model_longformer import LongformerForTokenClassification
from model_deberta import DebertaForTokenClassification
from model_deberta_v2 import DebertaV2ForTokenClassification
from model_funnel import FunnelForTokenClassification

warnings.filterwarnings("ignore")


import shutil
from pathlib import Path

transformers_path = Path("/data/home/anaconda2/envs/sulei03/lib/python3.7/site-packages/transformers")
input_dir = Path("../data/input/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py', "deberta__init__.py"]:
    if str(filename).startswith("deberta"):
        filepath = deberta_v2_path/str(filename).replace("deberta", "")
    else:
        filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)
print(f"Change reberta v2 fast tokenizer successfully!!!")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    NUM_JOBS = 12  # 12
    seed_everything(args.seed)
    model_saved_path = os.path.join(args.output, args.model_name.split('/')[-1] + f'/fold{args.fold}')
    if not os.path.isdir(model_saved_path):
        os.makedirs(model_saved_path, exist_ok=True)
    # df = pd.read_csv(os.path.join(args.input, "train_small.csv"))
    df = pd.read_csv(os.path.join(args.input, "train_10folds.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    train_df.to_csv(os.path.join(args.input + f"/fold/train_fold{args.fold}.csv"))
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)
    valid_df.to_csv(os.path.join(args.input + f"/fold/valid_fold{args.fold}.csv"))

    if "v2" in args.model_name:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_name)
        print(f"Load deberta v2 tokenizer fast!!!")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS)
    valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS)

    train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer)
    valid_dataset = FeedbackDataset(valid_samples, args.max_len, tokenizer)

    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.epochs)
    print(f"Train steps: {num_train_steps}")

    config = AutoConfig.from_pretrained(args.model_name)
    config.update(
        {
            "output_hidden_states": True,
            "add_pooling_layer": False,
            "num_labels": num_labels,
            "gradient_checkpointing": True
        }
    )
    if "bigbird" in args.model_name:
        model = BigBirdForTokenClassification.from_pretrained(args.model_name, config=config)
    elif "funnel" in args.model_name:
        model = FunnelForTokenClassification.from_pretrained(args.model_name, config=config)
    elif "longformer" in args.model_name:
        model = LongformerForTokenClassification.from_pretrained(args.model_name, config=config)
    elif "deberta" in args.model_name:
        if "v2" in args.model_name:
            model = DebertaV2ForTokenClassification.from_pretrained(args.model_name, config=config)
        else:
            model = DebertaForTokenClassification.from_pretrained(args.model_name, config=config)
    else:
        raise ValueError("Please input right pretrained model name.")

    # Trianer
    print(f"Model: {args.model_name}, saved to '{model_saved_path}'")
    training_args = TrainingArguments(
        output_dir=model_saved_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # eval_steps=None,  # change evaluation_strategy to steps to use this
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.train_batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        dataloader_num_workers=args.num_workers,
        logging_dir=args.log,
        save_strategy="epoch",    # "epoch"
        logging_strategy="steps",    # "epoch"
        logging_steps=100,
        # group_by_length=True,    # This can also help speed training
        fp16=args.fp16,
        # THE ONLY CHANGE YOU NEED TO MAKE TO USE DEEPSPEED
        # deepspeed=ds_config_dict
    )

    # # Optimizer. Here is the key cell where the 8-bit Adam optimizer gets set.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    #
    # # These are the only changes you need to make. The first part sets the optimizer to use 8-bits
    # # The for loop sets embeddings to use 32-bits
    # if args.adam_bits == 32:
    #     optimizer = bnb.optim.Adam32bit(optimizer_grouped_parameters, lr=args.lr)
    # if args.adam_bits == 8:
    #     optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.lr)
    #
    #     # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.Embedding):
    #             bnb.optim.GlobalOptimManager.get_instance().register_module_override(
    #                 module, 'weight', {'optim_bits': 32}
    #             )
    #
    # num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps
    # if args.max_steps == -1 or args.max_steps is None:
    #     args.max_steps = args.epochs * num_update_steps_per_epoch
    # else:
    #     args.epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)
    #
    # if args.warmup_ratio is not None:
    #     args.num_warmup_steps = int(args.warmup_ratio * args.max_steps)
    #
    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.max_steps,
    # )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # optimizers=(optimizer, lr_scheduler)
    )

    trainer.train()
    # train_result = trainer.train()
    # metrics = train_result.metrics
    # trainer.save_model()  # Saves the tokenizer too
    # metrics["train_samples"] = len(train_dataset)
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()
