import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--fold", type=int, default=1, required=False)
# parser.add_argument("--model_name", type=str, default="allenai/longformer-large-4096", required=True)
# parser.add_argument("--model_name", type=str, default="google/bigbird-roberta-large", required=False)
parser.add_argument("--model_name", type=str, default="funnel-transformer/large", required=False)
parser.add_argument("--lr", type=float, default=2e-5, required=False)
parser.add_argument("--output", type=str, default="../model", required=False)
parser.add_argument("--log", type=str, default="../log", help="Path of the log.")
parser.add_argument("--input", type=str, default="../data/input", required=False)
parser.add_argument("--max_len", type=int, default=1600, required=False)
# parser.add_argument("--adam_bits", type=int, default=8, required=False, help="8/32 bits")
parser.add_argument("--max_steps", type=int, default=-1, required=False, help="For TrainingArguments")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", required=False)
parser.add_argument("--train_batch_size", type=int, default=4, required=False)    # 4
parser.add_argument("--num_workers", type=int, default=1, help="Number of subprocesses for data loading")
parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
parser.add_argument("--epochs", type=int, default=6, required=False)
parser.add_argument("--seed", type=int, default=42, required=False)
parser.add_argument("--fp16", type=bool, default=True, required=False)
parser.add_argument("--gradient_checkpointing", type=bool, default=True, required=False)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, required=False)

args = parser.parse_args()

print("Initial arguments", args)
