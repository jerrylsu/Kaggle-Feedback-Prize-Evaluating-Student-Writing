import copy
import os
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

target_id_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}

num_labels = len(target_id_map) - 1

id_target_map = {v: k for k, v in target_id_map.items()}


def _prepare_training_data_helper(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in tqdm(train_ids):
        filename = os.path.join(args.input, "train", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()
            text = text.replace(u'\xa0', u' ')
            text = text.rstrip()
            text = text.lstrip()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=4096,
            truncation=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


def _prepare_test_data_helper(args, tokenizer, ids):
    test_samples = []
    for idx in ids:
        filename = os.path.join(args.input, "train", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()
            text = text.replace(u'\xa0', u' ')
            text = text.rstrip()
            text = text.lstrip()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=4096,
            truncation=True,
        )

        input_ids = encoded_text["input_ids"]
        offset_mapping = encoded_text["offset_mapping"]

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        test_samples.append(sample)
    return test_samples


def prepare_test_data(df, tokenizer, args):
    test_samples = []
    ids = df["id"].unique()
    ids_splits = np.array_split(ids, 4)

    results = Parallel(n_jobs=4, backend="multiprocessing")(
        delayed(_prepare_test_data_helper)(args, tokenizer, idx) for idx in ids_splits
    )
    for result in results:
        test_samples.extend(result)

    return test_samples


class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [target_id_map[x] for x in input_labels]
        other_label_id = target_id_map["O"]
        padding_label_id = target_id_map["PAD"]
        # print(input_ids)
        # print(input_labels)

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:    # "left"
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_labels, dtype=torch.long),
        }


class FeedbackValidDataset_bk:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        id_ = self.samples[idx]["id"]
        text = self.samples[idx]["text"]
        offset_mapping = self.samples[idx]["offset_mapping"]
        input_ids = self.samples[idx]["input_ids"]
        # print(input_ids)
        # print(input_labels)

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask

        return {
            "id": id_,
            "index": idx,
            "text": text,
            "token_offset": str(offset_mapping),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class FeedbackValidDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        id = self.samples[idx]["id"]
        text = self.samples[idx]["text"]
        token_offset = self.samples[idx]["offset_mapping"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "id": id,
            'text': text,
            "token_id": input_ids,
            "token_mask": attention_mask,
            "token_offset": str(token_offset),
        }


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["id"] = [sample["id"] for sample in batch]
        output["token_offset"] = [sample["token_offset"] for sample in batch]
        output["text"] = [sample["text"] for sample in batch]

        output["token_id"] = [sample["token_id"] for sample in batch]
        output["token_mask"] = [sample["token_mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(token_id) for token_id in output["token_id"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["token_id"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["token_id"]]
            output["token_mask"] = [s + (batch_max - len(s)) * [0] for s in output["token_mask"]]
        else:
            output["token_id"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["token_id"]]
            output["token_mask"] = [(batch_max - len(s)) * [0] + s for s in output["token_mask"]]

        # convert to tensors
        output["token_id"] = torch.tensor(output["token_id"], dtype=torch.long)
        output["token_mask"] = torch.tensor(output["token_mask"], dtype=torch.long)

        return output