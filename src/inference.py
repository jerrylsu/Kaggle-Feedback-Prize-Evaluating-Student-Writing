import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pandas as pd
import numpy as np
import gc
import glob
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.cuda.amp as amp
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from dataset import num_labels, prepare_test_data, FeedbackValidDataset, Collate
from args import args
from model_bigbird import BigBirdForTokenClassification
from model_longformer import LongformerForTokenClassification
from model_deberta import DebertaForTokenClassification
from utils import text_to_word, compute_lb_f1_score, word_probability_to_predict_df, do_threshold


is_amp   = True
is_debug = True

# [model, model_name, checkpoint]
model_type = [
    # (BigBirdForTokenClassification, 'google/bigbird-roberta-large'),
    #(LongformerForTokenClassification, 'allenai/longformer-large-4096'),
    (DebertaForTokenClassification, 'microsoft/deberta-large'),
]

checkpoints = [
    # [
    #     '../model/bigbird-roberta-large/fold0/checkpoint-7016/pytorch_model.bin',
    # ],
    # [
         # '../model/longformer-large-4096/fold0/checkpoint-7016/pytorch_model.bin',
         # '../model/longformer-large-4096/fold1/checkpoint-8770/pytorch_model.bin',
    #     '../model/longformer-large-4096/fold3/checkpoint-7016/pytorch_model.bin',
    #],
    [
        #'../model/deberta-large/fold0/checkpoint-3508/pytorch_model.bin',
        #'../model/deberta-large/fold1/checkpoint-10524/pytorch_model.bin',
       '../model/deberta-large/fold7/checkpoint-8770/pytorch_model.bin',
    ]
]

num_net = sum([len(i) for i in checkpoints])
num_net1 = sum([len(i) for i in checkpoints[:1]])
num_net2 = sum([len(i) for i in checkpoints[:2]])

total_model = len(model_type)


def run_submit():
    if is_debug:
        # df = pd.read_csv(os.path.join(args.input, "train_10folds.csv"))
        # valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)
        valid_df = pd.read_csv('../data/input/fold/valid_fold7.csv')
        # valid_df = valid_df[:100]
        valid_id = valid_df['id'].unique()
    else:
        text_dir = '../input/feedback-prize-2021/test'
        valid_id = [f.split('/')[-1][:-4] for f in glob.glob(text_dir + '/*.txt')]
        valid_id = sorted(valid_id)
    num_valid = len(valid_id)
    print('len(valid_id)', len(valid_id))

    df_text = []
    for id in valid_id:
        text_file = f'../data/input/train/{id}.txt'
        with open(text_file, 'r') as f:
            text = f.read()

        text = text.replace(u'\xa0', u' ')
        text = text.rstrip()
        text = text.lstrip()
        df_text.append((id, text))
    df_text = pd.DataFrame(df_text, columns=['id', 'text'])
    df_text['text_len'] = df_text['text'].apply(lambda x: len(x))
    df_text = df_text.sort_values('text_len').reset_index(drop=True)
    del df_text['text_len']
    print('df_text.shape', df_text.shape)


    results = []
    for model_num in range(total_model):              #  all model type!
        Model, model_name = model_type[model_num]
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": False,
                "num_labels": num_labels,
                "gradient_checkpointing": True
            }
        )
        model = Model.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        valid_samples = prepare_test_data(df_text, tokenizer, args)
        collate = Collate(tokenizer=tokenizer)
        valid_dataset = FeedbackValidDataset(valid_samples, args.max_len, tokenizer)
        valid_dataloader  = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size  = 8, # 4, #
            drop_last   = False,
            num_workers = 2, # 0, #
            pin_memory  = False,
            collate_fn = collate,
        )

        for n in range(len(checkpoints[model_num])):    # all checkpoint for one model type!
            model.load_state_dict(torch.load(checkpoints[model_num][n], map_location='cuda:0'))
            model.cuda()
            print(f"Load OK: [{n}] '{model_name}' from {checkpoints[model_num][n]}")
            # start here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            results_n = {'id':[], 'token_mask':[], 'token_offset':[], 'probability':[],}
            T = 0
            for t, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
                batch_size = len(batch['id'])
                token_mask = batch['token_mask']
                token_id = batch['token_id']
                token_mask = token_mask.cuda()
                token_id = token_id.cuda()

                model.eval()
                with torch.no_grad():
                    with amp.autocast(enabled=is_amp):
                        # (8, 1600, 15) (batch_size, max_len, num_labels)
                        probability = data_parallel(model, (token_id, token_mask))
                        # probability = net[n](token_id, token_mask)
                        pp = (probability[0] * 255).byte().data.cpu().numpy()
                        if pp.shape[1] > args.max_len:
                            pp = pp[:, :args.max_len, :]
                        else:
                            pp = np.pad(pp, ((0, 0), (0, args.max_len - pp.shape[1]), (0, 0)), 'constant', constant_values=0)
                        # probability = 1
                        # pp = np.random.randint(0,255,size=[len(batch['token_offset']), max_length, 15]).astype('int8')
                        results_n['probability'].append(pp)
                        if n == 0:
                            results_n['token_offset'] += [eval(x) for x in batch['token_offset']]
                        T += batch_size

            # ----------------------------
            torch.cuda.empty_cache()
            print('')
            if n == 0:
                results.append({
                    'probability': np.concatenate(results_n['probability']),
                    'token_offset': np.array(results_n['token_offset'], object)
                })
            else:
                results.append({
                    'probability': np.concatenate(results_n['probability']),
                })

            del probability, pp, results_n
            gc.collect()
            print()
            # ------------------------------------------------------------------------
        del model, valid_samples, Model, tokenizer
        gc.collect()
        print()

    submit_df = []
    for i in range(num_valid):
        d = df_text.iloc[i]
        id = d.id
        text = d.text
        word, word_offset = text_to_word(text)
        token_to_text_probability = np.full((len(text), num_labels), 0, np.float32)
        for j in range(num_net):
            p = results[j]['probability'][i][1:] / 255
            if j < num_net1:
                for t, (start, end) in enumerate(results[0]['token_offset'][i]):
                    if t == args.max_len - 1: break  # assume max_length, else use token_mask to get length
                    token_to_text_probability[start:end] += p[t]  # **0.5
            elif j < num_net2:
                for t, (start, end) in enumerate(results[num_net1]['token_offset'][i]):
                    if t == args.max_len - 1: break  # assume max_length, else use token_mask to get length
                    token_to_text_probability[start:end] += p[t]  # **0.5
            else:
                for t, (start, end) in enumerate(results[num_net2]['token_offset'][i]):
                    if t == args.max_len - 1: break  # assume max_length, else use token_mask to get length
                    token_to_text_probability[start:end] += p[t]  # **0.5

        token_to_text_probability = token_to_text_probability / num_net

        text_to_word_probability = np.full((len(word), num_labels), 0, np.float32)
        for t, (start, end) in enumerate(word_offset):
            text_to_word_probability[t] = token_to_text_probability[start:end].mean(0)

        predict_df = word_probability_to_predict_df(text_to_word_probability, id)
        submit_df.append(predict_df)
    print('')

    # ----------------------------------------
    submit_df = pd.concat(submit_df).reset_index(drop=True)
    submit_df = do_threshold(submit_df, use=['length', 'probability'])
    submit_df.to_csv('submission.csv', index=False)

    print('----')
    print(submit_df.head())
    print('submission ok!----')
    if is_debug:
        f1_score = compute_lb_f1_score(submit_df, valid_df)
        print('f1 macro : %f\n' % np.mean([v for v in f1_score.values()]))
        for k, v in f1_score.items():
            print('%20s : %05f' % (k, v))


if __name__ == "__main__":
    run_submit()
