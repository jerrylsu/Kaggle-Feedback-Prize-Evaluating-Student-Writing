import torch
import numpy as np
import pandas as pd
from dataset import target_id_map, id_target_map


def compute_overlap(predict, truth):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    # Length of each and intersection
    try:
        len_truth   = len(truth)
        len_predict = len(predict)
        intersect = len(truth & predict)
        overlap1 = intersect/ len_truth
        overlap2 = intersect/ len_predict
        return overlap1, overlap2
    except:  # at least one of the input is NaN
        return 0, 0


def compute_f1_score_one(predict_df, truth_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    t_df = truth_df.loc[truth_df['discourse_type'] == discourse_type,   ['id', 'predictionstring']].reset_index(drop=True)
    p_df = predict_df.loc[predict_df['class'] == discourse_type,  ['id', 'predictionstring']].reset_index(drop=True)

    p_df.loc[:,'predict_id'] = p_df.index
    t_df.loc[:,'truth_id'] = t_df.index
    p_df.loc[:,'predictionstring'] = [set(p.split(' ')) for p in p_df['predictionstring']]
    t_df.loc[:,'predictionstring'] = [set(p.split(' ')) for p in t_df['predictionstring']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = p_df.merge(t_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_p','_t')
                          )
    overlap = [compute_overlap(*predictionstring) for predictionstring in zip(joined.predictionstring_p, joined.predictionstring_t)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(o[0] >= 0.5 and o[1] >= 0.5) for o in overlap]
    joined['max_overlap' ] = [max(*o) for o in overlap]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp\
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','truth_id'])['predict_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['predict_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids   = joined_tp['truth_id'].unique()
    unmatched_gt_ids = set(joined['truth_id'].unique()) -  set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    f1 = TP / (TP + 0.5*(FP+FN))
    return f1


def compute_lb_f1_score(predict_df, truth_df):
    f1_score = {}
    for discourse_type in truth_df.discourse_type.unique():
        f1_score[discourse_type] = compute_f1_score_one(predict_df, truth_df, discourse_type)
    # f1 = np.mean([v for v in class_scores.values()])
    return f1_score


def text_to_word(text):
    word = text.split()
    word_offset = []

    start = 0
    for w in word:
        r = text[start:].find(w)

        if r==-1:
            raise NotImplementedError
        else:
            start = start+r
            end   = start+len(w)
            word_offset.append((start,end))
            #print('%32s'%w, '%5d'%start, '%5d'%r, text[start:end])
        start = end

    return word, word_offset


def word_probability_to_predict_df(text_to_word_probability, id):
    len_word = len(text_to_word_probability)
    word_predict = text_to_word_probability.argmax(-1)
    word_score   = text_to_word_probability.max(-1)
    predict_df = []

    t = 0
    while 1:
        if word_predict[t] not in [target_id_map['O'], target_id_map['PAD']]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t+1
            if t== len_word-1: break
            continue

        t = t+1
        if t== len_word-1: break

        #----
        if id_target_map[b_marker_label][0]=='B':
            i_marker_label = b_marker_label+1
        elif id_target_map[b_marker_label][0]=='I':
            i_marker_label = b_marker_label
        else:
            raise NotImplementedError

        while 1:
            #print(t)
            if (word_predict[t] != i_marker_label) or (t ==len_word-1):
                end = t
                prediction_string = ' '.join([str(i) for i in range(start,end)]) #np.arange(start,end).tolist()
                discourse_type = id_target_map[b_marker_label][2:]
                discourse_score = word_score[start:end].tolist()
                predict_df.append((id, discourse_type, prediction_string, str(discourse_score)))
                #print(predict_df[-1])
                break
            else:
                t = t+1
                continue
        if t== len_word-1: break

    predict_df = pd.DataFrame(predict_df, columns=['id', 'class', 'predictionstring', 'score'])
    return predict_df


length_threshold = {
    'Lead'                : 9,
    'Position'            : 5,
    'Claim'               : 3,
    'Counterclaim'        : 6,
    'Rebuttal'            : 4,
    'Evidence'            : 14,
    'Concluding Statement': 11,
}
probability_threshold = {
    'Lead'                : 0.70,
    'Position'            : 0.55,
    'Claim'               : 0.55,
    'Counterclaim'        : 0.50,
    'Rebuttal'            : 0.55,
    'Evidence'            : 0.65,
    'Concluding Statement': 0.70,
}


def do_threshold(submit_df, use=['length','probability']):
    df = submit_df.copy()
    df = df.fillna('')

    if 'length' in use:
        df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
        for key, value in length_threshold.items():
            #value=3
            index = df.loc[df['class'] == key].query('l<%d'%value).index
            df.drop(index, inplace=True)

    if 'probability' in use:
        df['s'] = df.score.apply(lambda x: np.mean(eval(x)))
        for key, value in probability_threshold.items():
            index = df.loc[df['class'] == key].query('s<%f'%value).index
            df.drop(index, inplace=True)

    df = df[['id', 'class', 'predictionstring']]
    return df
