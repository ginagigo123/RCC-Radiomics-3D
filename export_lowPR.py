#%%
from collections import defaultdict
from functools import partial
import numpy as np
import re
import pandas as pd
import joblib
from glob import glob
from itertools import product
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import os

#%%

def result_stats(path):
    all_sub = dict()

    # data_raw = joblib.load(path)
    data_raw = pd.read_csv(path, index_col=0)

    # re -> replace multi space to one space
    # remove [] and strip the string
    # turn it into numpy array
    tf_float = data_raw['y_pred_prob'].apply(lambda x: re.sub(' +', ' ', x.replace('[', '').replace(']', '').strip()))
    tf_float = tf_float.apply(lambda x: [ float(val) for val in x.split(' ')])
    tf_float = tf_float.apply(lambda x: np.array(x))
    data_raw['y_pred_prob'] = tf_float
        
    # data_raw["Key"] = data_raw["case_id"]
    data_raw["all_subgroups"] = True
    # data_raw['t1a'] = data_raw['stage'] == '1a'
        
    data_raw['kits'] = data_raw['set'] == 'kits'
    data_raw['tcga'] = data_raw['set'] == 'tcga'
    data_raw['vghtc'] = data_raw['set'] == 'vghtc'

    group_name = [
        'all_subgroups',
        'kits',
        'tcga',
        'vghtc',
        # 't1a',
    ]

    for fn_k in group_name:
        ### Complete ###
        data = data_raw.copy()
        in_subgroup = data[fn_k]
        data = data[in_subgroup]
        data_complete = data.copy()
        print(fn_k, len(data_complete))

        #### Metrics ####
        res = dict()
        data = data_complete
        
        pred_prob = np.vstack(data['y_pred_prob'].values)
        
        prob_sum = pred_prob.sum(axis=1)
        if not np.allclose(1, prob_sum):
            pred_prob = pred_prob / prob_sum[:, np.newaxis]
            
        pred_prob = pred_prob[:, 1]
        
        pred_class = np.array(data['y_pred_class'].values).astype(int)
        truth_multi = np.array(data['y_truth'].values).astype(int)
        truth_bin = data['y_truth'].map({0:0, 1:1, 2:0})
        truth_bin = np.array(truth_bin).astype(int)

        average_precision = average_precision_score(truth_bin, pred_prob, pos_label=1)
        precision, recall, threshold = precision_recall_curve(truth_bin, pred_prob, pos_label=1)
        # reverse order to make recall increasing
        precision, recall, threshold = precision[::-1], recall[::-1], threshold[::-1]

        res["Pre@0.25Sen"] = np.interp(0.25, recall, precision)
        res["Pre@0.50Sen"] = np.interp(0.50, recall, precision)
        res["Pre@0.60Sen"] = np.interp(0.60, recall, precision)
        res["Pre@0.75Sen"] = np.interp(0.75, recall, precision)
        res["AP"] = average_precision
        
        res['CM'] = confusion_matrix(truth_multi, pred_class)

        res['Benign'] = (truth_multi == 0).sum()
        res['Low'] = (truth_multi == 1).sum()
        res['High'] = (truth_multi == 2).sum()
        res["Size"] = len(data)

        all_sub[fn_k] = res
    return all_sub


# %%
feature_dir = '/mnt/Internal/JNJ/HMI/gina/classification/outputs/radiomics_3D_final/co_datasets'
export_path = '/mnt/Internal/JNJ/HMI/gina/classification/outputs/radiomics_3D_final/table_lowPR'

for feature in os.listdir(feature_dir):
    all_res = dict()
    res = result_stats(os.path.join(feature_dir, feature))

    df_bin = pd.DataFrame(res).T
    df_bin.to_csv(os.path.join(export_path, feature))


# %%
