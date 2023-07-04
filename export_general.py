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
def calc_sen_spe(cm):
    # Calculate sensitivity (recall) for each class
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)

    # Calculate specificity (true negative rate) for each class
    specificity = np.diag(cm) / np.sum(cm, axis=0)

    # Calculate weighted average sensitivity and specificity
    weights = np.sum(cm, axis=1) / np.sum(cm)
    weighted_sensitivity = np.sum(sensitivity * weights)
    weighted_specificity = np.sum(specificity * weights)
    return weighted_sensitivity, weighted_specificity

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
        
        pred_class = np.array(data['y_pred_class'].values).astype(int)
        truth = np.array(data['y_truth'].values).astype(int)

        cm = confusion_matrix(truth, pred_class)
        print(cm)
        sen, spe = calc_sen_spe(cm)

        res["Accuracy"] = accuracy_score(truth, pred_class)
        res['AUC'] = roc_auc_score(truth, pred_prob, average='weighted', multi_class='ovo', labels=[0, 1, 2])
        res['F1-Score'] = f1_score(truth, pred_class, average='weighted')
        res['Sensitivity'] = sen
        res['Specificity'] = spe
        
        res['CM'] = confusion_matrix(truth, pred_class)

        res['Benign'] = (truth == 0).sum()
        res['Low'] = (truth == 1).sum()
        res['High'] = (truth == 2).sum()
        res["Size"] = len(data)

        all_sub[fn_k] = res
    return all_sub


# %%

feature_dir = '/mnt/Internal/JNJ/HMI/gina/classification/outputs/radiomics_3D_final/co_datasets'
export_path = '/mnt/Internal/JNJ/HMI/gina/classification/outputs/radiomics_3D_final/table_general'

all_res = dict()
for feature in os.listdir(feature_dir):
    # table format
    res = result_stats(os.path.join(feature_dir, feature))

    # export
    df_bin = pd.DataFrame(res).T
    df_bin.to_csv(os.path.join(export_path, feature))

    # for sorting
    all_res[feature] = df_bin

# %%

top_5 = sorted(all_res.items(), key=lambda x: max(x[1]['F1-Score']))[-5:]
print(top_5)
