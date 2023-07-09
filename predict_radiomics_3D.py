#%%
import SimpleITK as sitk
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt
import monai
from monai.data import DataLoader, Dataset, NumpyReader
from monai.transforms import LoadImaged, Compose, Resized, EnsureChannelFirstd, ScaleIntensityd, MapLabelValued, IntensityStatsd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from glob import glob
import os
from os.path import join, exists
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from encoder3d import DeepProfiler
import argparse
import joblib
import regex
from util import Result, draw_result
import torch.nn as nn

label_type_options = ['t1a','lowhigh','malignant', 'total']
parser = argparse.ArgumentParser(description='CNN')

parser.add_argument('--batch_size','--bz', type=int, default=32,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs','-e', type=int, default=20,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--use_mask','-m', action='store_true', default=False,
                    help='train with mask')
parser.add_argument('--label_type', '-l',type=str, default='malignant',
                        choices=label_type_options)

args = parser.parse_args()

if args.label_type == 'total':
    num_classes = 3
    class_list = [0,1,2]
else:
    num_classes = 2
    class_list = [0,1]

# load fixed fold index
train_set = []
val_set = []
test_set = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_idx = []
val_idx = []
test_idx = []

radiomics_3d_feature_list = [
    'original_glcm_Autocorrelation',
    'original_glcm_ClusterProminence',
    'original_glcm_ClusterShade',
    'original_glcm_ClusterTendency',
    'original_glcm_Contrast',
    'original_glcm_Correlation',
    'original_glcm_DifferenceAverage',
    'original_glcm_DifferenceEntropy',
    'original_glcm_DifferenceVariance',
    'original_glcm_Id',
    'original_glcm_Idm',
    'original_glcm_Idmn',
    'original_glcm_Idn',
    'original_glcm_Imc1',
    'original_glcm_Imc2',
    'original_glcm_InverseVariance',
    'original_glcm_JointAverage',
    'original_glcm_JointEnergy',
    'original_glcm_JointEntropy',
    'original_glcm_MCC',
    'original_glcm_MaximumProbability',
    'original_glcm_SumAverage',
    'original_glcm_SumEntropy',
    'original_glcm_SumSquares',
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_HighGrayLevelRunEmphasis',
    'original_glrlm_LongRunEmphasis',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_LowGrayLevelRunEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_RunLengthNonUniformityNormalized',
    'original_glrlm_RunPercentage',
    'original_glrlm_RunVariance',
    'original_glrlm_ShortRunEmphasis',
    'original_glrlm_ShortRunHighGrayLevelEmphasis',
    'original_glrlm_ShortRunLowGrayLevelEmphasis',
    'original_firstorder_10Percentile',
    'original_firstorder_90Percentile',
    'original_firstorder_Energy',
    'original_firstorder_Entropy',
    'original_firstorder_InterquartileRange',
    'original_firstorder_Kurtosis',
    'original_firstorder_Maximum',
    'original_firstorder_MeanAbsoluteDeviation',
    'original_firstorder_Mean',
    'original_firstorder_Median',
    'original_firstorder_Minimum',
    'original_firstorder_Range',
    'original_firstorder_RobustMeanAbsoluteDeviation',
    'original_firstorder_RootMeanSquared',
    'original_firstorder_Skewness',
    'original_firstorder_TotalEnergy',
    'original_firstorder_Uniformity',
    'original_firstorder_Variance',
    'log-sigma-3-0-mm-3D_glcm_Autocorrelation',
    'log-sigma-3-0-mm-3D_glcm_ClusterProminence',
    'log-sigma-3-0-mm-3D_glcm_ClusterShade',
    'log-sigma-3-0-mm-3D_glcm_ClusterTendency',
    'log-sigma-3-0-mm-3D_glcm_Contrast',
    'log-sigma-3-0-mm-3D_glcm_Correlation',
    'log-sigma-3-0-mm-3D_glcm_DifferenceAverage',
    'log-sigma-3-0-mm-3D_glcm_DifferenceEntropy',
    'log-sigma-3-0-mm-3D_glcm_DifferenceVariance',
    'log-sigma-3-0-mm-3D_glcm_Id',
    'log-sigma-3-0-mm-3D_glcm_Idm',
    'log-sigma-3-0-mm-3D_glcm_Idmn',
    'log-sigma-3-0-mm-3D_glcm_Idn',
    'log-sigma-3-0-mm-3D_glcm_Imc1',
    'log-sigma-3-0-mm-3D_glcm_Imc2',
    'log-sigma-3-0-mm-3D_glcm_InverseVariance',
    'log-sigma-3-0-mm-3D_glcm_JointAverage',
    'log-sigma-3-0-mm-3D_glcm_JointEnergy',
    'log-sigma-3-0-mm-3D_glcm_JointEntropy',
    'log-sigma-3-0-mm-3D_glcm_MCC',
    'log-sigma-3-0-mm-3D_glcm_MaximumProbability',
    'log-sigma-3-0-mm-3D_glcm_SumAverage',
    'log-sigma-3-0-mm-3D_glcm_SumEntropy',
    'log-sigma-3-0-mm-3D_glcm_SumSquares',
    'log-sigma-3-0-mm-3D_glrlm_GrayLevelNonUniformity',
    'log-sigma-3-0-mm-3D_glrlm_GrayLevelNonUniformityNormalized',
    'log-sigma-3-0-mm-3D_glrlm_GrayLevelVariance',
    'log-sigma-3-0-mm-3D_glrlm_HighGrayLevelRunEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_LongRunEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_LongRunHighGrayLevelEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_LongRunLowGrayLevelEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_LowGrayLevelRunEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_RunEntropy',
    'log-sigma-3-0-mm-3D_glrlm_RunLengthNonUniformity',
    'log-sigma-3-0-mm-3D_glrlm_RunLengthNonUniformityNormalized',
    'log-sigma-3-0-mm-3D_glrlm_RunPercentage',
    'log-sigma-3-0-mm-3D_glrlm_RunVariance',
    'log-sigma-3-0-mm-3D_glrlm_ShortRunEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_ShortRunHighGrayLevelEmphasis',
    'log-sigma-3-0-mm-3D_glrlm_ShortRunLowGrayLevelEmphasis',
    'wavelet-LH_glcm_Autocorrelation',
    'wavelet-LH_glcm_ClusterProminence',
    'wavelet-LH_glcm_ClusterShade',
    'wavelet-LH_glcm_ClusterTendency',
    'wavelet-LH_glcm_Contrast',
    'wavelet-LH_glcm_Correlation',
    'wavelet-LH_glcm_DifferenceAverage',
    'wavelet-LH_glcm_DifferenceEntropy',
    'wavelet-LH_glcm_DifferenceVariance',
    'wavelet-LH_glcm_Id',
    'wavelet-LH_glcm_Idm',
    'wavelet-LH_glcm_Idmn',
    'wavelet-LH_glcm_Idn',
    'wavelet-LH_glcm_Imc1',
    'wavelet-LH_glcm_Imc2',
    'wavelet-LH_glcm_InverseVariance',
    'wavelet-LH_glcm_JointAverage',
    'wavelet-LH_glcm_JointEnergy',
    'wavelet-LH_glcm_JointEntropy',
    'wavelet-LH_glcm_MCC',
    'wavelet-LH_glcm_MaximumProbability',
    'wavelet-LH_glcm_SumAverage',
    'wavelet-LH_glcm_SumEntropy',
    'wavelet-LH_glcm_SumSquares',
    'wavelet-LH_glrlm_GrayLevelNonUniformity',
    'wavelet-LH_glrlm_GrayLevelNonUniformityNormalized',
    'wavelet-LH_glrlm_GrayLevelVariance',
    'wavelet-LH_glrlm_HighGrayLevelRunEmphasis',
    'wavelet-LH_glrlm_LongRunEmphasis',
    'wavelet-LH_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-LH_glrlm_LongRunLowGrayLevelEmphasis',
    'wavelet-LH_glrlm_LowGrayLevelRunEmphasis',
    'wavelet-LH_glrlm_RunEntropy',
    'wavelet-LH_glrlm_RunLengthNonUniformity',
    'wavelet-LH_glrlm_RunLengthNonUniformityNormalized',
    'wavelet-LH_glrlm_RunPercentage',
    'wavelet-LH_glrlm_RunVariance',
    'wavelet-LH_glrlm_ShortRunEmphasis',
    'wavelet-LH_glrlm_ShortRunHighGrayLevelEmphasis',
    'wavelet-LH_glrlm_ShortRunLowGrayLevelEmphasis',
    'wavelet-HL_glcm_Autocorrelation',
    'wavelet-HL_glcm_ClusterProminence',
    'wavelet-HL_glcm_ClusterShade',
    'wavelet-HL_glcm_ClusterTendency',
    'wavelet-HL_glcm_Contrast',
    'wavelet-HL_glcm_Correlation',
    'wavelet-HL_glcm_DifferenceAverage',
    'wavelet-HL_glcm_DifferenceEntropy',
    'wavelet-HL_glcm_DifferenceVariance',
    'wavelet-HL_glcm_Id',
    'wavelet-HL_glcm_Idm',
    'wavelet-HL_glcm_Idmn',
    'wavelet-HL_glcm_Idn',
    'wavelet-HL_glcm_Imc1',
    'wavelet-HL_glcm_Imc2',
    'wavelet-HL_glcm_InverseVariance',
    'wavelet-HL_glcm_JointAverage',
    'wavelet-HL_glcm_JointEnergy',
    'wavelet-HL_glcm_JointEntropy',
    'wavelet-HL_glcm_MCC',
    'wavelet-HL_glcm_MaximumProbability',
    'wavelet-HL_glcm_SumAverage',
    'wavelet-HL_glcm_SumEntropy',
    'wavelet-HL_glcm_SumSquares',
    'wavelet-HL_glrlm_GrayLevelNonUniformity',
    'wavelet-HL_glrlm_GrayLevelNonUniformityNormalized',
    'wavelet-HL_glrlm_GrayLevelVariance',
    'wavelet-HL_glrlm_HighGrayLevelRunEmphasis',
    'wavelet-HL_glrlm_LongRunEmphasis',
    'wavelet-HL_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-HL_glrlm_LongRunLowGrayLevelEmphasis',
    'wavelet-HL_glrlm_LowGrayLevelRunEmphasis',
    'wavelet-HL_glrlm_RunEntropy',
    'wavelet-HL_glrlm_RunLengthNonUniformity',
    'wavelet-HL_glrlm_RunLengthNonUniformityNormalized',
    'wavelet-HL_glrlm_RunPercentage',
    'wavelet-HL_glrlm_RunVariance',
    'wavelet-HL_glrlm_ShortRunEmphasis',
    'wavelet-HL_glrlm_ShortRunHighGrayLevelEmphasis',
    'wavelet-HL_glrlm_ShortRunLowGrayLevelEmphasis',
    'wavelet-HH_glcm_Autocorrelation',
    'wavelet-HH_glcm_ClusterProminence',
    'wavelet-HH_glcm_ClusterShade',
    'wavelet-HH_glcm_ClusterTendency',
    'wavelet-HH_glcm_Contrast',
    'wavelet-HH_glcm_Correlation',
    'wavelet-HH_glcm_DifferenceAverage',
    'wavelet-HH_glcm_DifferenceEntropy',
    'wavelet-HH_glcm_DifferenceVariance',
    'wavelet-HH_glcm_Id',
    'wavelet-HH_glcm_Idm',
    'wavelet-HH_glcm_Idmn',
    'wavelet-HH_glcm_Idn',
    'wavelet-HH_glcm_Imc1',
    'wavelet-HH_glcm_Imc2',
    'wavelet-HH_glcm_InverseVariance',
    'wavelet-HH_glcm_JointAverage',
    'wavelet-HH_glcm_JointEnergy',
    'wavelet-HH_glcm_JointEntropy',
    'wavelet-HH_glcm_MCC',
    'wavelet-HH_glcm_MaximumProbability',
    'wavelet-HH_glcm_SumAverage',
    'wavelet-HH_glcm_SumEntropy',
    'wavelet-HH_glcm_SumSquares',
    'wavelet-HH_glrlm_GrayLevelNonUniformity',
    'wavelet-HH_glrlm_GrayLevelNonUniformityNormalized',
    'wavelet-HH_glrlm_GrayLevelVariance',
    'wavelet-HH_glrlm_HighGrayLevelRunEmphasis',
    'wavelet-HH_glrlm_LongRunEmphasis',
    'wavelet-HH_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-HH_glrlm_LongRunLowGrayLevelEmphasis',
    'wavelet-HH_glrlm_LowGrayLevelRunEmphasis',
    'wavelet-HH_glrlm_RunEntropy',
    'wavelet-HH_glrlm_RunLengthNonUniformity',
    'wavelet-HH_glrlm_RunLengthNonUniformityNormalized',
    'wavelet-HH_glrlm_RunPercentage',
    'wavelet-HH_glrlm_RunVariance',
    'wavelet-HH_glrlm_ShortRunEmphasis',
    'wavelet-HH_glrlm_ShortRunHighGrayLevelEmphasis',
    'wavelet-HH_glrlm_ShortRunLowGrayLevelEmphasis',
    ]

# seg_max_max_npy_folder = 'inputs/cc3d/vghtc_manual_seg_img_crop_128'
f_name = 'co_datasets'

def load_kits_data(label_type, feature_name):
    kits_record_raw = pd.read_json('inputs/kits.json', orient='records')
    kits_record_raw = kits_record_raw[['case_id', 'pathology_t_stage', 'malignant', 'tumor_isup_grade', 'voxel_spacing']]

    ct_path = "inputs/radiomics_3D"

    datalist = []

    for _idx, row in kits_record_raw.iterrows():
        uid = row['case_id']
        x = f"{ct_path}/{uid}_{feature_name}.npy"
        if label_type == 'malignant':
            # 0 -> Benign, 1 -> Malignant
            y = int(row['malignant'])
        elif label_type == 'total':
            # 0 -> Benign, 1 -> Low, 2 -> High
            if row['malignant'] == True:
                y = int(row['tumor_isup_grade'] > 2) + 1 
            else:
                y = 0
        else:
            # 0 -> Low, 1 -> High
            y = int(row['tumor_isup_grade'] > 2)
            
        # some rows did not have tumor_isup_grade label -> skip this row
        if np.isnan(y):
            print('---- kits has nan y:', y)
            continue

        # check whether the feature exists
        if os.path.exists(x) == False:
            continue
        datalist.append({"image": x, "label": y, 'set': 'kits'})

    return datalist

# for each feature in radiomics
for feature_name in radiomics_3d_feature_list:
    # load data from diff datasets
    kits_set = load_kits_data(args.label_type, feature_name)

    total_set.extend(kits_set)
    total_set = pd.DataFrame(total_set)

    # split the data into 5 fold
    for train_idx, test_idx in kf.split(total_set):
        train_data = total_set.iloc[train_idx]
        test_data = total_set.iloc[test_idx]
        train_data, val_data = train_test_split(train_data, random_state=42, test_size=0.2)

        train_set.append(train_data)
        val_set.append(val_data)
        test_set.append(test_data)
    
    print('1 fold : vghtc train:', len(train_set[0]), ', val:', len(val_set[0]), ', test:', len(test_set[0]))
        
    fold_history = {
        'Fold': [],
        'Name': [],
        'Accuracy': [],
        'AUC': [],
        'CM': [],
        'F1 Score': []
    }

    n_split=5
    result = Result()
    for fold_idx in range(n_split):
        x_train = []
        x_val = []

        train, val, test = train_set[fold_idx], val_set[fold_idx], test_set[fold_idx]
        train, val, test = pd.DataFrame(train), pd.DataFrame(val), pd.DataFrame(test)
        
        trainlist = []
        for x, y, z in zip(train['image'], train['label'], train['set']):
            trainlist.append({"image": x, "label": y, "set": z})
        vallist = []
        for x, y, z in zip(val['image'], val['label'], val['set']):
            vallist.append({"image": x, "label": y, "set": z})
        testlist = []
        for x, y, z in zip(test['image'], test['label'], test['set']):
            testlist.append({"image": x, "label": y, "set": z})
        
        print('after codataset:',len(trainlist), len(vallist), len(testlist))

        # define transforms
        ct_transform = Compose(
            [
                #LoadImaged(keys=["image", "mask"], reader="ITKReader"),
                LoadImaged(keys=["image"], reader="NumpyReader"),
                # ScaleIntensityd(
                #     keys=["image"], minv=-1.0, maxv=1.0, 
                # ),
                # IntensityStatsd(keys=["image"], ops=[lambda x: np.mean(x), 'max'], key_prefix='orig'),
                EnsureChannelFirstd(keys=["image"]),
                Resized(keys=["image"], spatial_size=(64, 64, 64), mode='nearest')
            ]
        )

        train_loader = DataLoader(
            Dataset(
                data=trainlist,
                transform=ct_transform
        ),
            batch_size=3,
            num_workers=1
        )

        val_loader = DataLoader(
            Dataset(
                data=vallist,
                transform=ct_transform
            ),
            batch_size=3,
            num_workers=1,
        )
        test_loader = DataLoader(
            Dataset(
                data=testlist,
                transform=ct_transform
            ),
            batch_size=3,
            num_workers=1,
        )

        print("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:2")

        # # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        # model = monai.networks.nets.DenseNet121(
        #    spatial_dims=3,
        #    in_channels=1,
        #    out_channels=2
        # ).to(device)
        
        print('num_classes:', num_classes)
        model = DeepProfiler(n_classes=num_classes, use_mask=False)
        model.to(device)

        model_name='radiomics_3D_final'
        model_folder = f"model/{model_name}/{f_name}/"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f"The new directory {model_folder} is created!")
        output_folder = f"outputs/{model_name}/{f_name}/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"The new directory {output_folder} is created!")
        loss_function = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        train_mode = True
        if train_mode:
            val_interval = 1
            best_metric = 0
            best_metric_epoch = -1
            epoch_loss_values = []
            epoch_val_loss_values = []

            epoch_acc_list = []

            acc_values = []
            auc_values = []
            writer = SummaryWriter()
            max_epochs = args.epochs
            train_ds = len(trainlist)
            softmax = nn.Softmax(dim=1)

            for epoch in range(max_epochs):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{max_epochs}")
                model.train()
                epoch_loss = 0
                running_corrects = 0
                total = 0
                step = 0
                prob_all = []
                pred_all = []
                 
                label_all = []
                progress_bar = tqdm(train_loader)
                for batch_data in progress_bar:
                    progress_bar.set_description('Epoch ' + str(epoch))
                    step += 1
                    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device) # , masks, radiomics 
                    optimizer.zero_grad()
                    if args.use_mask:
                        outputs, out_radiomics = model(inputs)
                    else:
                        outputs, out_radiomics = model(inputs)

                    preds = torch.max(outputs.data, 1)[1]
                    if args.label_type == 'total':
                        prob_all.extend(softmax(outputs.cpu().detach()).numpy())
                    else:
                        prob_all.extend(outputs[:,1].cpu().detach().numpy())

                    pred_all.extend(preds.cpu())
                    label_all.extend(labels.cpu())

                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_len = train_ds // train_loader.batch_size
                    running_corrects += torch.sum(preds == labels.data)
                    total += labels.size(0)

                    progress_bar.set_postfix(
                        train_loss='%.4f' % ( epoch_loss/step ),
                        acc='%.4f' % (running_corrects/total))
                    writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)

                epoch_acc = running_corrects / len(trainlist)
                epoch_acc_list.append(epoch_acc)
                cf_matrix = confusion_matrix(label_all, pred_all,labels=class_list)
                if args.label_type == 'total':
                    try:
                        roc_auc = roc_auc_score(label_all, prob_all,labels=class_list,multi_class='ovr')
                    except:
                            roc_auc = np.nan
                else:
                    roc_auc = roc_auc_score(label_all, prob_all,labels=class_list)
                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, acc: {epoch_acc: .4f}, AUC: {roc_auc: .4f}")
                print('confusion matrix:\n', cf_matrix)

                if (epoch + 1) % val_interval == 0:
                    model.eval()

                    num_correct = 0.0
                    metric_count = 0
                    step = 0
                    prob_all = []
                    pred_all = []
                    label_all = []
                    epoch_val_loss = 0

                    for val_data in val_loader:
                        step += 1
                        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    
                        with torch.no_grad():
                            if args.use_mask:
                                val_outputs, pre_radiomics = model(val_images, val_masks)
                            else:
                                val_outputs, pre_radiomics = model(val_images)
                            preds = torch.max(val_outputs.data, 1)[1]

                            loss = loss_function(val_outputs, val_labels)
                            epoch_val_loss += loss.item()

                            # for auc used
                            if args.label_type == 'total':
                                #print(softmax(outputs.cpu().detach()))
                                prob_all.extend(softmax(val_outputs.cpu().detach()).numpy())
                            else:
                                prob_all.extend(val_outputs[:,1].cpu().detach().numpy())
                                # print(outputs[:,:].cpu().detach().numpy())
                            label_all.extend(val_labels.cpu())
                            pred_all.extend(preds.cpu())

                            assert np.max(val_outputs.argmax(dim=1))<num_classes
                            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                            # print('outputs:', val_outputs.argmax(dim=1), 'labels:', val_labels, 'value:', value)
                            metric_count += len(value)
                            num_correct += value.sum().item()

                    epoch_val_loss /= step
                    epoch_val_loss_values.append(epoch_val_loss)

                    acc = num_correct / metric_count
                    acc_values.append(acc)

                    if args.label_type == 'total':
                        try:
                            auc = roc_auc_score(label_all, prob_all,labels=class_list,multi_class='ovr')
                        except:
                            auc = np.nan
                    else:
                        print('size: ', len(label_all), len(prob_all))
                        auc = roc_auc_score(label_all, prob_all, labels=class_list)
                    auc_values.append(auc)
                    cf_matrix = confusion_matrix(label_all, pred_all,labels=class_list)
                    
                    if auc > best_metric or best_metric ==0:
                        best_metric = auc
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), f"model/{model_name}/{f_name}/{feature_name}_{args.label_type}_fold_{fold_idx}.pth")
                        print("saved new best metric model")

                    print(f"Current epoch: {epoch+1} accuracy: {acc:.4f}, AUC: {auc: .4f}")
                    print('confusion matrix:\n', cf_matrix)
                    print(f"Best AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
                    writer.add_scalar("val_accuracy", acc, epoch + 1)
                # end of validation
                    
            print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            writer.close()

        model.load_state_dict(torch.load(f"model/{model_name}/{f_name}/{feature_name}_{args.label_type}_fold_{fold_idx}.pth"))
        model.eval()

        result.evaluation_v2(model, test_loader, device, False, True, class_list=class_list)

        print('fold', fold_idx, ':\n', result.fold_history)
    # end of 5 fold
    pd.DataFrame(result.export_dict_v2).to_csv(f'outputs/{model_name}/{f_name}/{feature_name}_{args.label_type}.csv')
    print(result.export_dict_v2)
