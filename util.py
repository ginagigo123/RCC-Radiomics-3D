from scipy import ndimage
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
def resize_volume(img, desired_depth=64, desired_width=128, desired_height=128):
    """Resize across z-axis"""
    
    # Get current depth
    current_width = img.shape[0]
    current_height = img.shape[1]
    current_depth = img.shape[-1]
    
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def read_fold(path):
    data = pd.read_pickle(path)
    # print(data)
    data_dict = {
        'case_id': [],
        'nii_dir': [],
        'label': []
    }

    data_dict['case_id'] = [ _['case_id'] for _ in data]
    data_dict['nii_dir'] = [ _['dir'] for _ in data]
    data_dict['label'] = [ _['label'] for _ in data]

    data = pd.DataFrame(data_dict)

    return data

class Result:
    def __init__(self) -> None:
        self.export_dict = {
            'Name': [],
            'Accuracy': [],
            'AUC': [],
            'F1 Score': [],
            'CM': [],
            'Sensitivity': [],
            'Specificity': []
        }
        self.export_dict_v2 = {
            'set': [],
            'y_pred_prob': [],
            'y_pred_class': [],
            'y_truth': []
        }

        self.fold_history = {
            'Fold': [],
            'Name': [],
            'Accuracy': [],
            'AUC': [],
            'CM': [],
            'F1 Score': []
        }
        self.prob_5fold = {
            'val':[],
            'test':[]
        }
        self.label_5fold = {
            'val':[],
            'test':[]
        }

    def detail_info(self, cm1):
        try:   
            sensitivity1 = cm1[1][1] / (cm1[1][0] + cm1[1][1])
        except:
            sensitivity1 = 0
        print('Sensitivity : ', sensitivity1 )
        
        try:
            specificity1 = cm1[0][0] / (cm1[0][0] + cm1[0][1])
        except:
            specificity1 = 0
        print('Specificity : ', specificity1)

        try:
            precision = cm1[1][1] / (cm1[1][1] + cm1[1][0])
        except:
            precision = 0
        try:
            recall = cm1[1][1] / (cm1[1][1] + cm1[0][1])
        except:
            recall = 0
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except:
            f1_score = 0
        print('F1 Score : ', f1_score)
        return sensitivity1, specificity1, f1_score

    def combine_cf_matrix(self, start_idx, name,class_list=[0,1]):
        array = self.fold_history
        cf_array = array['CM']
        interval = 1
        interval_index = [i for i, value in enumerate(array['Name']) if value == name]
        if len(interval_index) > 1:
            interval = interval_index[1] - interval_index[0]
        con_result = cf_array[start_idx].copy()
        avg_acc = 0
        avg_auc = 0
        iter_count = 0
        if len(con_result) == 1 and len(con_result[0]) == 1:
            con_result = [[0, 0], [0, cf_array[start_idx][0][0]]]
        if len(class_list)==2:
            total_auc = roc_auc_score(self.label_5fold[name], self.prob_5fold[name])
        else:
            total_auc = roc_auc_score(self.label_5fold[name], self.prob_5fold[name],labels=class_list,multi_class='ovo')
        for i in range(start_idx + interval, len(cf_array), interval):
            iter_count += 1
            avg_acc += array['Accuracy'][i]
            avg_auc += array['AUC'][i]
            if len(cf_array[i]) == 1 and len(cf_array[i][0]) == 1:
                con_result[1][1] += cf_array[i][0][0]
                continue
            for row_idx in range(len(class_list)):
                for col_idx in range(len(class_list)):
                    con_result[row_idx][col_idx] += cf_array[i][row_idx][col_idx]
        if len(class_list)==2:
            sensitivity1, specificity1, f1_score = self.detail_info(con_result)
        else:
            sensitivity1 = []
            specificity1 = []
            f1_score = []
            for class_idx in class_list:
                k = con_result.copy()
                if class_idx == 0:
                    k[:,[0,2]] = k[:,[2,0]]
                    k[[2,0],:] = k[[0,2],:]
                elif class_idx == 1:
                    k[:,[1,2]] = k[:,[2,1]]
                    k[[2,1],:] = k[[1,2],:]
                elif class_idx == 2:
                    pass
                else:
                    raise Exception("Error")
                con_result_ = np.array([[k[0][0]+k[0][1]+k[1][0]+k[1][1],k[0][2]+k[1][2]],[k[2][0]+k[2][1],k[2][2]]])
                sensitivity1_byclass, specificity1_byclass, f1_score_byclass = self.detail_info(con_result_)
                sensitivity1.append(sensitivity1_byclass)
                specificity1.append(specificity1_byclass)
                f1_score.append(f1_score_byclass)
        self.export_dict['Name'].append(name)
        self.export_dict['Accuracy'].append(avg_acc/iter_count)
        #self.export_dict['AUC'].append(avg_auc/iter_count)
        self.export_dict['AUC'].append(total_auc)
        self.export_dict['CM'].append(con_result)
        self.export_dict['F1 Score'].append(f1_score)
        self.export_dict['Sensitivity'].append(sensitivity1)
        self.export_dict['Specificity'].append(specificity1)

        return con_result
    def evaluation(self, model, data_loader, fold_idx, name, device , use_mask, is_radiomics=False,class_list=[0,1]):
        num_correct = 0.0
        metric_count = 0
        prob_all = []
        pred_all = []
        label_all = [] 
        label_all_byclass = []
        pred_all_byclass = []
        acc_all_byclass = []
        softmax = nn.Softmax(dim=1)
        for val_data in data_loader:
            #val_images, val_labels, val_masks = val_data["image"].to(device), val_data["label"].to(device), val_data["mask"].to(device)
            if use_mask:
                val_images, val_labels, val_masks = val_data["image"].to(device), val_data["label"].to(device), val_data["mask"].to(device)
            else:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            with torch.no_grad():   

                if use_mask:
                    if is_radiomics:
                        val_outputs, val_radiomics = model(val_images,val_masks)
                    else:    
                        val_outputs = model(val_images,val_masks)
                else:
                    if is_radiomics:
                        val_outputs, val_radiomics = model(val_images)
                    else:    
                        val_outputs = model(val_images)
                
                preds = torch.max(val_outputs.data, 1)[1]   

                # for auc used
                if len(class_list)==3:
                    prob_all.extend(softmax(val_outputs.cpu().detach()).numpy())
                    
                else:
                    prob_all.extend(val_outputs[:,1].cpu().detach().numpy())
                #prob_all.extend(val_outputs[:,1].cpu().detach().numpy())
                label_all.extend(val_labels.cpu())
                pred_all.extend(preds.cpu())
                if len(class_list)==3:
                    self.prob_5fold[name].extend(softmax(val_outputs.cpu().detach()).numpy())
                else:
                    self.prob_5fold[name].extend(val_outputs[:,1].cpu().detach().numpy())
                #self.prob_5fold[name].extend(val_outputs[:,1].cpu().detach().numpy())
                self.label_5fold[name].extend(val_labels.cpu())

                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                # print('outputs:', val_outputs.argmax(dim=1), 'labels:', val_labels, 'value:', value)
                metric_count += len(value)
                num_correct += value.sum().item()
            #print(metric_count)
            #exit()
        acc = num_correct / metric_count
        try:
            #auc = roc_auc_score(label_all, prob_all)
            if len(class_list) == 3:
                auc = roc_auc_score(label_all, prob_all,labels=class_list,multi_class='ovo')
                #for class_idx in range(len(class_list)):
                #    label_all_byclass.append([int(label==class_idx) for label in label_all])
                #    pred_all_byclass.append([int(pred==class_idx) for pred in pred_all])
                #auc = []
#
                #prob_all = np.asarray(prob_all)
                #for class_idx in range(len(class_list)):
                #    print('label',label_all_byclass[class_idx])
                #    print('pred',pred_all_byclass[class_idx])
                #    auc.append(roc_auc_score(label_all_byclass[class_idx], prob_all[:,class_idx],labels=[0,1]))
                #    equal_list = [int(label_all_byclass[class_idx][i]==pred_all_byclass[class_idx][i]) for i in range(len(label_all_byclass[class_idx])) ]
                #    print(sum(equal_list)/len(equal_list))
                #    acc_all_byclass.append(sum(equal_list)/len(equal_list))
                #print(acc)
                #print(auc)
            else:
                auc = roc_auc_score(label_all, prob_all,labels=class_list)
        except ValueError:
            auc = np.nan
            pass    

        cf_matrix = confusion_matrix(label_all, pred_all,labels=class_list)
        print(cf_matrix)
        if len(class_list) == 3:
            f1 = f1_score(label_all, pred_all,average='macro')  
        else:
            f1 = f1_score(label_all, pred_all)  

        self.fold_history['Fold'].append(fold_idx)
        self.fold_history['Name'].append(name)
        self.fold_history['Accuracy'].append(acc)
        self.fold_history['AUC'].append(auc)
        self.fold_history['CM'].append(cf_matrix)
        self.fold_history['F1 Score'].append(f1)

    def evaluation_v2(self, model, data_loader, device , use_mask, is_radiomics=False,class_list=[0,1]):
        prob_all = []
        pred_all = []
        label_all = [] 
        set_all = []
        softmax = nn.Softmax(dim=1)
        for val_data in data_loader:
            if use_mask:
                val_images, val_labels, val_masks = val_data["image"].to(device), val_data["label"].to(device), val_data["mask"].to(device)
            else:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            with torch.no_grad():
                val_outputs, val_radiomics = model(val_images)
                preds = torch.max(val_outputs.data, 1)[1]   

                # for auc used
                if len(class_list)==3:
                    prob_all.extend(softmax(val_outputs.cpu().detach()).numpy())
                    
                else:
                    prob_all.extend(val_outputs[:,1].cpu().detach().numpy())
                #prob_all.extend(val_outputs[:,1].cpu().detach().numpy())
                label_all.extend(val_labels.cpu().numpy())
                pred_all.extend(preds.cpu().numpy())
                set_all.extend(val_data["set"])
        
        assert len(set_all) == len(prob_all) == len(pred_all) ==len(label_all)
        self.export_dict_v2['set'].extend(set_all)
        self.export_dict_v2['y_pred_prob'].extend(prob_all)
        self.export_dict_v2['y_pred_class'].extend(pred_all)
        self.export_dict_v2['y_truth'].extend(label_all)

def draw_result(lst_iter, lst_loss, lst_cal_loss, lst_rad_loss, lst_val_loss, path, title):
    plt.plot(lst_iter, lst_loss, '-b', label='loss')
    plt.plot(lst_iter, lst_val_loss, '-g', label='val loss')
    plt.plot(lst_iter, lst_cal_loss, '-r', label='classification loss')
    plt.plot(lst_iter, lst_rad_loss, '-c', label='radiomics loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('loss per epoch')

    # save image
    plt.savefig(path + '/'  + title+ ".png")  # should before show method

    # show
    plt.show()
    plt.clf()

