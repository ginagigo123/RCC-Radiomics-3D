import radiomics
from radiomics import featureextractor
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import six
import joblib
import pickle
from glob import glob
import regex

import faulthandler

# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码


## 111866

# read parameter
# label_want = int(input("Using kidney(1) / tumor(2): "))
paramsFile = os.path.join('/mnt/Internal/JNJ/HMI/judy/pyradiomics','example3D.yaml')

# feature extract
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
# extractor.enableAllFeatures()

print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor.enabledImagetypes)
print("Enabled features:\n\t", extractor.enabledFeatures)

records = []
export_path = '/mnt/Internal/JNJ/HMI/gina/classification/inputs/radiomics_3D/vghtc_0603_max_npy'

# import vghtc data
lowres_path = "/mnt/Internal/JNJ/HMI/gina/classification/inputs/cc3d/vghtc_0603_seg_img_max_min/{}.nii.gz"
img_path = "/mnt/Internal/JNJ/HMI/gina/classification/inputs/cc3d/vghtc_0603_img_max_min/{}.nii.gz"
# lowres_path = "/mnt/External/JNJ/huanyu/JNJ_Alternity2/Results/nnU-Net_prediction_kitsonvghtc_phase3/3d_lowres/{}.nii.gz"
# img_path = "/mnt/External/JNJ/huanyu/JNJ_Alternity2_Segate/Data/vghtc/serve_phase3_uid/{}.nii.gz"


p = '/mnt/Internal/JNJ/huanyu/JNJ_Alternity2/Processed/Labels/vghtc_manual_sel.pkl'
raw_data = joblib.load(p)

voxel_df = pd.read_csv('/mnt/Internal/JNJ/HMI/gina/classification/inputs/cc3d/vghtc_voxel.csv', index_col=0, dtype=str)
huge_voxel = voxel_df.loc[voxel_df['voxel'].astype(int) > 40000]
print(huge_voxel['uid'].values)

# listCase = [64, 103, 234]
failed_case = []

flag = False
count = 0
# print(raw_data.shape)
for _, row in raw_data.iterrows():
    # count += 1
    # if count < 199:
    #     continue
#     labelPath = os.path.join(dataDir, 'labelsTr', case + '.nii.gz')
    # if flag == False:
    #     # some rare case: 99999.11761326679111025454
    #     # '99999.11761326679111025454', '99999.1156176136189161963287', , '99999.128211912131442675'
    #     print(row['seriesInstanceUID'])
    #     if row['seriesInstanceUID'] in ['99999.1156176136189161963287']:
    #         flag = True
    #         # continue
    #     else:
    #         continue
    # if flag == True:
    #     if row['seriesInstanceUID'] in ['99999.11761326679111025454', '99999.1156176136189161963287']:
    #         continue

    # print('running:', row['seriesInstanceUID'])
    print(row['seriesInstanceUID'])
    if row['seriesInstanceUID'] in huge_voxel['uid'].values:
        print('huge:', row['seriesInstanceUID'])
        continue

    try:
        img_path_uid = img_path.format(row['seriesInstanceUID'])
        lowres_path_uid = lowres_path.format(row['seriesInstanceUID'])

        result = extractor.execute(img_path_uid, lowres_path_uid, voxelBased=True) # , voxelBased=True
        record = {}
        record["Case Name"] = row['seriesInstanceUID']
        for featureName, featureValue in six.iteritems(result):
            if isinstance(featureValue, sitk.Image):  # Feature map
                file_name = record["Case Name"] + '_' + featureName + '.npy'
                file_path = os.path.join(export_path, file_name)

                # turn to numpy
                npImage = sitk.GetArrayFromImage(featureValue)
                npImage[np.isnan(npImage)] = 0
                # print(f"{row['seriesInstanceUID']} shape:", npImage.shape)

                np.save(file_path, npImage)
                # sitk.WriteImage(featureValue, file_path , True)
                print("Stored feature %s in %s" % (featureName, record["Case Name"] + ".npy"))
            else:  # Diagnostic information
                print("\t%s: %s" %(featureName, featureValue))

        # break
    except Exception as e:
        print(row['seriesInstanceUID'], "failed")
        failed_case.append(row['seriesInstanceUID'])
        # break
    
# print(failed_case)
# df = pd.DataFrame(records)
# df.to_csv(outputPath + "KiTS_near_kidney_radiomics_feature_lowres.csv")
print("finish!")