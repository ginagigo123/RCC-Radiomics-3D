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

# read parameter
paramsFile = os.path.join('example3D.yaml')

# feature extract
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor.enabledImagetypes)
print("Enabled features:\n\t", extractor.enabledFeatures)

records = []
export_path = 'inputs/radiomics_3D'

# import vghtc data
img_path = "inputs/cc3d/img_case_0.nii.gz"
seg_path = "inputs/cc3d/seg_case_0.nii.gz"

voxel_df = pd.read_csv('inputs/cc3d/vghtc_voxel.csv', index_col=0, dtype=str)
huge_voxel = voxel_df.loc[voxel_df['voxel'].astype(int) > 40000]
print(huge_voxel['uid'].values)

if 'case_0' in huge_voxel['uid'].values:
    print('huge:', row['seriesInstanceUID'])

try:
    # Radiomics extract
    result = extractor.execute(img_path, seg_path, voxelBased=True)
    record = {}
    record["Case Name"] = row['seriesInstanceUID']
    for featureName, featureValue in six.iteritems(result):
        if isinstance(featureValue, sitk.Image):  # Feature map
            file_name = record["Case Name"] + '_' + featureName + '.npy'
            file_path = os.path.join(export_path, file_name)

            # turn to numpy and avoid nan value
            npImage = sitk.GetArrayFromImage(featureValue)
            npImage[np.isnan(npImage)] = 0

            # save the npy
            np.save(file_path, npImage)
            print("Stored feature %s in %s" % (featureName, record["Case Name"] + ".npy"))
        else:  # Diagnostic information
            print("\t%s: %s" %(featureName, featureValue))
except Exception as e:
    print("failed")

print("finish!")