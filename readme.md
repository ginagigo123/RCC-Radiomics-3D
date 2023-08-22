# RCC: Tumor Classification

## Classification of Tumor Label
We had implemented 3 different experiments. 
1. Task - Benign v.s Low Grade v.s High Grade
2. Task - Benign v.s Malignant
3. Task - Low Grade v.s High Grade

## Data
Please put data in the `inputs/` folder. The example data comes from KiTS datasets. Please download the `img_case_0.nii.gz` file [here](https://drive.google.com/file/d/1YMkDObtnUcdLIiUebf6PdOe0Bdr4ragn/view?usp=sharing). The file is too large and can not be uploaded to Github.

## Installation
There are 2 env we should install.
1. Monai (include CC3D):
Please follow the installation of official website [here](https://docs.monai.io/en/stable/installation.html).

2. Radiomics
Please follow the installation [here](https://github.com/LinYuXuan-judy/KidneyTumorclassification).

## Exeperiment Process
![process](image\image.png)

Data (Ct image and segmentation) -> Preprocessing `connected_component_cropped.ipynb` -> Cropped Data (both CT image and segmentation) -> Radiomics feature extract `radiomics_3D.py` -> Model predict `predict_radiomics.py` -> export table `export_general.py` & `export_lowPR.py`

### Preprocessing: Connected Componenet
The file will crop the original CT image and segmentation according to region of the maximum tumors. It will also visualize the result of cropping.

Please activate the `monai` env first.
```
conda activate monai
```

And run all the cell in `connected_component_cropped.ipynb`

After running, the cropped image will show in the `inputs/img/` folder.

![case](image\case_00000.png)

### Preprocessing: Radiomics
Radiomics will extract the 3D features and generate approximately 580 features in total.

Please activate the `radiomics` env first and run the rdaiomics.

```
conda activate radiomics
python preprocessing/radiomics_3D.py
```

If we visualize the process, it will be like:
![extraction](image\extraction.png)

### Prediction
In model training, the Deep Profiler will be used and it based on the `monai` framework. So, here activate the env `monai` first.

The Deep Profiler structure:
![model](image\model.png)

Before running the prediction, there are some args you can set:
```
options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE, --bz BATCH_SIZE
                        input batch size for training (default: 128)
  --epochs EPOCHS, -e EPOCHS
                        number of epochs to train (default: 10)
  --use_mask, -m        train with mask
  --label_type {t1a,lowhigh,malignant,total}, -l {t1a,lowhigh,malignant,total}
```

If you want to classify 3 types, you can run the below command:
```
python predict_radiomics_3D.py --lable_type total
```

### Export the result
The result of prediction will be put in `ouputs`. Then you can run both `export_general.py` and `export_lowPR.py` to generate table.
