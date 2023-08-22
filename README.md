# DSFR MTM Radiology

This repository includes the code for the Deep learning, Pyradiomics, R Statistics, as well as some baseline clinical information for the paper titled "Dual-Energy CT Deep Learning Radiomics to Predict Macrotrabecular-massive Hepatocellular Carcinoma" accepted in RADIOLOGY. 


If you have any question regarding the current code, please feel free to contact us.


## Usage
* *_Data Preparation_*
```
For Deep Learning
dataset_root
 ├── AP_MoneE-40
 │   ├── id_number.h5
 │   └── (S1234_1.h5)
 │   └── (S1234_2.h5)
 ├── PVP_MoneE-40
 │   ├── id_number.h5
 │   └── (S1234_1.h5)
 │   └── (S1234_2.h5)
 ├── AP_ElectronDensity
 │   ├── id_number.h5
 │   └── (S1234_1.h5)
 │   └── (S1234_2.h5)
 ├── .......
```
Datasets should be named `id_number.h5`, where `id` denotes the patient's ID and `number` indicates the z-axis slice count of the 3D image.  

Each `.h5` file should contain `image` expanded by 15 pixels from the ROI and then resized to 224×224, alongside the corresponding `label`segmentation gold standards.

```
For Radiomics
images_root
 ├── IMG
 │   ├── MoneE-40
 │   │   └── AP_S1234.nii.gz
 │   │   └── AP_S2345.nii.gz
 │   │   └── PVP_S1234.nii.gz
 │   ├── ElectronDensity
 │      
 ├── MASK
 │   ├── MoneE-40
 │   │   └── AP_S1234.nii.gz
 │   │   └── AP_S2345.nii.gz
 │   │   ........   
```

`datainfo.csv` is located in `R` folder.

* *_Deep Learnings Feature_*
```bash
1. python Train_Validation_UCtransnet.py
```
```bash
2. python Match_libary.py 
```
```bash
3. python Feature_extract.py 
```

* *_Radiomics Feature_*  
```bash
python pyradiomics\FeatureExtract.py 
```

* *_Feature selection_*  
```bash
python select_features.py
```

* *_Model building and statistic analysis_*  
```bash
nomogram_model.R
```

## Citation

If you find this code and paper useful for your research, please kindly cite our paper.

```

```
