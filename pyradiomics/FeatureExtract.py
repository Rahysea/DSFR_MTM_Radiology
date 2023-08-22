import os
import re
import h5py
import numpy as np
from radiomics import featureextractor as FEE
sep = os.sep

def cir_get_features(imageFilepath, maskFilepath, yamlpath):
    extractor = FEE.RadiomicsFeatureExtractor(yamlpath)
    result = extractor.execute(imageFilepath, maskFilepath)
    feature = {}
    for key in result.keys():
        if not 'diagnostics' in key:
            feature[key] = float(result[key])
    return feature

def RadioFeature(Imgpath, savefeaturepath, yamlpath, maps):
    '''
    :param Imgpath:
                -Imgpath
                ---IMG
                ----M40
                -------AP_S1.nii.gz
                -------AP_S2.nii.gz
                -------....nii.gz
                ----MASK
                -------AP_S1.nii.gz
                -------AP_S2.nii.gz
                -------....nii.gz
    :param savefeaturepath:
                -AP_M40
                ---S1.h5
                ---S2.h5
                ---....h5
    :param yamlpath:
    :return: features.h5 files
    '''
    os.makedirs(savefeaturepath, exist_ok=True)
    flag = 0
    imgpath = Imgpath + sep + 'IMG' + sep + maps
    maskpath = Imgpath + sep + 'MASK'
    img = sorted(os.listdir(imgpath))
    mask = sorted(os.listdir(maskpath))
    num = len(img)
    for imgname, maskname in zip(img, mask):
        name = re.split(r'[.]+', imgname)[0]
        name_check = re.split(r'[.]+', maskname)[0]
        if name == name_check:
            flag += 1
            imageFilepath = os.path.join(imgpath, imgname)
            maskFilepath = os.path.join(maskpath, maskname)
            datainfo = imgname.split('.')[0]
            # files information
            id = datainfo.split('_')[1]
            phase_map = datainfo.split('_')[0]+'_'+maps
            print('Schedule:' + str(flag) + ' / ' + str(num) + '||   Current: ' + str(imgname) + ' | ' + str(maskname))
            # h5 save features
            finall_save_path = os.path.join(savefeaturepath, phase_map)
            os.makedirs(finall_save_path, exist_ok=True)
            feature = cir_get_features(imageFilepath, maskFilepath, yamlpath)
            value_array = np.array(list(feature.values()))   # dist to ndarray
            value_array = value_array.reshape(1, -1)
            with h5py.File(os.path.join(finall_save_path, '%s.h5' % id), mode='w') as f:
                f.create_dataset('f_values', data=value_array)
            f.close()
        else:
            raise ValueError("mriname: ", imgname, "roiname:", maskname)


def run():
    radiopath = r'./Imgpath'
    savefeaturepath = r'./radio_features'
    maps = 'M40'
    yamlpath = r'./Params_labels.yaml'
    RadioFeature(radiopath, savefeaturepath, yamlpath, maps)

if __name__=='__main__':
    run()
