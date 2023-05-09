import h5py
from loss_dice.dsc import *
from model.UCTransNet import *
from model.Config import get_CTranS_config
from dataloader.dataloader_seg import data_loaders
import os
import time
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTHONHASHSEED'] = str(666)
sep = os.sep

features_get = []
def hook(module, input, output):
    # module: model.conv2
    # input :in forward function  [#2]
    # output:is  [#3 self.conv2(out)]
    features_get.append(output.detach().cpu())
    # output is saved  in a list

if __name__=='__main__':
    key_word = 'AP_M40'
    datapath = r'./AP_M40'
    csvpath = os.path.join(os.path.abspath('..'), 'data_info.csv')
    modelpath = os.path.join(os.path.abspath('..'), 'Weight_model', key_word, 'uctransnet.pt')
    resultsave = os.path.join(os.path.abspath('..'), 'Match_library', key_word)

    # path
    gpu = True
    os.makedirs(resultsave, exist_ok=True)

    # load config
    config_vit = get_CTranS_config()
    ucnet = UCTransNet(config_vit, n_channels=config_vit.n_channels, n_classes=config_vit.n_labels,
                       img_size=config_vit.img_size)
    if gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ucnet.to(device)

    # dataloader
    dataloader_train, _ = data_loaders(datapath, csvpath, task='MTM', batch_size=1, workers=40)
    # load model
    state_dict = torch.load(modelpath)
    ucnet.load_state_dict(state_dict)
    ucnet.eval()

    input_list = []
    pred_list = []
    true_list = []
    feature_values = np.zeros((len(dataloader_train), 512, 1, 1))
    print(len(dataloader_train))
    for i, data in enumerate(dataloader_train):
        t1 = time.time()
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            handle = ucnet.down4.register_forward_hook(hook)
            y_pred = ucnet(x)
            # get features
            feature_save = features_get[0]  # tensor 1x512xMxN
            # feature_list.extend([feature_save[s] for s in range(feature_save.shape[0])])
            G = torch.nn.AdaptiveAvgPool2d((1, 1))
            GAP = G(feature_save)
            GAP_2 = GAP.detach().cpu().numpy()
            # save feature
            feature_values[i, :, :, :] = GAP_2
            # print(feature_values)
            features_get = []
            print(i, '/ ', len(dataloader_train.dataset.chooseslicepath),' ',time.time() - t1, ' s')
    handle.remove()

    feature_values_mean = np.average(feature_values,axis=0)
    feature_values_save = feature_values_mean.reshape((1,512))
    # save match library
    save_path = os.path.join(resultsave, 'match_data.h5')
    os.makedirs(resultsave, exist_ok=True)
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('f_values', data=feature_values_save)
    print('Match - ok')



