import h5py
import copy
from sklearn.cluster import KMeans
from loss_dice.dsc import *
from model.UCTransNet import *
from model.Config import get_CTranS_config
from dataloader.dataloader_seg import data_loaders_all
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
    MatchPath = os.path.join(os.path.abspath('..'), 'Match_library', key_word)
    resultsave =  os.path.join(os.path.abspath('..'), 'deep_features', key_word)

    print(key_word)
    matchfile_name = os.listdir(MatchPath)
    Match_path = os.path.join(MatchPath, matchfile_name[0])
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    args = parser.parse_args()

    os.makedirs(resultsave, exist_ok=True)

    config_vit = get_CTranS_config()
    config_vit.batch_size = args.batch_size
    ucnet = UCTransNet(config_vit, n_channels=config_vit.n_channels, n_classes=config_vit.n_labels, img_size=config_vit.img_size)

    if args.gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ucnet.to(device)

    # Read match library
    with h5py.File(Match_path, mode='r') as f:
        match_feature = f['f_values'][:]

    # datasets
    dataloader_all = data_loaders_all(datapath, csvpath, batch_size=config_vit.batch_size, workers=0)

    # load model
    state_dict = torch.load(modelpath)
    ucnet.load_state_dict(state_dict)
    ucnet.eval()

    input_list = []
    pred_list = []
    true_list = []
    feature_values = np.zeros((len(dataloader_all.dataset.chooseslicepath), 512, 1, 1))
    flag = 0
    print('number: ', len(dataloader_all.dataset.chooseslicepath) / config_vit.batch_size)
    for i, data in enumerate(dataloader_all):
        t1 = time.time()
        x, y_true = data  # x=data, y_true=label
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            handle = ucnet.down4.register_forward_hook(hook)
            y_pred = ucnet(x)
            # get features
            feature_save = features_get[0]  # tensor 1x512xMxN
            G = torch.nn.AdaptiveAvgPool2d((1, 1))
            GAP = G(feature_save)
            GAP_2 = GAP.detach().cpu().numpy()
            # save feature
            if config_vit.batch_size > 1:
                for h in range(GAP_2.shape[0]):
                    feature_values[flag, :, :, :] = GAP_2[h]
                    flag += 1
            else:
                feature_values[i, :, :, :] = GAP_2
            features_get = []
            print(i, '/ ', len(dataloader_all.dataset.chooseslicepath),' ', time.time() - t1, ' s')
        handle.remove()
    volumes = feature_per_volume(
        feature_values,
        dataloader_all.dataset.chooseslicepath
    )
    # k-means
    for i, patient in enumerate(volumes):
        features_values = volumes[patient][:]
        if features_values.shape[0] != 1:
            features_values_2 = np.squeeze(features_values)  # remove 1
            #  Kmeans f-value cluster
            n_cluster = KMeans(n_clusters=2).fit_predict(copy.deepcopy(features_values_2))
            max_cluster = np.argmax(np.bincount(n_cluster))
            print(max_cluster, n_cluster)
            # Kmeans Max and Min cluster f-values
            Kmeans_max_mask_f_value = features_values_2[n_cluster == max_cluster, :]
            Kmeans_min_mask_f_value = features_values_2[n_cluster != max_cluster, :]
            #  average cluster f-values(N * 1 * 1024 --> 1 * 1024)
            f_max_values = np.average(Kmeans_max_mask_f_value, axis=0)
            f_max_values = f_max_values.reshape((1, 512))
            f_min_values = np.average(Kmeans_min_mask_f_value, axis=0)
            f_min_values = f_min_values.reshape((1, 512))

            max_d = np.linalg.norm(match_feature - f_max_values, ord=2)
            min_d = np.linalg.norm(match_feature - f_min_values, ord=2)
            if max_d < min_d:
                f_values = f_max_values
            else:
                f_values = f_min_values
        else:
            f_values = copy.deepcopy(features_values)

        # save features
        with h5py.File(os.path.join(resultsave, patient + '.h5'), 'w') as f:
            f.create_dataset('f_values', data=f_values)
        print('{}/{} open-mtm-dsfr feature extract {} done!'.format(i + 1, len(volumes), patient))





