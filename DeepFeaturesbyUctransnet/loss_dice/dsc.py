import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
sep = os.sep

def findindex(vinSets):
    list_vin = []
    for i in vinSets:
        address_index = [x for x in range(len(vinSets)) if vinSets[x] == i]
        list_vin.append([i, address_index])
    dict_address = dict(list_vin)
    return dict_address

def dsc(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    if (np.sum(y_pred) + np.sum(y_true)) == 0:
        return 0
    else:
        return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def dsc_distribution(volumes):
    dsc_dict = {}
    for p in volumes.keys():
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        dsc_dict[p] = dsc(y_pred, y_true)
    return dsc_dict


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    '''Calculate DSC according to the patiens'''
    dsc_list = []
    num_slices = [p.split(sep)[-1].split('_')[0]+'_'+p.split(sep)[-1].split('_')[1] for p in patient_slice_index]
    dict_address = findindex(num_slices)
    for p in dict_address.keys():
        y_pred = []
        y_true = []
        patient_index = dict_address[p][:]
        for get in patient_index:
            y_pred.append(validation_pred[get])
            y_true.append(validation_true[get])
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        dsc_list.append(dsc(y_pred, y_true))
    return dsc_list


def postprocess_per_volume(input_list, pred_list, true_list, patient_slice_index):
    volumes = {}
    num_slices = [p.split(sep)[-1].split('_')[0] for p in patient_slice_index]
    dict_address = findindex(num_slices)
    for p in dict_address.keys():
        y_pred = []
        y_true = []
        y_input = []
        patient_index = dict_address[p][:]
        for get in patient_index:
            y_input.append(input_list[get])
            y_pred.append(pred_list[get])
            y_true.append(true_list[get])
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_input = np.array(y_input)
        volumes[p] = (y_input, y_pred, y_true)
    return volumes

def feature_per_volume(feature_list, patient_slice_index):
    '''
    input: feature(numpy),N x 512 x 1 x 1
    output: dict { ID:feature(numpy) }
    '''
    volumes = {}
    num_slices = [p.split(sep)[-1].split('_')[0] for p in patient_slice_index]
    dict_address = findindex(num_slices)
    for p in dict_address.keys():
        y_input = []
        patient_index = dict_address[p][:]
        for get in patient_index:
            y_input.append(feature_list[get])
        y_input = np.array(y_input)
        volumes[p] = (y_input)
    return volumes


def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))

def plot_dsc(dsc_dist):
    y_positions = np.arange(len(dsc_dist))
    dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dsc_dist]
    labels = [x[0] for x in dsc_dist]
    #labels = ["_".join(l.split("_")[1:-1]) for l in labels]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))
