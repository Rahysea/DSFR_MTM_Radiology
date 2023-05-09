# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:44 下午
# @Author  : Haonan Wang
# @File    : Config.py
# @Software: PyCharm
import time
import ml_collections


task_name = 'MTM'
model_name = 'UCTransNet_384'
train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name +'/'+ model_name +'/' + session_name + '/'
model_path  = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path  = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [8,4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    config.img_size = [224, 224]
    config.save_model = True
    config.tensorboard = True
    config.n_channels = 1
    config.n_labels = 1
    config.batch_size = 1
    config.epochs = 500
    config.cosineLR = True  # whether use cosineLR or not
    config.print_frequency = 1
    config.save_frequency = 5000
    config.vis_frequency = 10
    config.early_stopping_patience = 50
    config.pretrain = False
    config.learning_rate = 1e-3
    return config
