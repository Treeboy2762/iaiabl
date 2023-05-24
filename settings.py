import torch
import numpy as np

base_architecture = 'resnet18'
img_size = 288
prototype_shape = (10, 512, 1, 1)
num_classes = 2

prototype_activation_function = "log"
prototype_activation_function_in_numpy = prototype_activation_function

class_specific = True

add_on_layers_type = 'regular'

experiment_run = '1218_fa='
data_path = '/Users/kkds7/Desktop/Medical/iaiabl/data/'
train_dir = data_path + 'train/'
test_dir = data_path + 'val/'
finer_annotation_dir = data_path + 'mask/'
train_push_dir = '/Users/kkds7/Desktop/Medical/iaiabl/data/push'

train_batch_size = 75
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 2e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'fine': 0.001,
}

num_train_epochs = 130
num_warm_epochs = 10

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
