"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

from attack import attacks
#from attack.merge_face import merge
from torchvision import transforms as t
from analysis.ssim_cal import calculate_metrics as cal_ssim
from dml_csr.ASMA_parsing import parse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument("--test_dataset", nargs="+", default="FaceForensice++")
parser.add_argument('--target_detector_path', type=str, help='path to the target detector YAML file')
parser.add_argument('--target_weights_path', type=str, help='path to the target detector weights file')
#parser.add_argument("--lmdb", action='store_true', default=False)
parser.add_argument('--attack_method', type=str, default='ASMA', help='adversarial attack method from [fgsm, pgd, cw, df, ASMA, pASMA, jitter, BSR]')
parser.add_argument('--eps', type=str, default='2/255', help='epsilon for adversarial attack')
parser.add_argument('--eval_detector_path', type=str, help='path to the eval detector YAML file')
parser.add_argument('--eval_weights_path', type=str, help='path to the eval detector weights file')
parser.add_argument('--features', type=str, nargs='+', default=['nose',  'mouth', 'hair'], help='attack feature for ASMA')  # 接受多个字符串
parser.add_argument('--iqa', type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def test_one_dataset_attack(model, model2, key, data_loader, attack):
    """
    Generate adversarial examples and compute predictions for a dataset.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        attack: Adversarial attack instance (e.g., PGD).

    Returns:
        tuple: (predictions, labels, features)
    """
    prediction_lists = []
    feature_lists = []
    label_lists = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)

        # move data to GPU

        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)


        data_dict['image'] = (data_dict['image'] + 1) / 2

        #Generate adversarial examples
        if args.attack_method == 'ASMA':
        
            adv_data = parse(data_dict['image'], attack, data_dict['label'], features=args.features)
            
        else:
            adv_data = attack(data_dict['image'], data_dict['label'])
        


        mse_list = []
        mae_list = []
        ssim_list = []
        mssim_list = []
        psnr_list = []
        if (args.iqa):
            ssim, mssim, psnr, mse, mae = cal_ssim(data_dict['image'], adv_data)
            ssim_list.append(ssim)
            mssim_list.append(mssim)
            psnr_list.append(psnr)
            mse_list.append(mse)
            mae_list.append(mae)

        data_dict['image'] = adv_data
        # Model forward without considering gradient computation
        data_dict['image'] = data_dict['image'].to('cuda:2')
        data_dict['label'] = data_dict['label'].to('cuda:2')
        torch.cuda.set_device(2)
        predictions = inference(model2, data_dict)
        torch.cuda.set_device(0)
        # Record results
        label_ = list(data_dict['label'].cpu().detach().numpy())
        predict_ = list(predictions['prob'].cpu().detach().numpy())
        label_lists += label_
        prediction_lists += predict_
        feature_lists += list(predictions['feat'].cpu().detach().numpy())

    if (args.iqa):
        print("evaluation:")
        print("mse", np.mean(mse_list))
        print("mae", np.mean(mae_list))
        print("psnr", np.mean(psnr_list))
        print("ssim", np.mean(ssim_list))
        print("ms_ssim", np.mean(mssim_list))

    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def test_attack(model, model2, test_data_loaders):

    # define test recorder
    metrics_all_datasets = {}

    if args.attack_method == 'fgsm':
        ad = attacks.FGSM(model, eval(args.eps))
    elif args.attack_method == 'pgd':
        ad = attacks.PGD(model, eval(args.eps), alpha=2/255, steps=10, random_start=False)
    elif args.attack_method == 'cw':
        ad = attacks.CW(model, c=1, kappa=0, steps=10, lr=0.1)
    elif args.attack_method == 'jitter':
        ad = attacks.Jitter(model)
    elif args.attack_method == 'BSR':
        ad = attacks.BSR(model)
    elif args.attack_method == 'ASMA':
        ad = attacks.ASMA(model, eps=eval(args.eps), alpha=2/255, steps=10, random_start=False)
    elif args.attack_method == 'pASMA':
        ad = attacks.pASMA(model)

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        
        # generate adversarial examples and compute predictions for each dataset
        predictions_nps, label_nps, feat_nps = test_one_dataset_attack(model, model2, key, test_data_loaders[key], ad)
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets


def main():
    # parse options and load config
    with open(args.target_detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.eval_detector_path, 'r') as f:
         config_yaml2 = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    weights_path2 = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.target_weights_path:
        config['weights_path'] = args.target_weights_path
        weights_path = args.target_weights_path
    
    if args.test_dataset:
        config_yaml2['test_dataset'] = args.test_dataset
    if args.eval_weights_path:
        config_yaml2['weights_path'] = args.eval_weights_path
        weights_path2 = args.eval_weights_path

    #init seed
    init_seed(config)
    init_seed(config_yaml2)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load target checkpoint done!')
    else:
        print('Fail to load the target pre-trained weights')

    model_class2 = DETECTOR[config_yaml2['model_name']]
    model2 = model_class2(config_yaml2).to('cuda:2')
    epoch2 = 0
    if weights_path2:
        try:
            epoch2 = int(weights_path2.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch2 = 0
        ckpt2 = torch.load(weights_path2, map_location='cuda:2')
        model2.load_state_dict(ckpt2, strict=True)
        print('===> Load eval checkpoint done!')
    else:
         print('Fail to load the eval pre-trained weights')

    attack_metric = test_attack(model, model2, test_data_loaders)
    print("==>attack done")
    

if __name__ == '__main__':
    main()
