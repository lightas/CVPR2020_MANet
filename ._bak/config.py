# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

#class Configuration():
#        def __init__(self):
#                self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__")))
#                self.EXP_NAME = 'deeplabv3+coco'
#
#                self.NUM_WORKER=1
#                self.KNNS=1
#                self.PRETRAINED_MODEL='./model_best.pth.tar'
#                self.RESULT_ROOT = os.path.join(self.ROOT_DIR,'inter_result_new_re_80000')
#                self.DATA_NAME = 'COCO2017'
#                self.DATA_AUG = True
#                self.DATA_WORKERS =4 
#                self.DATA_RESCALE = 417
#                self.DATA_RANDOMCROP = 417
#                self.DATA_RANDOMROTATION = 0
#                self.DATA_RANDOMSCALE = 2
#                self.DATA_RANDOM_H = 10
#                self.DATA_RANDOM_S = 10
#                self.DATA_RANDOM_V = 10
#                self.DATA_RANDOMFLIP = 0.5
#                
#                self.DATA_ROOT= '/home/miaojiaxu/jiaxu_2/data/DAVIS'
#                self.YTB_DATAROOT='/raid/dataset/jiaxu/vos'
#
#                self.MODEL_NAME = 'deeplabv3plus'
#                self.MODEL_BACKBONE = 'res101_atrous'
#                self.MODEL_OUTPUT_STRIDE = 16
#                self.MODEL_ASPP_OUTDIM = 256
#                self.MODEL_SHORTCUT_DIM = 48
#                self.MODEL_SHORTCUT_KERNEL = 1
#                self.MODEL_NUM_CLASSES = 21
#                self.MODEL_SEMANTIC_EMBEDDING_DIM=100
#                self.MODEL_HEAD_EMBEDDING_DIM=256
#                self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)
#                self.MODEL_LOCAL_DOWNSAMPLE=True
#                self.MODEL_MAX_LOCAL_DISTANCE=8
#                self.INTER_USE_GLOBAL=False
#                self.MODEL_GLOBAL_ATTEN=False
#                self.MODEL_SIAMESE_ATTETION=False
#
#
#                self.TRAIN_LR = 0.0007
#                self.TRAIN_LR_GAMMA = 0.1
#                self.TRAIN_MOMENTUM = 0.9
#                self.TRAIN_WEIGHT_DECAY = 0.00004
#                self.TRAIN_POWER = 0.9
#                self.TRAIN_GPUS = 4
#                self.TRAIN_BATCH_SIZE = 2
#                self.TRAIN_SHUFFLE = True
#                self.TRAIN_CLIP_GRAD_NORM= 5.
#                self.TRAIN_MINEPOCH = 9
#                self.TRAIN_EPOCHS = int(200000*self.TRAIN_BATCH_SIZE/60.)
#                self.TRAIN_TOTAL_STEPS=101000
#                self.TRAIN_LOSS_LAMBDA = 0
#                self.TRAIN_TBLOG = False
#                self.TRAIN_CKPT = '/home/jiaxu/project/interactive_seg/deeplabv3plus-pytorch//model/deeplabv3+coco/deeplabv3plus_xception_COCO2017_itr55000.pth'
#                self.TRAIN_BN_MOM = 0.0003
#                self.TRAIN_TOP_K_PERCENT_PIXELS=0.15
#                self.TRAIN_HARD_MINING_STEP=50000
#                self.TRAIN_LR_STEPSIZE=2000
#                self.TRAIN_INTER_USE_TRUE_RESULT=True
#                self.TRAIN_TRIPLET_MARGIN=0.2
#                self.TRAIN_LAMBDA=0.05
#                self.TRAIN_TRI_SELECT_NUM=100
#
#
#                self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)
#
#                self.TEST_MULTISCALE = [1.0]#[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
#                self.TEST_FLIP = False
#                self.TEST_CKPT = None#os.path.join(self.ROOT_DIR,'model/danetvoc/DANet_xception_VOC2012_epoch60_all.pth')
#                self.TEST_GPUS = 2
#                self.TEST_BATCHES = 8           
#
#                self.__check()
#                self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
#                
#        def __check(self):
#                if not torch.cuda.is_available():
#                        raise ValueError('config.py: cuda is not avalable')
#                if self.TRAIN_GPUS == 0:
#                        raise ValueError('config.py: the number of GPU is 0')
#                #if self.TRAIN_GPUS != torch.cuda.device_count():
#                #       raise ValueError('config.py: GPU number is not matched')
#                if not os.path.isdir(self.LOG_DIR):
#                        os.makedirs(self.LOG_DIR)
#                if not os.path.isdir(self.MODEL_SAVE_DIR):
#                        os.makedirs(self.MODEL_SAVE_DIR)
#
#        def __add_path(self, path):
#                if path not in sys.path:
#                        sys.path.insert(0, path)
#
#
#
#cfg = Configuration()   
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='intvos config')
parser.add_argument('--ROOT_DIR' ,type=str, default=os.path.abspath(os.path.join(os.path.dirname("__file__"))))
parser.add_argument('--EXP_NAME', type=str, default='deeplabv3+coco')
parser.add_argument('--SAVE_RESULT_DIR',type=str,default='../afs/result/')
parser.add_argument('--SAVE_VOS_RESULT_DIR',type=str,default='')
parser.add_argument('--NUM_WORKER',type=int,default=1)
parser.add_argument('--KNNS',type=int,default=1)
parser.add_argument('--PRETRAINED_MODEL',type=str,default='./model_best.pth.tar')
parser.add_argument('--RESULT_ROOT',type=str,default= os.path.join('../afs/vos_result/result_total_80000'))
parser.add_argument('--DATA_NAME',type=str,default= 'COCO2017')
parser.add_argument('--DATA_AUG' ,type=str2bool,default= True)
parser.add_argument('--DATA_WORKERS',type=int,default=4) 
parser.add_argument('--DATA_RESCALE',type=int,default= 416)
parser.add_argument('--DATA_RANDOMCROP',type=int,default = 416)
parser.add_argument('--DATA_RANDOMROTATION',type=int,default = 0)

parser.add_argument('--DATA_RANDOM_H',type=int,default= 10)
parser.add_argument('--DATA_RANDOM_S',type=int,default = 10)
parser.add_argument('--DATA_RANDOM_V' ,type=int,default= 10)
parser.add_argument('--DATA_RANDOMFLIP',type=float, default=0.5)

                
parser.add_argument('--DATA_ROOT',type=str,default= '../data/DAVIS')
parser.add_argument('--YTB_DATAROOT',type=str,default='/raid/dataset/jiaxu/vos')

parser.add_argument('--MODEL_NAME',type=str,default = 'deeplabv3plus')
parser.add_argument('--MODEL_BACKBONE',type=str,default = 'res101_atrous')
parser.add_argument('--MODEL_OUTPUT_STRIDE',type=int,default = 16)
parser.add_argument('--MODEL_ASPP_OUTDIM',type=int,default = 256)
parser.add_argument('--MODEL_SHORTCUT_DIM',type=int,default = 48)
parser.add_argument('--MODEL_SHORTCUT_KERNEL',type=int,default = 1)
parser.add_argument('--MODEL_NUM_CLASSES',type=int,default = 21)
parser.add_argument('--MODEL_SEMANTIC_EMBEDDING_DIM',type=int,default=100)
parser.add_argument('--MODEL_HEAD_EMBEDDING_DIM',type=int,default=256)
parser.add_argument('--MODEL_SAVE_DIR',type=str,default = '')
parser.add_argument('--MODEL_LOCAL_DOWNSAMPLE',type=str2bool,default=True)
parser.add_argument('--MODEL_MAX_LOCAL_DISTANCE',type=int,default=12)
parser.add_argument('--INTER_USE_GLOBAL',type=str2bool,default=False)
parser.add_argument('--MODEL_GLOBAL_ATTEN',type=str2bool,default=False)
parser.add_argument('--MODEL_SIAMESE_ATTETION',type=str2bool,default=False)
parser.add_argument('--MODEL_PREROUND_MAP',type=str2bool,default=False)
parser.add_argument('--MODEL_MULTI_LOCAL',type=int,default=2)
parser.add_argument('--MODEL_SELECT_PERCENT',type=float,default=0.8)
parser.add_argument('--MODEL_USE_EDGE',type=str2bool,default=False)
parser.add_argument('--MODEL_USE_EDGE_2',type=str2bool,default=False)
parser.add_argument('--MODEL_USE_EDGE_3',type=str2bool,default=False)
parser.add_argument('--USE2ROUND',type=int,default=5)
parser.add_argument('--USE1DIST',type=int,default=15)
parser.add_argument('--CENTER_CAT',type=str2bool,default=False)
parser.add_argument('--USE_EXTREME_POINT',type=str2bool,default=True)
#parser.add_argument('--EXTREME_CAT',type=str2bool,default=False)

parser.add_argument('--FUSE',type=str2bool,default=False)
parser.add_argument('--USE_PRE',type=str2bool,default=False)



parser.add_argument('--TRAIN_LR',type=float,default = 0.0007)
parser.add_argument('--TRAIN_LR_GAMMA',type=float,default = 0.1)
parser.add_argument('--TRAIN_MOMENTUM',type=float,default = 0.9)
parser.add_argument('--TRAIN_WEIGHT_DECAY',type=float,default = 0.00004)
parser.add_argument('--TRAIN_POWER',type=float,default = 0.9)
parser.add_argument('--TRAIN_GPUS',type=int,default = 4)
parser.add_argument('--TRAIN_BATCH_SIZE',type=int,default = 2)
parser.add_argument('--TRAIN_SHUFFLE',type=str2bool,default = True)
parser.add_argument('--TRAIN_CLIP_GRAD_NORM',type=float,default= 5.)
parser.add_argument('--TRAIN_MINEPOCH',type=int,default = 9)
#parser.add_argument('--TRAIN_EPOCHS',type=int,default = int(200000*self.TRAIN_BATCH_SIZE/60.))
parser.add_argument('--TRAIN_TOTAL_STEPS',type=int,default=101000)
parser.add_argument('--TRAIN_LOSS_LAMBDA',type=int,default = 0)
parser.add_argument('--TRAIN_TBLOG',type=str2bool,default = False)
parser.add_argument('--TRAIN_CKPT',type=str,default = '/home/jiaxu/project/interactive_seg/deeplabv3plus-pytorch//model/deeplabv3+coco/deeplabv3plus_xception_COCO2017_itr55000.pth')
parser.add_argument('--TRAIN_BN_MOM',type=float,default = 0.0003)
parser.add_argument('--TRAIN_TOP_K_PERCENT_PIXELS',type=float,default=0.15)
parser.add_argument('--TRAIN_HARD_MINING_STEP',type=int,default=50000)
parser.add_argument('--TRAIN_LR_STEPSIZE',type=int,default=2000)
parser.add_argument('--TRAIN_INTER_USE_TRUE_RESULT',type=str2bool,default=True)
parser.add_argument('--TRAIN_TRIPLET_MARGIN',type=float,default=2)
parser.add_argument('--TRAIN_LAMBDA',type=float,default=0.05)
parser.add_argument('--TRAIN_TRI_SELECT_NUM',type=int,default=50)
parser.add_argument('--TRAIN_RESUME_DIR',type=str,default='')

parser.add_argument('--LOG_DIR',type=str,default = os.path.join('./log'))

parser.add_argument('--TEST_MULTISCALE',type=float,default = [1.0])#[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
parser.add_argument('--TEST_FLIP',type=str2bool,default = False)
parser.add_argument('--TEST_GPUS',type=int,default = 2)
parser.add_argument('--TEST_BATCHES',type=int,default = 8)           
parser.add_argument('--TEST_CHECKPOINT',type=str,default='save_step_100000.pth')

cfg=parser.parse_args()
cfg.TRAIN_EPOCHS=int(200000*cfg.TRAIN_BATCH_SIZE/60.)
                
if not torch.cuda.is_available():
        raise ValueError('config.py: cuda is not avalable')
if cfg.TRAIN_GPUS == 0:
        raise ValueError('config.py: the number of GPU is 0')
#if self.TRAIN_GPUS != torch.cuda.device_count():
#       raise ValueError('config.py: GPU number is not matched')
#if not os.path.isdir(cfg.LOG_DIR):
#        os.makedirs(cfg.LOG_DIR)
#if not os.path.isdir(cfg.MODEL_SAVE_DIR):
#        os.makedirs(cfg.MODEL_SAVE_DIR)


