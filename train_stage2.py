import cv2
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms 
from dataloaders.davis_2017_f import DAVIS2017_Train,DAVIS2017_VOS_Train,DAVIS2017_VOS_Test,DAVIS2017_Test
import dataloaders.custom_transforms_f as tr
from dataloaders.samplers import RandomIdentitySampler
from networks.deeplab import DeepLab
from networks.IntVOS import IntVOS
from networks.loss import Added_BCEWithLogitsLoss,Added_CrossEntropyLoss
from config import cfg
from utils.meters import AverageMeter
from utils.mask_damaging import damage_masks,mask_damager
from utils.utils import label2colormap
from PIL import Image
import random
import scipy.misc as sm
import time
import davisinteractive.robot.interactive_robot as interactive_robot
if cfg.TRAIN_TBLOG:
    from tensorboardX import SummaryWriter
    tblogger = SummaryWriter(cfg.LOG_DIR)

torch.autograd.set_detect_anomaly(True)
class Manager(object):
    def __init__(self, use_gpu=True,time_budget=None,
        save_result_dir=cfg.SAVE_RESULT_DIR,pretrained=True,interactive_test=False):

        self.save_res_dir = save_result_dir
        self.time_budget=time_budget
        self.feature_extracter = DeepLab(backbone='resnet')

        if pretrained:
            pretrained_dict = torch.load(cfg.PRETRAINED_MODEL)
            pretrained_dict = pretrained_dict['state_dict']
            self.load_network(self.feature_extracter,pretrained_dict)
            print('load pretrained model successfully.')
        self.model = IntVOS(cfg,self.feature_extracter)
        print(self.model)
        pd = torch.load(os.path.join(cfg.SAVE_VOS_RESULT_DIR,'save_step_100000.pth'))

        self.load_network(self.model,pd)
        self.use_gpu=use_gpu
        if use_gpu:
            self.model = self.model.cuda()

##################################
    def train(self,damage_initial_previous_frame_mask=True,lossfunc='cross_entropy',model_resume=False,eval_total=False,init_prev=False):
        ###################
        interactor = interactive_robot.InteractiveScribblesRobot()
        self.model.train()
        running_loss = AverageMeter()
        optimizer = optim.SGD(self.model.inter_seghead.parameters(),lr=cfg.TRAIN_LR,momentum=cfg.TRAIN_MOMENTUM,weight_decay=cfg.TRAIN_WEIGHT_DECAY)
        

        ###################

        composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
                                                    tr.RandomScale(),
                                                     tr.RandomCrop((cfg.DATA_RANDOMCROP,cfg.DATA_RANDOMCROP),10),
                                                     tr.Resize(cfg.DATA_RESCALE),
                                                     tr.ToTensor()])
        print('dataset processing...')
        train_dataset = DAVIS2017_Train(root=cfg.DATA_ROOT, transform=composed_transforms)
        train_list = train_dataset.seqs

        print('dataset processing finished.')
        if lossfunc=='bce':
            criterion = Added_BCEWithLogitsLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS,cfg.TRAIN_HARD_MINING_STEP)
        elif lossfunc=='cross_entropy':
            criterion = Added_CrossEntropyLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS,cfg.TRAIN_HARD_MINING_STEP)
        else:
            print('unsupported loss funciton. Please choose from [cross_entropy,bce]')

        max_itr = cfg.TRAIN_TOTAL_STEPS

        step=0
        round_=3
        epoch_per_round=30
        if model_resume:
            saved_model_=os.path.join(self.save_res_dir,'save_step_75000.pth')

            saved_model_ = torch.load(saved_model_)
            self.model=self.load_network(self.model,saved_model_)
            step=75000
            print('resume from step {}'.format(step))
        while step<cfg.TRAIN_TOTAL_STEPS:
            
            if step>100001:
                break

            for r in range(round_):
                if r==0: #### r==0: Train the interaction branch in the first round
                    print('start new')
                    global_map_tmp_dic={}
                    train_dataset.transform=transforms.Compose([tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
                                                    tr.RandomScale(),
                                                     tr.RandomCrop((cfg.DATA_RANDOMCROP,cfg.DATA_RANDOMCROP)),
                                                     tr.Resize(cfg.DATA_RESCALE),
                                                     tr.ToTensor()])
                    train_dataset.init_ref_frame_dic()

                trainloader = DataLoader(train_dataset,batch_size=cfg.TRAIN_BATCH_SIZE,
                        sampler = RandomIdentitySampler(train_dataset.sample_list), 
                        shuffle=False,num_workers=cfg.NUM_WORKER,pin_memory=True)
                print('round:{} start'.format(r))
                for epoch in range(epoch_per_round):



                    for ii, sample in enumerate(trainloader):
                        now_lr=self._adjust_lr(optimizer,step,max_itr)
                        ref_imgs = sample['ref_img'] #batch_size * 3 * h * w
                        ref_scribble_labels = sample['ref_scribble_label'] #batch_size * 1 * h * w
                        seq_names = sample['meta']['seq_name'] 
                        obj_nums = sample['meta']['obj_num']
                        ref_frame_nums = sample['meta']['ref_frame_num']
                        ref_frame_gts=sample['ref_frame_gt']
                        bs,_,h,w = ref_imgs.size()
                        ##########
                        if self.use_gpu:
                            inputs = ref_imgs.cuda()
                        
                            ref_scribble_labels=ref_scribble_labels.cuda()
                            ref_frame_gts = ref_frame_gts.cuda()
                        ##########        
                        with torch.no_grad():
                            self.model.feature_extracter.eval()
                            self.model.semantic_embedding.eval()
                            ref_frame_embedding = self.model.extract_feature(inputs)
                        if r==0:
                            first_inter=True

                            tmp_dic = self.model.int_seghead(ref_frame_embedding=ref_frame_embedding,ref_scribble_label=ref_scribble_labels,
                                    prev_round_label=None,normalize_nearest_neighbor_distances=True,global_map_tmp_dic={},
                                 seq_names=seq_names,gt_ids=obj_nums,k_nearest_neighbors=cfg.KNNS,
                                 frame_num=ref_frame_nums,first_inter=first_inter)
                        else:
                            first_inter=False
                            prev_round_label=sample['prev_round_label']
                            prev_round_label=prev_round_label.cuda()
                            tmp_dic = self.model.int_seghead(ref_frame_embedding=ref_frame_embedding,ref_scribble_label=ref_scribble_labels,
                                    prev_round_label=prev_round_label,normalize_nearest_neighbor_distances=True,global_map_tmp_dic={},
                                 seq_names=seq_names,gt_ids=obj_nums,k_nearest_neighbors=cfg.KNNS,
                                 frame_num=ref_frame_nums,first_inter=first_inter)




                        label_and_obj_dic={}
                        label_dic={}
                        for i, seq_ in enumerate(seq_names):
                            label_and_obj_dic[seq_]=(ref_frame_gts[i],obj_nums[i])
                        for seq_ in tmp_dic.keys():
                            tmp_pred_logits = tmp_dic[seq_]
                            tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits,size=(h,w),mode = 'bilinear',align_corners=True)
                            tmp_dic[seq_]=tmp_pred_logits        

                            label_tmp,obj_num = label_and_obj_dic[seq_]
                            obj_ids = np.arange(0,obj_num+1)
                            obj_ids = torch.from_numpy(obj_ids)
                            obj_ids = obj_ids.int()
                            if torch.cuda.is_available():
                                obj_ids = obj_ids.cuda()
                            if lossfunc == 'bce':
                                label_tmp = label_tmp.permute(1,2,0)
                                label = (label_tmp.float()==obj_ids.float())
                                label = label.unsqueeze(-1).permute(3,2,0,1)
                                label_dic[seq_]=label.float()
                            elif lossfunc =='cross_entropy':
                                label_dic[seq_]=label_tmp.long()        
        
        
                        loss = criterion(tmp_dic,label_dic,step)
                        loss =loss/bs
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        

                        running_loss.update(loss.item(),bs)
                        if step%50==0:
                            torch.cuda.empty_cache()
                            print('step:{},now_lr:{} ,loss:{:.4f}({:.4f})'.format(step,now_lr ,running_loss.val,running_loss.avg))
                            
                            show_ref_img = ref_imgs.cpu().numpy()[0]

                            mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
                            sigma = np.array([[[0.229]], [[0.224]], [[0.225]]])        

                            show_ref_img = show_ref_img*sigma+mean
        


                            show_gt = ref_frame_gts.cpu()[0].squeeze(0).numpy()
                            show_gtf = label2colormap(show_gt).transpose((2,0,1))        
                            show_scrbble=ref_scribble_labels.cpu()[0].squeeze(0).numpy()
                            show_scrbble=label2colormap(show_scrbble).transpose((2,0,1))
                            if r!=0:
                                show_prev_round_label=prev_round_label.cpu()[0].squeeze(0).numpy()
                                show_prev_round_label=label2colormap(show_prev_round_label).transpose((2,0,1))
                            else:
                                show_prev_round_label = np.zeros_like(show_gt)
                                
                                show_prev_round_label = label2colormap(show_prev_round_label).transpose((2,0,1))

                            ##########
                            show_preds = tmp_dic[seq_names[0]].cpu()
                            show_preds=nn.functional.interpolate(show_preds,size=(h,w),mode = 'bilinear',align_corners=True)
                            show_preds = show_preds.squeeze(0)
                            if lossfunc=='bce':
                                show_preds = show_preds[1:]

                                show_preds = (torch.sigmoid(show_preds)>0.5)
                                marker = torch.argmax(show_preds,dim=0)
                                show_preds_s = torch.zeros((h,w))
                                for i in range(show_preds.size(0)):
                                    tmp_mask = (marker==i) & (show_preds[i]>0.5)
                                    show_preds_s[tmp_mask]=i+1
                            elif lossfunc=='cross_entropy':
                                show_preds_s = torch.argmax(show_preds,dim=0)
                            show_preds_s = show_preds_s.numpy()
                            show_preds_sf = label2colormap(show_preds_s).transpose((2,0,1))        

                            pix_acc = np.sum(show_preds_s==show_gt)/(h*w)        
        
        
        
                            if cfg.TRAIN_TBLOG:
                                tblogger.add_scalar('loss', running_loss.avg, step)
                                tblogger.add_scalar('pix_acc', pix_acc, step)
                                tblogger.add_scalar('now_lr', now_lr, step)
                                tblogger.add_image('Reference image', show_ref_img, step)
                                #tblogger.add_image('Previous frame image', show_img1, step)
                                
                           #     tblogger.add_image('Current frame image', show_img2, step)
                                tblogger.add_image('Groud Truth', show_gtf, step)
                                tblogger.add_image('Predict label', show_preds_sf, step)        
        
                                tblogger.add_image('Scribble', show_scrbble, step)
                                tblogger.add_image('prev_round_label', show_prev_round_label, step)

                            ###########TODO
                        if step%20000==0 and step!=0:
                            self.save_network(self.model,step)        
        
        
                
                
                        step+=1
                print('trainset evaluating...')
                print('*'*100)


                if cfg.TRAIN_INTER_USE_TRUE_RESULT:
                    if r!=round_-1:
                        if r ==0:
                            prev_round_label_dic={}
                        self.model.eval()
                        with torch.no_grad():
                            round_scribble={}

                            frame_num_dic= {}
                            train_dataset.transform=transforms.Compose([tr.Resize(cfg.DATA_RESCALE),tr.ToTensor()])
                            trainloader = DataLoader(train_dataset,batch_size=1,
                                sampler = RandomIdentitySampler(train_dataset.sample_list), 
                                shuffle=False,num_workers=cfg.NUM_WORKER,pin_memory=True)
                            for ii, sample in enumerate(trainloader):
                                ref_imgs = sample['ref_img'] #batch_size * 3 * h * w
                                img1s = sample['img1'] 
                                img2s = sample['img2']
                                ref_scribble_labels = sample['ref_scribble_label'] #batch_size * 1 * h * w
                                label1s = sample['label1']
                                label2s = sample['label2']
                                seq_names = sample['meta']['seq_name'] 
                                obj_nums = sample['meta']['obj_num']
                                frame_nums = sample['meta']['frame_num']
                                bs,_,h,w = img2s.size()
                                inputs = torch.cat((ref_imgs,img1s,img2s),0)
                                if r==0:
                                    ref_scribble_labels=self.rough_ROI(ref_scribble_labels)
                                print(seq_names[0])
                                label1s_tocat=torch.Tensor()
                                for i in range(bs):
                                    l = label1s[i]
                                    l = l.unsqueeze(0)
                                    l = mask_damager(l,0.0)
                                    l = torch.from_numpy(l)

                                    l = l.unsqueeze(0).unsqueeze(0)
                        
                                    label1s_tocat = torch.cat((label1s_tocat,l.float()),0)
                                label1s = label1s_tocat
                                if self.use_gpu:
                                    inputs = inputs.cuda()
                                    ref_scribble_labels=ref_scribble_labels.cuda()
                                    label1s = label1s.cuda()
                                    
                                tmp_dic, global_map_tmp_dic = self.model(inputs,ref_scribble_labels,label1s,seq_names=seq_names,
                                                            gt_ids=obj_nums,k_nearest_neighbors=cfg.KNNS,global_map_tmp_dic=global_map_tmp_dic,
                                                           frame_num=frame_nums)
                                pred_label = tmp_dic[seq_names[0]].detach().cpu()
                                pred_label = nn.functional.interpolate(pred_label,size=(h,w),mode = 'bilinear',align_corners=True)            

                                pred_label=torch.argmax(pred_label,dim=1)
                                pred_label= pred_label.unsqueeze(0)
                                try:
                                    pred_label=damage_masks(pred_label)
                                except:
                                    pred_label=pred_label
                                pred_label=pred_label.squeeze(0)
                                round_scribble[seq_names[0]]=interactor.interact(seq_names[0],pred_label.numpy(),label2s.float().squeeze(0).numpy(),obj_nums)
                                frame_num_dic[seq_names[0]]=frame_nums[0]        
                                pred_label=pred_label.unsqueeze(0)
                                img_ww=Image.open(os.path.join(cfg.DATA_ROOT,'JPEGImages/480p/',seq_names[0],'00000.jpg'))
                                img_ww=np.array(img_ww)
                                or_h,or_w = img_ww.shape[:2]
                                pred_label = torch.nn.functional.interpolate(pred_label.float(),(or_h,or_w),mode='nearest')

                                prev_round_label_dic[seq_names[0]]=pred_label.squeeze(0)
#                                torch.cuda.empty_cache() 
                        train_dataset.update_ref_frame_and_label(round_scribble,frame_num_dic,prev_round_label_dic)
                    print('trainset evaluating finished!')
                    print('*'*100)
                    self.model.train()
                    print('updating ref frame and label')


                    train_dataset.transform=composed_transforms 
                    print('updating ref frame and label finished!')



                else:
                    if r!=round_-1:
                        round_scribble={}    

                        if r ==0:
                            prev_round_label_dic={}
                        frame_num_dic= {}
                        train_dataset.transform=tr.ToTensor()
                        trainloader = DataLoader(train_dataset,batch_size=1,
                            sampler = RandomIdentitySampler(train_dataset.sample_list), 
                            shuffle=False,num_workers=1,pin_memory=True)
                        
                        self.model.eval()
                        with torch.no_grad():
                            for ii, sample in enumerate(trainloader):
                                ref_imgs = sample['ref_img'] #batch_size * 3 * h * w
                                img1s = sample['img1'] 
                                img2s = sample['img2']
                                ref_scribble_labels = sample['ref_scribble_label'] #batch_size * 1 * h * w
                                label1s = sample['label1']
                                label2s = sample['label2']
                                seq_names = sample['meta']['seq_name'] 
                                obj_nums = sample['meta']['obj_num']
                                frame_nums = sample['meta']['frame_num']
                                bs,_,h,w = img2s.size()
                                
                                print(seq_names[0])
                                label2s_ = mask_damager(label2s,0.1)
                                round_scribble[seq_names[0]]=interactor.interact(seq_names[0],np.expand_dims(label2s_,axis=0),label2s.float().squeeze(0).numpy(),obj_nums)
                                label2s__=torch.from_numpy(label2s_)

                                frame_num_dic[seq_names[0]]=frame_nums[0]
                                prev_round_label_dic[seq_names[0]]=label2s__    
    

                        print('trainset evaluating finished!')
                        print('*'*100)
                        print('updating ref frame and label')    

                        train_dataset.update_ref_frame_and_label(round_scribble,frame_num_dic,prev_round_label_dic)
                        self.model.train()
                        train_dataset.transform=composed_transforms 
                        print('updating ref frame and label finished!')
#############################################






    def rough_ROI(self,ref_scribble_labels):
        #### b*1*h*w
        dist=15
        b,_,h,w = ref_scribble_labels.size()
        filter_= torch.zeros_like(ref_scribble_labels)
        to_fill = torch.zeros_like(ref_scribble_labels)
        for i in range(b):

            no_background = (ref_scribble_labels[i]!=-1)
            no_background = no_background.squeeze(0)
            

            no_b=no_background.nonzero()
            (h_min,w_min),_ = torch.min(no_b,0)
            (h_max,w_max),_ = torch.max(no_b,0)


            filter_[i,0,max(h_min-dist,0):min(h_max+dist,h-1),max(w_min-dist,0):min(w_max+dist,w-1)]=1

        final_scribble_labels = torch.where(filter_.byte(),ref_scribble_labels,to_fill)
        return final_scribble_labels


    def load_network(self,net,pretrained_dict):

        #pretrained_dict = pretrained_dict
        model_dict = net.state_dict()
           # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        return net
    def save_network(self,net,step):
        save_path = self.save_res_dir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = 'save_step_%s.pth'%(step)
        torch.save(net.state_dict(),os.path.join(save_path,save_file))

    def _adjust_lr(self,optimizer, itr, max_itr):
        now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
        optimizer.param_groups[0]['lr'] = now_lr
        return now_lr

_palette=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

manager = Manager()
manager.train()



