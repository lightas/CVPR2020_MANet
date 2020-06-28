from __future__ import division
import json
import os
import shutil
import numpy as np
import torch, cv2
from random import choice
from torch.utils.data import Dataset
import json
from PIL import Image
from  davisinteractive.utils.scribbles import scribbles2mask,annotated_frames
import sys
sys.path.append("..")
from config import cfg

class DAVIS2017_VOS_Test(Dataset):
    """
    """

    def __init__(self,split='val',root=cfg.DATA_ROOT,transform=None,rgb=False,result_root=None,seq_name=None):
        self.split=split
        self.db_root_dir = root
        self.result_root=result_root
        self.rgb=rgb
        self.transform = transform
        self.seq_name = seq_name
        self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                          '_'.join(self.split) + '_instances.txt')

        self.seqs = []
        for splt in self.split:
            with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)

        if not self._check_preprocess():
            self._preprocess()

        assert self.seq_name in self.seq_dict.keys(), '{} not in {} set.'.format(self.seq_name, '_'.join(self.split))
        names_img = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', str(seq_name))))
        img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
        name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', str(seq_name))))
        labels = list(map(lambda x: os.path.join('Annotations/480p/', str(seq_name), x), name_label))

        if not os.path.isfile(os.path.join(self.result_root,seq_name,name_label[0])):
            if not os.path.exists(os.path.join(self.result_root,seq_name)):
                os.makedirs(os.path.join(self.result_root,seq_name))

                shutil.copy(os.path.join(self.db_root_dir,labels[0]),os.path.join(self.result_root,seq_name,name_label[0]))
            else:
                shutil.copy(os.path.join(self.db_root_dir,labels[0]),os.path.join(self.result_root,seq_name,name_label[0]))
        self.first_img = names_img[0]
        self.first_label = name_label[0]
        self.img_list=names_img[1:]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):

        img= self.img_list[idx]
        imgpath = os.path.join(self.db_root_dir,'JPEGImages/480p/',str(self.seq_name),img)

        num_frame  = int(img.split('.')[0])
        ref_img = os.path.join(self.db_root_dir,'JPEGImages/480p/',str(self.seq_name),self.first_img)
        prev_frame = num_frame-1
        prev_frame = str(prev_frame)
        while len(prev_frame)!=5:
            prev_frame = '0'+prev_frame
        prev_img = os.path.join(self.db_root_dir,'JPEGImages/480p/',str(self.seq_name),prev_frame+'.'+img.split('.')[-1])

        current_img = cv2.imread(imgpath)
        current_img = np.array(current_img,dtype=np.float32)

        ref_img = cv2.imread(ref_img)
        ref_img = np.array(ref_img,dtype=np.float32)

        prev_img = cv2.imread(prev_img)
        prev_img = np.array(prev_img,dtype=np.float32)

        ref_label = os.path.join(self.db_root_dir,'Annotations/480p/',str(self.seq_name),self.first_label)
        ref_label = Image.open(ref_label)
        ref_label = np.array(ref_label,dtype=np.uint8)

        prev_label = os.path.join(self.result_root,str(self.seq_name),prev_frame+'.'+self.first_label.split('.')[-1])
        prev_label = Image.open(prev_label)
        prev_label = np.array(prev_label,dtype=np.uint8)

        obj_num = self.seq_dict[self.seq_name][-1]
        sample = {'ref_img':ref_img,'prev_img':prev_img,'current_img':current_img,'ref_label':ref_label,'prev_label':prev_label}
        sample['meta']={'seq_name':self.seq_name,'frame_num':num_frame,'obj_num':obj_num,'current_name':img}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/', seq, name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj+1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
            for ii in range(1, len(self.seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')


class DAVIS2017_VOS_Train(Dataset):
    """DAVIS2017 dataset for training
    
    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,split='train',
                root=cfg.DATA_ROOT,
                transform=None,
                rgb=False
                ):
        self.split=split
        self.db_root_dir=root
        self.rgb=rgb
        self.transform=transform
        self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                          '_'.join(self.split) + '_instances.txt')
        self.seqs = []
        for splt in self.split:
            with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)
        self.imglistdic={}
        if not self._check_preprocess():
            self._preprocess()
        self.sample_list=[]
        for seq_name in self.seqs:
            images = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', seq_name.strip())))
            images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(), x), images))
            lab = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq_name.strip())))
            lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq_name.strip(), x), lab))
            self.imglistdic[seq_name]=(images,lab)





    def __len__(self):
        return len(self.seqs)
    def __getitem__(self,idx):
        seqname=self.seqs[idx]
        imagelist,lablist = self.imglistdic[seqname]
        prev_img = np.random.choice(imagelist[:-1],1)
        prev_img = prev_img[0]
        frame_num = int(prev_img.split('.')[0])+1
        next_frame=str(frame_num)
        while len(next_frame)!=5:
            next_frame='0'+next_frame


        ###############################Processing two adjacent frames and labels
        img2path=os.path.join('JPEGImages/480p/',seqname,next_frame+'.'+prev_img.split('.')[-1])
        img2 = cv2.imread(os.path.join(self.db_root_dir, img2path))
        img2 = np.array(img2,dtype=np.float32)

        imgpath = os.path.join('JPEGImages/480p/',seqname,prev_img)
        img1 = cv2.imread(os.path.join(self.db_root_dir, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        labelpath = os.path.join('Annotations/480p/', seqname,prev_img.split('.')[0]+'.'+lablist[0].split('.')[-1])
        label1 = Image.open(os.path.join(self.db_root_dir, labelpath))
        label2path=os.path.join('Annotations/480p/',seqname,next_frame+'.'+lablist[0].split('.')[-1])
        label2 = Image.open(os.path.join(self.db_root_dir,label2path))




        label1 = np.array(label1, dtype=np.uint8)
        label2 = np.array(label2, dtype=np.uint8)
        ###################
        ref_img = np.random.choice(imagelist,1)
        ref_img = ref_img[0]
        ref_img_name = ref_img
        ref_scribble_label = Image.open(os.path.join(self.db_root_dir,'Annotations/480p/',seqname,ref_img_name.split('.')[0]+'.'+lablist[0].split('.')[-1]))
        ref_scribble_label = np.array(ref_scribble_label,dtype=np.uint8)
        while len(np.unique(ref_scribble_label))<self.seq_dict[seqname][-1]+1 or ref_img==prev_img or ref_img==(next_frame+'.'+prev_img.split('.')[-1]):
            ref_img = np.random.choice(imagelist,1)
            ref_img = ref_img[0]
            ref_img_name = ref_img
            ref_scribble_label = Image.open(os.path.join(self.db_root_dir,'Annotations/480p/',seqname,ref_img_name.split('.')[0]+'.'+lablist[0].split('.')[-1]))
            ref_scribble_label = np.array(ref_scribble_label,dtype=np.uint8)
        ref_img = os.path.join('JPEGImages/480p/',seqname,ref_img)
        ref_img = cv2.imread(os.path.join(self.db_root_dir,ref_img))
        ref_img=np.array(ref_img,dtype=np.float32)
        ####
        ###################
        if self.rgb:
            img1 = img1[:, :, [2, 1, 0]]
            img2 = img2[:, :, [2, 1, 0]]
            ref_img = ref_img[:,:,[2,1,0]]
        obj_num=self.seq_dict[seqname][-1]
        sample={'ref_img':ref_img,'img1':img1,'img2':img2,'ref_scribble_label':ref_scribble_label,'label1':label1,'label2':label2}

        sample['meta']={'seq_name':seqname,'frame_num':frame_num,'obj_num':obj_num}
        if self.transform is not None:
            sample = self.transform(sample)


        return sample

########################


    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/', seq, name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj+1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
            for ii in range(1, len(self.seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

class DAVIS2017_Train(Dataset):
    """DAVIS2017 dataset for training
    
    Return: imgs: N*2*3*H*W,label: N*2*1*H*W, seq-name: N, frame_num:N
    """
    def __init__(self,split='train',
                root=cfg.DATA_ROOT,
                transform=None,
                rgb=False
                ):
        self.split=split
        self.db_root_dir=root
        self.rgb=rgb
        self.transform=transform
        self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017',
                                          '_'.join(self.split) + '_instances.txt')
        self.seqs = []
        for splt in self.split:
            with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', self.split + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            self.seqs.extend(seqs_tmp)        

        if not self._check_preprocess():
            self._preprocess()
        self._init_ref_frame_dic()
        self.sample_list=[]
        for seq_name in self.seqs:
            images = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', seq_name.strip())))
            images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq_name.strip(), x), images))
            lab = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq_name.strip())))
            lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq_name.strip(), x), lab))


            for img_path,label_path in zip(images_path[:-1],lab_path[:-1]):
                tmp_dic={'img':img_path,'label':label_path,'seq_name':seq_name,
                        'frame_num':img_path.split('/')[-1].split('.')[0]}
                self.sample_list.append(tmp_dic)



    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self,idx):
        tmp_sample=self.sample_list[idx]
        imgpath=tmp_sample['img']
        labelpath=tmp_sample['label']
        seqname=tmp_sample['seq_name']
        frame_num=int(tmp_sample['frame_num'])+1

        next_frame=str(frame_num)
        while len(next_frame)!=5:
            next_frame='0'+next_frame
        ###############################Processing two adjacent frames and labels
        img2path=os.path.join('JPEGImages/480p/',seqname,next_frame+'.'+imgpath.split('.')[-1])
        img2 = cv2.imread(os.path.join(self.db_root_dir, img2path))
        img2 = np.array(img2,dtype=np.float32)

        img1 = cv2.imread(os.path.join(self.db_root_dir, imgpath))
        img1 = np.array(img1, dtype=np.float32)
        ###############
        label1 = Image.open(os.path.join(self.db_root_dir, labelpath))
        label2path=os.path.join('Annotations/480p/',seqname,next_frame+'.'+labelpath.split('.')[-1])
        label2 = Image.open(os.path.join(self.db_root_dir,label2path))




        label1 = np.array(label1, dtype=np.uint8)
        label2 = np.array(label2, dtype=np.uint8)
        ###################
        ref_tmp_dic=self.ref_frame_dic[seqname]
        ref_img = ref_tmp_dic['ref_frame']
        ref_scribble_label = ref_tmp_dic['scribble_label']
        ref_img = cv2.imread(os.path.join(self.db_root_dir,ref_img))
        ref_img=np.array(ref_img,dtype=np.float32)

        ###################
        if self.rgb:
            img1 = img1[:, :, [2, 1, 0]]
            img2 = img2[:, :, [2, 1, 0]]
            ref_img = ref_img[:,:,[2,1,0]]
        obj_num=self.seq_dict[seqname][-1]
        sample={'ref_img':ref_img,'img1':img1,'img2':img2,'ref_scribble_label':ref_scribble_label,'label1':label1,'label2':label2}

        sample['meta']={'seq_name':seqname,'frame_num':frame_num,'obj_num':obj_num}
        if self.transform is not None:
            sample = self.transform(sample)


        return sample

#    def ref_frame_and_label(self,seq_name,frame_num,pred_label):
        ##########Update reference frame and scribbles
#        self.ref_frame_dic[seq_name]=

    def _init_ref_frame_dic(self):
        self.ref_frame_dic={}
        scribbles_path=os.path.join(self.db_root_dir,'Scribbles')
        for seq in self.seqs:
            scribble=os.path.join(self.db_root_dir,'Scribbles',seq,'001.json')
            with open(scribble) as f:
                scribble=json.load(f)
                scr_frame=annotated_frames(scribble)[0]
                scr_f=str(scr_frame)
                while len(scr_f)!=5:
                    scr_f='0'+scr_f

                ref_frame_path=os.path.join('JPEGImages/480p',seq,scr_f+'.jpg')
                ref_tmp = cv2.imread(os.path.join(self.db_root_dir,ref_frame_path))
                h_,w_=ref_tmp.shape[:2]
                scribble_masks=scribbles2mask(scribble,(h_,w_))
                scribble_label=scribble_masks[scr_frame]
                self.ref_frame_dic[seq]={'ref_frame':ref_frame_path,'scribble_label':scribble_label}
#    def _get_scribble_label(self,):


#######################TODO
#    def previous_round_label(self,seq_name,frame_num,pred_label,round):


########################


    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            return False
        else:
            self.seq_dict = json.load(open(self.seq_list_file, 'r'))
            return True

    def _preprocess(self):
        self.seq_dict = {}
        for seq in self.seqs:
            # Read object masks and get number of objects
            name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
            label_path = os.path.join(self.db_root_dir, 'Annotations/480p/', seq, name_label[0])
            _mask = np.array(Image.open(label_path))
            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]

            self.seq_dict[seq] = list(range(1, n_obj+1))

        with open(self.seq_list_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
            for ii in range(1, len(self.seqs)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

# class DAVIS2017_Test():

# class DAVIS2017(Dataset):
#     """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""

#     def __init__(self, split='val',
#                  root=Path.db_root_dir(),
#                  num_frames=None,
#                  custom_frames=None,
#                  transform=None,
#                  retname=False,
#                  seq_name=None,
#                  obj_id=None,
#                  gt_only_first_frame=False,
#                  no_gt=False,
#                  batch_gt=False,
#                  rgb=False,
#                  effective_batch=None):
#         """Loads image to label pairs for tool pose estimation
#         split: Split or list of splits of the dataset
#         root: dataset directory with subfolders "JPEGImages" and "Annotations"
#         num_frames: Select number of frames of the sequence (None for all frames)
#         custom_frames: List or Tuple with the number of the frames to include
#         transform: Data transformations
#         retname: Retrieve meta data in the sample key 'meta'
#         seq_name: Use a specific sequence
#         obj_id: Use a specific object of a sequence (If None and sequence is specified, the batch_gt is True)
#         gt_only_first_frame: Provide the GT only in the first frame
#         no_gt: No GT is provided
#         batch_gt: For every frame sequence batch all the different objects gt
#         rgb: Use RGB channel order in the image
#         """
#         if isinstance(split, str):
#             self.split = [split]
#         else:
#             split.sort()
#             self.split = split
#         self.db_root_dir = root
#         self.transform = transform
#         self.seq_name = seq_name
#         self.obj_id = obj_id
#         self.num_frames = num_frames
#         self.custom_frames = custom_frames
#         self.retname = retname
#         self.rgb = rgb
#         if seq_name is not None and obj_id is None:
#             batch_gt = True
#         self.batch_gt = batch_gt
#         self.all_seqs_list = []

#         self.seqs = []
#         for splt in self.split:
#             with open(os.path.join(self.db_root_dir, 'ImageSets', '2017', splt + '.txt')) as f:
#                 seqs_tmp = f.readlines()
#             seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
#             self.seqs.extend(seqs_tmp)
#         self.seq_list_file = os.path.join(self.db_root_dir, 'ImageSets', '2017',
#                                           '_'.join(self.split) + '_instances.txt')
#         # Precompute the dictionary with the objects per sequence
#         if not self._check_preprocess():
#             self._preprocess()

#         if self.seq_name is None:
#             img_list = []
#             labels = []
#             for seq in self.seqs:
#                 images = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', seq.strip())))
#                 images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
#                 lab = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq.strip())))
#                 lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
#                 if num_frames is not None:
#                     seq_len = len(images_path)
#                     num_frames = min(num_frames, seq_len)
#                     frame_vector = np.arange(num_frames)
#                     frames_ids = list(np.round(frame_vector*seq_len/float(num_frames)).astype(np.int))
#                     frames_ids[-1] = min(frames_ids[-1], seq_len)
#                     images_path = [images_path[x] for x in frames_ids]
#                     if no_gt:
#                         lab_path = [None] * len(images_path)
#                     else:
#                         lab_path = [lab_path[x] for x in frames_ids]
#                 elif isinstance(custom_frames, tuple) or isinstance(custom_frames, list):
#                     assert min(custom_frames) >= 0 and max(custom_frames) <= len(images_path)
#                     images_path = [images_path[x] for x in custom_frames]
#                     if no_gt:
#                         lab_path = [None] * len(images_path)
#                     else:
#                         lab_path = [lab_path[x] for x in custom_frames]
#                 if gt_only_first_frame:
#                     lab_path = [lab_path[0]]
#                     lab_path.extend([None] * (len(images_path) - 1))
#                 elif no_gt:
#                     lab_path = [None] * len(images_path)
#                 if self.batch_gt:
#                     obj = self.seq_dict[seq]
#                     if -1 in obj:
#                         obj.remove(-1)
#                     for ii in range(len(img_list), len(images_path)+len(img_list)):
#                         self.all_seqs_list.append([obj, ii])
#                 else:
#                     for obj in self.seq_dict[seq]:
#                         if obj != -1:
#                             for ii in range(len(img_list), len(images_path)+len(img_list)):
#                                 self.all_seqs_list.append([obj, ii])

#                 img_list.extend(images_path)
#                 labels.extend(lab_path)
#         else:
#             # Initialize the per sequence images for online training
#             assert self.seq_name in self.seq_dict.keys(), '{} not in {} set.'.format(self.seq_name, '_'.join(self.split))
#             names_img = np.sort(os.listdir(os.path.join(self.db_root_dir, 'JPEGImages/480p/', str(seq_name))))
#             img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
#             name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', str(seq_name))))
#             labels = list(map(lambda x: os.path.join('Annotations/480p/', str(seq_name), x), name_label))
#             if num_frames is not None:
#                 seq_len = len(img_list)
#                 num_frames = min(num_frames, seq_len)
#                 frame_vector = np.arange(num_frames)
#                 frames_ids = list(np.round(frame_vector * seq_len / float(num_frames)).astype(np.int))
#                 frames_ids[-1] = min(frames_ids[-1], seq_len)
#                 img_list = [img_list[x] for x in frames_ids]
#                 if no_gt:
#                     labels = [None] * len(img_list)
#                 else:
#                     labels = [labels[x] for x in frames_ids]
#             elif isinstance(custom_frames, tuple) or isinstance(custom_frames, list):
#                 assert min(custom_frames) >= 0 and max(custom_frames) <= len(img_list)
#                 img_list = [img_list[x] for x in custom_frames]
#                 if no_gt:
#                     labels = [None] * len(img_list)
#                 else:
#                     labels = [labels[x] for x in custom_frames]
#             if gt_only_first_frame:
#                 labels = [labels[0]]
#                 labels.extend([None]*(len(img_list)-1))
#             elif no_gt:
#                 labels = [None] * len(img_list)
#             if obj_id is not None:
#                 assert obj_id in self.seq_dict[self.seq_name], \
#                     "{} doesn't have this object id {}.".format(self.seq_name, str(obj_id))
#             if self.batch_gt:
#                 self.obj_id = self.seq_dict[self.seq_name]
#                 if -1 in self.obj_id:
#                     self.obj_id.remove(-1)

#         assert (len(labels) == len(img_list))

#         if effective_batch:
#             self.img_list = img_list * effective_batch
#             self.labels = labels * effective_batch
#         else:
#             self.img_list = img_list
#             self.labels = labels

#         # print('Done initializing DAVIS2017 '+'_'.join(self.split)+' Dataset')
#         # print('Number of images: {}'.format(len(self.img_list)))
#         # if self.seq_name is None:
#         #     print('Number of elements {}'.format(len(self.all_seqs_list)))

#     def _check_preprocess(self):
#         _seq_list_file = self.seq_list_file
#         if not os.path.isfile(_seq_list_file):
#             return False
#         else:
#             self.seq_dict = json.load(open(self.seq_list_file, 'r'))
#             return True

#     def _preprocess(self):
#         self.seq_dict = {}
#         for seq in self.seqs:
#             # Read object masks and get number of objects
#             name_label = np.sort(os.listdir(os.path.join(self.db_root_dir, 'Annotations/480p/', seq)))
#             label_path = os.path.join(self.db_root_dir, 'Annotations/480p/', seq, name_label[0])
#             _mask = np.array(Image.open(label_path))
#             _mask_ids = np.unique(_mask)
#             n_obj = _mask_ids[-1]

#             self.seq_dict[seq] = list(range(1, n_obj+1))

#         with open(self.seq_list_file, 'w') as outfile:
#             outfile.write('{{\n\t"{:s}": {:s}'.format(self.seqs[0], json.dumps(self.seq_dict[self.seqs[0]])))
#             for ii in range(1, len(self.seqs)):
#                 outfile.write(',\n\t"{:s}": {:s}'.format(self.seqs[ii], json.dumps(self.seq_dict[self.seqs[ii]])))
#             outfile.write('\n}\n')

#         print('Preprocessing finished')

#     def __len__(self):
#         if self.seq_name is None:
#             return len(self.all_seqs_list)
#         else:
#             return len(self.img_list)

#     def __getitem__(self, idx):
#         # print(idx)
#         img, gt = self.make_img_gt_pair(idx)

#         sample = {'image': img, 'gt': gt}

#         if self.retname:
#             if self.seq_name is None:
#                 obj_id = self.all_seqs_list[idx][0]
#                 img_path = self.img_list[self.all_seqs_list[idx][1]]
#             else:
#                 obj_id = self.obj_id
#                 img_path = self.img_list[idx]
#             seq_name = img_path.split('/')[-2]
#             frame_id = img_path.split('/')[-1].split('.')[-2]
#             sample['meta'] = {'seq_name': seq_name,
#                               'frame_id': frame_id,
#                               'obj_id': obj_id,
#                               'im_size': (img.shape[0], img.shape[1])}

#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample

#     def make_img_gt_pair(self, idx):
#         """
#         Make the image-ground-truth pair
#         """
#         if self.seq_name is None:
#             obj_id = self.all_seqs_list[idx][0]
#             img_path = self.img_list[self.all_seqs_list[idx][1]]
#             label_path = self.labels[self.all_seqs_list[idx][1]]
#         else:
#             obj_id = self.obj_id
#             img_path = self.img_list[idx]
#             label_path = self.labels[idx]
#         seq_name = img_path.split('/')[-2]
#         n_obj = 1 if isinstance(obj_id, int) else len(obj_id)
#         img = cv2.imread(os.path.join(self.db_root_dir, img_path))
#         img = np.array(img, dtype=np.float32)
#         if self.rgb:
#             img = img[:, :, [2, 1, 0]]

#         if label_path is not None:
#             label = Image.open(os.path.join(self.db_root_dir, label_path))
#         else:
#             if self.batch_gt:
#                 gt = np.zeros(np.append(img.shape[:-1], n_obj), dtype=np.float32)
#             else:
#                 gt = np.zeros(img.shape[:-1], dtype=np.float32)

#         if label_path is not None:
#             gt_tmp = np.array(label, dtype=np.uint8)
#             if self.batch_gt:
#                 gt = np.zeros(np.append(n_obj, gt_tmp.shape), dtype=np.float32)
#                 for ii, k in enumerate(obj_id):
#                     gt[ii, :, :] = gt_tmp == k
#                 gt = gt.transpose((1, 2, 0))
#             else:
#                 gt = (gt_tmp == obj_id).astype(np.float32)

#         return img, gt

#     def get_img_size(self):
#         img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
#         return list(img.shape[:2])

#     def __str__(self):
#         return 'DAVIS2017'
