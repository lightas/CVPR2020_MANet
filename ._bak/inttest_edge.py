import torch,cv2
import torch.nn as nn
import os
import json
from PIL import Image
import timeit
import numpy as np 
from dataloaders.davis_2017_f import DAVIS2017_Test,DAVIS2017_Test_Manager
import dataloaders.custom_transforms_f as tr
from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis
from networks.deeplab_edge import DeepLab
from  davisinteractive.utils.scribbles import scribbles2mask,annotated_frames
from torch.utils.data import DataLoader
import time


from networks.IntVOS_inter_edge import IntVOS
#from networks.deeplabv3plus import deeplabv3plus
from config import cfg 
def main():
    total_frame_num_dic={}
    #################
    seqs = []
    with open(os.path.join(cfg.DATA_ROOT, 'ImageSets', '2017', 'val' + '.txt')) as f:
        seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seqs.extend(seqs_tmp)
    for seq_name in seqs:
        images = np.sort(os.listdir(os.path.join(cfg.DATA_ROOT, 'JPEGImages/480p/', seq_name.strip())))
        total_frame_num_dic[seq_name]=len(images)
    _seq_list_file=os.path.join(cfg.DATA_ROOT, 'ImageSets', '2017',
                                          'v_a_l' + '_instances.txt')
    seq_dict = json.load(open(_seq_list_file, 'r'))
    ##################
    seq_imgnum_dict_={}
    seq_imgnum_dict=os.path.join(cfg.DATA_ROOT,'ImageSets','2017','val_imgnum.txt')
    if os.path.isfile(seq_imgnum_dict):

        seq_imgnum_dict_=json.load(open(seq_imgnum_dict,'r'))
    else:
        for seq in os.listdir(os.path.join(cfg.DATA_ROOT,'JPEGImages/480p/')):
            seq_imgnum_dict_[seq]=len(os.listdir(os.path.join(cfg.DATA_ROOT,'JPEGImages/480p/',seq)))
        with open(seq_imgnum_dict,'w') as f:
            json.dump(seq_imgnum_dict_,f)

    ##################

    is_save_image=False
    report_save_dir= cfg.RESULT_ROOT
    save_res_dir = cfg.SAVE_RESULT_DIR
    if not os.path.exists(cfg.RESULT_ROOT):
        os.makedirs(cfg.RESULT_ROOT)
        # Configuration used in the challenges
    max_nb_interactions = 8  # Maximum number of interactions
    max_time_per_interaction = 10000  # Maximum time per interaction per object
    # Total time available to interact with a sequence and an initial set of scribbles
    max_time = max_nb_interactions * max_time_per_interaction  # Maximum time per object
    # Interactive parameters
    subset = 'val'
    host = 'localhost'  # 'localhost' for subsets train and val.

    feature_extracter = DeepLab(backbone='resnet_edge',freeze_bn=False)
    model = IntVOS(cfg,feature_extracter)
    model= model.cuda()
    print('model loading...')

    saved_model_dict = os.path.join(save_res_dir,'save_step_80000.pth')
    pretrained_dict = torch.load(saved_model_dict)
    load_network(model,pretrained_dict)

    print('model loading finished!')
    model.eval()
    inter_file=open(os.path.join(cfg.RESULT_ROOT,'inter_file.txt'),'w')

    seen_seq=[]
    with torch.no_grad():
        with DavisInteractiveSession(host=host,
                                davis_root=cfg.DATA_ROOT,
                                subset=subset,
                                report_save_dir=report_save_dir,
                                max_nb_interactions=max_nb_interactions,
                                max_time=max_time,
                                metric_to_optimize='J'
                                                            
                                ) as sess:
            while sess.next():
    #            torch.cuda.empty_cache()
                t_total=timeit.default_timer()
                # Get the current iteration scribbles

                sequence, scribbles, first_scribble = sess.get_scribbles(only_last=True)
                print(sequence)
                if len(annotated_frames(scribbles))==0:

                    final_masks=prev_label_storage[:seq_imgnum_dict_[sequence]]
                    sess.submit_masks(final_masks.cpu().numpy())
                else:
                    start_annotated_frame=annotated_frames(scribbles)[0]
                    
                    pred_masks=[]
                    pred_masks_reverse=[]


                    if first_scribble:
                        if sequence not in seen_seq:
                            is_save_image=True
                            seen_seq.append(sequence)
                        else:
                            is_save_image=False
                        anno_frame_list=[]
                        n_interaction=1
                        eval_global_map_tmp_dic={}
                        local_map_dics=({},{})
                        total_frame_num=total_frame_num_dic[sequence]
                        obj_nums = seq_dict[sequence][-1]
                        eval_data_manager=DAVIS2017_Test_Manager(split='val',root=cfg.DATA_ROOT,transform=tr.ToTensor(),
                                                                seq_name=sequence)

                    else:
                        n_interaction+=1
                    ##
                    inter_file.write(sequence+' '+'interaction'+str(n_interaction)+' '+'frame'+str(start_annotated_frame)+'\n')
                    ##



                    ##########################Reference image process
                    #########
                    scr_f = start_annotated_frame
                    anno_frame_list.append(start_annotated_frame)
                    print(start_annotated_frame)
                    scr_f = str(scr_f)
                    while len(scr_f)!=5:
                        scr_f='0'+scr_f
                    ref_img = os.path.join(cfg.DATA_ROOT,'JPEGImages/480p',sequence,scr_f+'.jpg')
                    ref_img = cv2.imread(ref_img)
                    h_,w_ = ref_img.shape[:2]
                    ref_img = np.array(ref_img,dtype=np.float32)

                    #ref_img = tr.ToTensor()(ref_img)
                    #ref_img = ref_img.unsqueeze(0)

                    img_sample = {'ref_img':ref_img}
                    img_sample = tr.ToTensor()(img_sample)
                    ref_img = img_sample['ref_img']

                    ref_img = ref_img.unsqueeze(0)

                    ref_img= ref_img.cuda()

                    ######
                    ####ref_frame_embedding = model.extract_feature(ref_img)

                    if first_scribble:
                        ref_frame_embedding,x1,x2,x3 = model.extract_feature(ref_img)
                        _,channel,emb_h,emb_w = ref_frame_embedding.size()
                        embedding_memory=torch.zeros((total_frame_num,channel,emb_h,emb_w))
                        embedding_memory = embedding_memory.cuda()
                        embedding_memory[start_annotated_frame]=ref_frame_embedding[0]
                        
                    else:
                        ref_frame_embedding = embedding_memory[start_annotated_frame]
                        ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
                    
                    _,x1,x2,x3 = model.extract_feature(ref_img)
                    edge,edge_fea = model.edge_layer(x1,x2,x3)
                    if cfg.MODEL_USE_EDGE_3:
                        input_edge=edge_fea
                    elif cfg.MODEL_USE_EDGE_2 or cfg.MODEL_USE_EDGE:
                        input_edge=edge
                    else:
                        input_edge=None
                   
                    ####
                    print('show_edge')
                    show_edge = edge.squeeze().cpu().numpy()
                    show_edge = Image.fromarray(show_edge.astype('uint8')).convert('L')
#                    show_edge.putpalette(_palette)                        
                    if not os.path.exists(os.path.join(cfg.RESULT_ROOT,sequence)):
                        os.makedirs(os.path.join(cfg.RESULT_ROOT,sequence))
                    show_edge.save(os.path.join(cfg.RESULT_ROOT,sequence,'edge_interactive'+str(n_interaction)+'_'+scr_f+'.png'))
                    ########
                    scribble_masks=scribbles2mask(scribbles,(emb_h,emb_w))
                    scribble_label=scribble_masks[start_annotated_frame]
                    scribble_sample = {'scribble_label':scribble_label}
                    scribble_sample=tr.ToTensor()(scribble_sample)
                    scribble_label = scribble_sample['scribble_label']

                    scribble_label =scribble_label.unsqueeze(0)

                    ######
                    if is_save_image:
                        ref_scribble_to_show = scribble_label.cpu().squeeze().numpy()
                        im_ = Image.fromarray(ref_scribble_to_show.astype('uint8')).convert('P')
                        im_.putpalette(_palette)
                        ref_img_name= scr_f
    
                        if not os.path.exists(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction))):
                            os.makedirs(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction)))
                        im_.save(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction),'inter_'+ref_img_name+'.png'))

                    scribble_label = scribble_label.cuda()

                    #######
                    if first_scribble:

                        prev_label= None
                        prev_label_storage=torch.zeros(104,h_,w_)
                        prev_label_storage=prev_label_storage.cuda()
                        
                        
                    else:
                        prev_label = prev_label_storage[start_annotated_frame]
                        prev_label = prev_label.unsqueeze(0).unsqueeze(0)
                        #print('prev_label_storage:{}'.format(prev_label_storage[:seq_imgnum_dict_[sequence]].size()))
#                    scribble_label_=torch.nn.functional.interpolate(scribble_label.float(),size=(emb_h,emb_w),mode='nearest')
                    if not first_scribble and torch.unique(scribble_label).size(0)==1:
                        print(torch.unique(scribble_label_))
                        final_masks=prev_label_storage[:seq_imgnum_dict_[sequence]]
                        sess.submit_masks(final_masks.cpu().numpy())
                        
                        
                    else:     
                                           # prev_label = os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction-1),scr_f+'.png')
                           # prev_label = Image.open(prev_label)
                           # prev_label = np.array(prev_label,dtype=np.uint8)
                           # prev_label = tr.ToTensor()({'label':prev_label})
                           # prev_label = prev_label['label'].unsqueeze(0)
                           # prev_label = prev_label.cuda()
                        ######
                        ###############
                        #tmp_dic, eval_global_map_tmp_dic= model.before_seghead_process(ref_frame_embedding,ref_frame_embedding,
                        #                            ref_frame_embedding,scribble_label,prev_label,
                        #                            normalize_nearest_neighbor_distances=True,use_local_map=False,
                        #                        seq_names=[sequence],gt_ids=torch.Tensor([obj_nums]),k_nearest_neighbors=cfg.KNNS,
                        #                        global_map_tmp_dic=eval_global_map_tmp_dic,frame_num=[start_annotated_frame],dynamic_seghead=model.dynamic_seghead)
                        tmp_dic,local_map_dics = model.int_seghead(ref_frame_embedding=ref_frame_embedding,ref_scribble_label=scribble_label,prev_round_label=prev_label,input_edge=input_edge,
                                global_map_tmp_dic=eval_global_map_tmp_dic,local_map_dics=local_map_dics,interaction_num=n_interaction,
                                seq_names=[sequence],gt_ids=torch.Tensor([obj_nums]),frame_num=[start_annotated_frame],first_inter=first_scribble)
                        pred_label = tmp_dic[sequence]
                        pred_label = nn.functional.interpolate(pred_label,size=(h_,w_),mode = 'bilinear',align_corners=True)    
    
                        pred_label=torch.argmax(pred_label,dim=1)
                        pred_masks.append(pred_label.float())
                        ########
                        prev_label_storage[start_annotated_frame]=pred_label

                        #######
                            ####
                        if is_save_image:
                            pred_label_to_save=pred_label.squeeze(0).cpu().numpy()
                            im = Image.fromarray(pred_label_to_save.astype('uint8')).convert('P')
                            im.putpalette(_palette)
                            imgname = str(start_annotated_frame)
                            while len(imgname)<5:
                                imgname='0'+imgname
                            if not os.path.exists(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction))):
                                os.makedirs(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction)))
                            im.save(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction),imgname+'.png'))
                        #######################################
                        if first_scribble:
                            scribble_label=rough_ROI(scribble_label)

                        ##############################
                        ref_prev_label = pred_label.unsqueeze(0)
                        prev_label = pred_label.unsqueeze(0)
                        prev_img = ref_img
                        prev_embedding = ref_frame_embedding
                        ####
                        s_time=time.time()
                        ####
                        for ii in range(start_annotated_frame+1,total_frame_num):
                            print('evaluating sequence:{} frame:{}'.format(sequence,ii))
                            sample = eval_data_manager.get_image(ii)
                            img = sample['img']
                            img = img.unsqueeze(0)
                            _,_,h,w = img.size()
                            img = img.cuda()
                        #    current_embedding= model.extract_feature(img)
                            if first_scribble:
                                current_embedding,x1,x2,x3= model.extract_feature(img)
                                embedding_memory[ii]=current_embedding[0]
####
                            
                            else:
                                current_embedding=embedding_memory[ii]
                                current_embedding = current_embedding.unsqueeze(0)
                                 
    
                            ######
                            t2 =time.time()
                            ######
                            prev_img =prev_img.cuda()
                            prev_label =prev_label.cuda()
                            tmp_dic, eval_global_map_tmp_dic,local_map_dics= model.before_seghead_process(ref_frame_embedding,prev_embedding,
                                                    current_embedding,scribble_label,prev_label,
                                                    normalize_nearest_neighbor_distances=True,use_local_map=True,
                                                seq_names=[sequence],gt_ids=torch.Tensor([obj_nums]),k_nearest_neighbors=cfg.KNNS,
                                                global_map_tmp_dic=eval_global_map_tmp_dic,local_map_dics=local_map_dics,
                                                interaction_num=n_interaction,start_annotated_frame=start_annotated_frame,
                                                frame_num=[ii],dynamic_seghead=model.dynamic_seghead)
                            pred_label = tmp_dic[sequence]
                            pred_label = nn.functional.interpolate(pred_label,size=(h,w),mode = 'bilinear',align_corners=True)    
    
                            pred_label=torch.argmax(pred_label,dim=1)
                            pred_masks.append(pred_label.float())
                            prev_label = pred_label.unsqueeze(0)
                            prev_img = img
                            prev_embedding = current_embedding
                            #####
                            prev_label_storage[ii]=pred_label
                            #####
                            f_time=time.time()
                            print('seg head time:{}'.format(f_time-t2))
                            print('total_time:{}'.format(f_time-s_time))
                            s_time=time.time()
                            #####
                            ####
                            if is_save_image:
                                pred_label_to_save=pred_label.squeeze(0).cpu().numpy()
                                im = Image.fromarray(pred_label_to_save.astype('uint8')).convert('P')
                                im.putpalette(_palette)
                                imgname = str(ii)
                                while len(imgname)<5:
                                    imgname='0'+imgname
                                if not os.path.exists(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction))):
                                    os.makedirs(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction)))
                                im.save(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction),imgname+'.png'))
                        #######################################
                        prev_label = ref_prev_label
                        prev_img = ref_img
                        prev_embedding = ref_frame_embedding
                        #######
                        
                        #######
                        ######################################
                        for ii in range(start_annotated_frame):
                            current_frame_num=start_annotated_frame-1-ii
                            print('evaluating sequence:{} frame:{}'.format(sequence,current_frame_num))
                            sample = eval_data_manager.get_image(current_frame_num)
                            img = sample['img']
                            img = img.unsqueeze(0)
                            _,_,h,w = img.size()
                            img = img.cuda()
    
                            #current_embedding= model.extract_feature(img)
                            if first_scribble:
                                current_embedding,x1,x2,x3= model.extract_feature(img)
                                embedding_memory[current_frame_num]=current_embedding[0]
####
                            else:
                                current_embedding = embedding_memory[current_frame_num]
                                current_embedding = current_embedding.unsqueeze(0)
                            prev_img =prev_img.cuda()
                            prev_label =prev_label.cuda()
                            tmp_dic, eval_global_map_tmp_dic,local_map_dics= model.before_seghead_process(ref_frame_embedding,prev_embedding,
                                                    current_embedding,scribble_label,prev_label,
                                                    normalize_nearest_neighbor_distances=True,use_local_map=True,
                                                seq_names=[sequence],gt_ids=torch.Tensor([obj_nums]),k_nearest_neighbors=cfg.KNNS,
                                                global_map_tmp_dic=eval_global_map_tmp_dic,local_map_dics=local_map_dics,interaction_num=n_interaction,start_annotated_frame=start_annotated_frame,frame_num=[current_frame_num],dynamic_seghead=model.dynamic_seghead)
                            pred_label = tmp_dic[sequence]
                            pred_label = nn.functional.interpolate(pred_label,size=(h,w),mode = 'bilinear',align_corners=True)    
    
                            pred_label=torch.argmax(pred_label,dim=1)
                            pred_masks_reverse.append(pred_label.float())
                            prev_label = pred_label.unsqueeze(0)
                            prev_img = img
                            prev_embedding = current_embedding
                            ####
                            prev_label_storage[current_frame_num]=pred_label
                            ###
                            if is_save_image:
                                pred_label_to_save=pred_label.squeeze(0).cpu().numpy()
                                im = Image.fromarray(pred_label_to_save.astype('uint8')).convert('P')
                                im.putpalette(_palette)
                                imgname = str(current_frame_num)
                                while len(imgname)<5:
                                    imgname='0'+imgname
                                if not os.path.exists(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction))):
                                    os.makedirs(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction)))
                                im.save(os.path.join(cfg.RESULT_ROOT,sequence,'interactive'+str(n_interaction),imgname+'.png'))
                        pred_masks_reverse.reverse()
                        pred_masks_reverse.extend(pred_masks)
                        final_masks = torch.cat(pred_masks_reverse,0)
                        #print('final_masks:{}'.format(final_masks.size()))
                        sess.submit_masks(final_masks.cpu().numpy())
            
                t_end = timeit.default_timer()
                print('Total time for single interaction: ' + str(t_end - t_total))
            report = sess.get_report()  
            summary = sess.get_global_summary(save_file=os.path.join(report_save_dir, 'summary.json'))              
    inter_file.close()
def rough_ROI(ref_scribble_labels):
    #### b*1*h*w
    dist=20
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








def load_network(net,pretrained_dict):

        #pretrained_dict = pretrained_dict
    model_dict = net.state_dict()
           # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

_palette=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]


if __name__=='__main__':
    main()
