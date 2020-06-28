import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import cv2
import sys
sys.path.append("..")
from networks.sync_batchnorm import SynchronizedBatchNorm2d
from config import cfg
import time
import torch.nn.functional as F
from PIL import Image
##TODO
###mask_damaging



#############################################################GLOBAL_DIST_MAP
USE_CORRELATION_COST = False
MODEL_UNFOLD=False
WRONG_LABEL_PADDING_DISTANCE = 1e20
if USE_CORRELATION_COST:
    #from correlation_package.correlation import Correlation
    from spatial_correlation_sampler import SpatialCorrelationSampler



def _pairwise_distances(x,y):
    """Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: Tensor of shape [n, feature_dim].
    y: Tensor of shape [m, feature_dim].
    Returns:
    Float32 distances tensor of shape [n, m].
    """

    xs = torch.sum(x*x,1)
    ys = torch.sum(y*y,1)
    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(0)
    d = xs+ys -2.*torch.matmul(x,torch.t(y))
    return d

##################
def _flattened_pairwise_distances(reference_embeddings, query_embeddings):
    """Calculates flattened tensor of pairwise distances between ref and query.
    Args:
    reference_embeddings: Tensor of shape [..., embedding_dim],
      the embedding vectors for the reference frame
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    Returns:
    A distance tensor of shape [reference_embeddings.size / embedding_dim,
    query_embeddings.size / embedding_dim]
    """
    embedding_dim = query_embeddings.size()[-1]
    reference_embeddings = reference_embeddings.view(-1, embedding_dim)
    first_dim = -1
    query_embeddings = query_embeddings.view(first_dim,embedding_dim)
    dists = _pairwise_distances(query_embeddings, reference_embeddings)
    return dists


def _nn_features_per_object_for_chunk(
    reference_embeddings, query_embeddings, wrong_label_mask,
    k_nearest_neighbors):
    """Extracts features for each object using nearest neighbor attention.
  Args:
    reference_embeddings: Tensor of shape [n_chunk, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [m_chunk, embedding_dim], the embedding
      vectors for the query frames.
    wrong_label_mask:
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m_chunk, n_objects, feature_dim].
    """
    reference_embeddings_key = reference_embeddings
    query_embeddings_key = query_embeddings
    dists = _flattened_pairwise_distances(reference_embeddings_key,query_embeddings_key)
    dists = (torch.unsqueeze(dists,1) +
            torch.unsqueeze(wrong_label_mask.float(),0) *
           WRONG_LABEL_PADDING_DISTANCE)
    if k_nearest_neighbors == 1:
        features,_= torch.min(dists,2,keepdim=True)
    else:
        dists,_=torch.topk(-dists,k=k_nearest_neighbors,dim=2)
        dists = -dists
        valid_mask = (dists<WRONG_LABEL_PADDING_DISTANCE)
        masked_dists = dists * valid_mask.float()
        pad_dist = torch.max(masked_dists, dim=2,keepdim=True)[0].repeat((1,1,masked_dists.size()[-1]))
        dists = torch.where(valid_mask, dists, pad_dist)
         # take mean of distances
        features = torch.mean(dists, dim=2, keepdim=True)


    return features

###
def _selected_pixel(ref_labels_flat,ref_emb_flat):
    index_list = torch.arange(len(ref_labels_flat))
    index_list=index_list.cuda()
    index_ = torch.masked_select(index_list,ref_labels_flat!=-1)
    
    index_=index_.long()
    ref_labels_flat=torch.index_select(ref_labels_flat,0,index_)
    ref_emb_flat=torch.index_select(ref_emb_flat,0,index_)

    return ref_labels_flat,ref_emb_flat
###


def _nearest_neighbor_features_per_object_in_chunks(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
    ref_obj_ids, k_nearest_neighbors, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
    reference_embeddings_flat: Tensor of shape [n, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings_flat: Tensor of shape [m, embedding_dim], the embedding
      vectors for the query frames.
    reference_labels_flat: Tensor of shape [n], the class labels of the
      reference frame.
    ref_obj_ids: int tensor of unique object ids in the reference labels.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m, n_objects, feature_dim].
    """
    chunk_size = int(np.ceil(float(query_embeddings_flat.size()[0])
                                        / n_chunks))
#    reference_labels_flat,reference_embeddings_flat=_selected_pixel(reference_labels_flat,reference_embeddings_flat)
    wrong_label_mask = (reference_labels_flat!=torch.unsqueeze(ref_obj_ids,1))
    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        features = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, query_embeddings_flat_chunk,
            wrong_label_mask, k_nearest_neighbors
            )
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = torch.cat(all_features, dim=0)
    return nn_features


def nearest_neighbor_features_per_object(
    reference_embeddings, query_embeddings, reference_labels,
     k_nearest_neighbors, gt_ids=None, n_chunks=100):
    """Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
    reference_embeddings: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    reference_labels: Tensor of shape [height, width, 1], the class labels of
      the reference frame.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling,
      or 0 for no subsampling.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    gt_ids: Int tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame. If None, it will be derived from
      reference_labels.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [n_query_images, height, width, n_objects, feature_dim].
    gt_ids: An int32 tensor of the unique sorted object ids present
      in the reference labels.
    """
    
    assert (reference_embeddings.size()[:2]==reference_labels.size()[:2])
    h,w,_ =query_embeddings.size()
    reference_labels_flat = reference_labels.view(-1)
    if gt_ids is None:
        ref_obj_ids = torch.unique(reference_labels_flat)[-1]
        ref_obj_ids = np.arange(0,ref_obj_ids.cpu()+1)
        gt_ids = torch.from_numpy(ref_obj_ids)
        gt_ids = gt_ids.int()
        if torch.cuda.is_available():
            gt_ids = gt_ids.cuda()
    else:
        gt_ids = gt_ids.cpu()
        gt_ids = np.arange(0,gt_ids+1)

        gt_ids = torch.from_numpy(gt_ids)
        gt_ids = gt_ids.int()
        if torch.cuda.is_available():
            gt_ids = gt_ids.cuda()
    embedding_dim = query_embeddings.size()[-1]
    query_embeddings_flat = query_embeddings.view(-1,embedding_dim)
    reference_embeddings_flat = reference_embeddings.view(-1,embedding_dim)
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        gt_ids, k_nearest_neighbors, n_chunks)
    nn_features_dim = nn_features.size()[-1]
    nn_features_reshape=nn_features.view(1,h,w,gt_ids.size(0),nn_features_dim)
    return nn_features_reshape, gt_ids
########################################################################LOCAL_DIST_MAP
def local_pairwise_distances(x, y, max_distance=9):
    """Computes pairwise squared l2 distances using a local search window.
    Optimized implementation using correlation_cost.
    Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
    Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].

    """
    if cfg.MODEL_LOCAL_DOWNSAMPLE:
            #####
        ori_h,ori_w,_=x.size()
        x = x.permute(2,0,1).unsqueeze(0)

        x = F.avg_pool2d(x,(2,2),(2,2))
        y= y.permute(2,0,1).unsqueeze(0)
        y = F.avg_pool2d(y,(2,2),(2,2))

        x =x.squeeze(0).permute(1,2,0)
        y = y.squeeze(0).permute(1,2,0)
        corr = cross_correlate(x, y, max_distance=max_distance)
        xs = torch.sum(x*x,2,keepdim=True)

        ys = torch.sum(y*y,2,keepdim=True)
        ones_ys = torch.ones_like(ys)
        ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
        d = xs + ys - 2 * corr
    # Boundary should be set to Inf.
        tmp = torch.zeros_like(d)
        boundary = torch.eq(
            cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
        d = torch.where(boundary, tmp.fill_(float('inf')),d)
        d = (torch.sigmoid(d)-0.5)*2
        d = d.permute(2,0,1).unsqueeze(0)
        d = F.interpolate(d,size=(ori_h,ori_w),mode='bilinear',align_corners=True)
        d = d.squeeze(0).permute(1,2,0)
    else:
        corr = cross_correlate(x, y, max_distance=max_distance)
        xs = torch.sum(x*x,2,keepdim=True)

        ys = torch.sum(y*y,2,keepdim=True)
        ones_ys = torch.ones_like(ys)
        ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
        d = xs + ys - 2 * corr
        # Boundary should be set to Inf.
        tmp = torch.zeros_like(d)
        boundary = torch.eq(
            cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
        d = torch.where(boundary, tmp.fill_(float('inf')),d)
    return d
def local_pairwise_distances2(x, y, max_distance=9):
    """Computes pairwise squared l2 distances using a local search window.
    Naive implementation using map_fn.
    Used as a slow fallback for when correlation_cost is not available.
    Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
    Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].
    """
    if cfg.MODEL_LOCAL_DOWNSAMPLE:
        ori_h,ori_w,_ = x.size()
        x = x.permute(2,0,1).unsqueeze(0)
        x = F.avg_pool2d(x,(2,2),(2,2))
        y = y.permute(2,0,1).unsqueeze(0)
        y = F.avg_pool2d(y,(2,2),(2,2))

        _,channels,height,width=x.size()
        padding_val=1e20
        padded_y = F.pad(y, (max_distance, max_distance,
            max_distance, max_distance), mode='constant', value=padding_val)
        offset_y = F.unfold(padded_y, kernel_size=(height, width)).view(1, channels, height, width, -1)
        x = x.view(1, channels, height, width, 1)
        minus = x - offset_y
        dists = torch.sum(torch.mul(minus, minus), dim=1).view(1, height, width, -1).permute(0, 3, 1, 2)
        dists = (torch.sigmoid(dists) - 0.5) * 2
        dists = F.interpolate(dists, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        dists = dists.squeeze(0).permute(1, 2, 0)


    else:
        padding_val = 1e20
        padded_y =nn.functional.pad(y,(0,0,max_distance, max_distance,
                              max_distance, max_distance),mode='constant', value=padding_val)
        height, width, _ = x.size()
        dists = []
        for y_start in range(2 * max_distance + 1):
            y_end = y_start + height
            y_slice = padded_y[y_start:y_end]
            for x_start in range(2 * max_distance + 1):
                x_end = x_start + width
                offset_y = y_slice[:, x_start:x_end]
                dist = torch.sum(torch.pow((x-offset_y),2), dim=2)
                dists.append(dist)
        dists = torch.stack(dists, dim=2)

    return dists


def cross_correlate(x, y, max_distance=9):
    """Efficiently computes the cross correlation of x and y.
  Optimized implementation using correlation_cost.
  Note that we do not normalize by the feature dimension.
  Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
  Returns:
    Float32 tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """
    #corr_op=Correlation(pad_size=max_distance, kernel_size=1, max_displacement=max_distance, stride1=1, stride2=1, corr_multiply=1)
    corr_op=SpatialCorrelationSampler(kernel_size=1, patch_size=2*max_distance+1, stride=1,dilation_patch=1,padding=0)

    xs = x.permute(2,0,1)
    xs = torch.unsqueeze(xs,0)
    ys = y.permute(2,0,1)
    ys = torch.unsqueeze(ys,0)
    corr = corr_op(xs,ys)
    #print(corr.size())
    bs,_,_,hh,ww = corr.size()
    corr = corr.view(bs,-1,hh,ww)
    corr = torch.squeeze(corr,0)
    corr = corr.permute(1,2,0)
#    feature_dim=x.size()[-1]
#    corr *= feature_dim
    return corr



def local_previous_frame_nearest_neighbor_features_per_object(
    prev_frame_embedding, query_embedding, prev_frame_labels,
      gt_ids, max_distance=15):
    """Computes nearest neighbor features while only allowing local matches.
  Args:
    prev_frame_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the last frame.
    query_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the query frames.
    prev_frame_labels: Tensor of shape [height, width, 1], the class labels of
      the previous frame.
    gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame.
    max_distance: Integer, the maximum distance allowed for local matching.
  Returns:
    nn_features: A float32 np.array of nearest neighbor features of shape
      [1, height, width, n_objects, 1].
    """

    if USE_CORRELATION_COST:
        
        d = local_pairwise_distances(query_embedding, prev_frame_embedding,
                                   max_distance=max_distance)
        
        #print(d.size())
    else:
        d = local_pairwise_distances2(query_embedding, prev_frame_embedding,
                                    max_distance=max_distance)
#    ttt2=time.time()
#    print(ttt2-ttt1)
#    d_tosave=d.detach().cpu().numpy()
#    np.save('d_corr.npy',d_tosave)
#    exit()
#    d = (torch.sigmoid(d) - 0.5) * 2
    height,width=prev_frame_embedding.size()[:2]

    if USE_CORRELATION_COST:
        #corr_op=Correlation(pad_size=max_distance, kernel_size=1, max_displacement=max_distance, stride1=1, stride2=1, corr_multiply=1)
        corr_op=SpatialCorrelationSampler(kernel_size=1, patch_size=2*max_distance+1, stride=1,dilation_patch=1,padding=0)
        # New, faster code with cross-correlation via correlation_cost.
      # Due to padding we have to add 1 to the labels.
        tmp_prev_frame_labels=(prev_frame_labels+1).float().permute(2,0,1)
        tmp_prev_frame_labels=torch.unsqueeze(tmp_prev_frame_labels,0)
#        print(tmp_prev_frame_labels.size())
        ones_ = torch.ones_like(tmp_prev_frame_labels)


        offset_labels = corr_op(ones_,tmp_prev_frame_labels)
#        print(offset_labels.size())
        bs,_,_,hh,ww=offset_labels.size()
        offset_labels=offset_labels.view(bs,-1,hh,ww)


        offset_labels = torch.squeeze(offset_labels,0)
        offset_labels = offset_labels.permute(1,2,0)
        offset_labels = torch.unsqueeze(offset_labels,3)
        offset_labels = torch.round(offset_labels - 1)
        offset_masks = torch.eq(
          offset_labels,
          gt_ids.float().unsqueeze(0).unsqueeze(0).unsqueeze(0))


    else:
        if MODEL_UNFOLD:

            labels = prev_frame_labels.float().permute(2, 0, 1).unsqueeze(0)
            padded_labels = F.pad(labels,
                                (2 * max_distance, 2 * max_distance,
                                 2 * max_distance, 2 * max_distance,
                                 ))
            offset_labels = F.unfold(padded_labels, kernel_size=(height, width), stride=(2, 2)).view(height, width, -1, 1)
            offset_masks = torch.eq(
                offset_labels,
                gt_ids.float().unsqueeze(0).unsqueeze(0).unsqueeze(0))
        else:

            masks = torch.eq(prev_frame_labels, gt_ids.unsqueeze(0).unsqueeze(0))
            padded_masks = nn.functional.pad(masks,
                            (0,0,max_distance, max_distance,
                             max_distance, max_distance,
                             ))
            offset_masks = []
#            ttt1=time.time()
            for y_start in range(2 * max_distance + 1):
                y_end = y_start + height
                masks_slice = padded_masks[y_start:y_end]
                for x_start in range(2 * max_distance + 1):
                    x_end = x_start + width
                    offset_mask = masks_slice[:, x_start:x_end]
                    offset_masks.append(offset_mask)
#            ttt2=time.time()
 #           print(ttt2-ttt1)
            offset_masks = torch.stack(offset_masks, dim=2)


#    pad = torch.ones((height, width, (2 * max_distance + 1) ** 2, gt_ids.size(0)))

    d_tiled = d.unsqueeze(-1).repeat((1,1,1,gt_ids.size(0)))
    pad = torch.ones_like(d_tiled)

#    ttt1=time.time()

#        d_tiled = d_tiled.cuda()

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists,_ = torch.min(d_masked, dim=2)
    dists = dists.view(1, height, width, gt_ids.size(0), 1)

    return dists
    
##############################################################









class _split_separable_conv2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=7):
        super(_split_separable_conv2d,self).__init__()
        self.conv1=nn.Conv2d(in_dim,in_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2),groups=in_dim)
        self.relu1=nn.ReLU(True)
        self.bn1 = SynchronizedBatchNorm2d(in_dim, momentum=cfg.TRAIN_BN_MOM)
        self.conv2=nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1)
        self.relu2=nn.ReLU(True)
        self.bn2 = SynchronizedBatchNorm2d(out_dim, momentum=cfg.TRAIN_BN_MOM)
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
##################
class Global_Atten(nn.Module):
    def __init__(self,in_dim=(cfg.MODEL_SEMANTIC_EMBEDDING_DIM+1),out_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,kernel_size=7):
        super(Global_Atten,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2))
        self.relu1=nn.ReLU(True)
        self.bn1 = SynchronizedBatchNorm2d(out_dim, momentum=cfg.TRAIN_BN_MOM)
        self.conv2 = nn.Conv2d(out_dim,1,kernel_size=1,stride=1)
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x        

#################
class DynamicSegHead(nn.Module):
    def __init__(self,in_dim=(cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3),embed_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,kernel_size=1,use_edge=False):
        super(DynamicSegHead,self).__init__()
        self.layer1=_split_separable_conv2d(in_dim,embed_dim)
        self.layer2=_split_separable_conv2d(embed_dim,embed_dim)
#        self.layer3=_split_separable_conv2d(embed_dim,embed_dim)
        self.layer4=_split_separable_conv2d(embed_dim,embed_dim)
        if use_edge:
            self.layer3=_split_separable_conv2d(embed_dim,embed_dim)
        else:

            self.layer3=_split_separable_conv2d(embed_dim,embed_dim)
        self.conv=nn.Conv2d(embed_dim,1,1,1)
            
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

################TODO
    def forward(self,x,edge_fea=None):
       
        x = self.layer1(x)
        x = self.layer2(x)
#        x = self.layer3(x)
        if edge_fea is not None:
#            x = torch.cat((x,edge_fea),1)
            x = x + edge_fea
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x

##################
class Edge_Module(nn.Module):

    def __init__(self,in_fea=[256,512,1024], mid_fea=256, out_fea=1):
        super(Edge_Module, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            SynchronizedBatchNorm2d(mid_fea, momentum=cfg.TRAIN_BN_MOM)
            )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            SynchronizedBatchNorm2d(mid_fea, momentum=cfg.TRAIN_BN_MOM)
            )

        self.conv3 =  nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            SynchronizedBatchNorm2d(mid_fea, momentum=cfg.TRAIN_BN_MOM)
            )
        self.conv4 = nn.Conv2d(mid_fea,out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        
        self.conv5 = nn.Conv2d(out_fea*3,out_fea, kernel_size=1, padding=0, dilation=1, bias=True)
        ########
        self.conv6 = nn.Conv2d(mid_fea*3,32,kernel_size=1, padding=0, dilation=1, bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.relu=nn.ReLU(True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)        
        
        edge2_fea =  F.interpolate(edge2_fea, size=(h, w), mode='bilinear',align_corners=True) 
        edge3_fea =  F.interpolate(edge3_fea, size=(h, w), mode='bilinear',align_corners=True) 
        edge2 =  F.interpolate(edge2, size=(h, w), mode='bilinear',align_corners=True)
        edge3 =  F.interpolate(edge3, size=(h, w), mode='bilinear',align_corners=True) 
 
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)
        ########
        edge_fea = self.conv6(edge_fea)
        edge_fea = self.bn(edge_fea)
        edge_fea = self.relu(edge_fea)
         
        return edge, edge_fea     

##################

    ###############
#    return current_frame_embedding,global_match_map,local_match_map,scale_previous_frame_label
class IntVOS(nn.Module):
    def __init__(self,cfg,feature_extracter,dynamic_seghead=DynamicSegHead(),freeze_bn=False):
        super(IntVOS,self).__init__()
        self.feature_extracter=feature_extracter
        self.feature_extracter.cls_conv=nn.Sequential()
        self.feature_extracter.upsample4=nn.Sequential()
        self.semantic_embedding=None
        self.seperate_conv=nn.Conv2d(cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_ASPP_OUTDIM,kernel_size=3,stride=1,padding=1,groups=cfg.MODEL_ASPP_OUTDIM)
        self.bn1 = SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM)
        self.relu1=nn.ReLU(True)
        self.embedding_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_SEMANTIC_EMBEDDING_DIM,1,1)
        self.relu2=nn.ReLU(True)
        self.bn2  = SynchronizedBatchNorm2d(cfg.MODEL_SEMANTIC_EMBEDDING_DIM, momentum=cfg.TRAIN_BN_MOM)
        self.semantic_embedding=nn.Sequential(*[self.seperate_conv,self.bn1,self.relu1,self.embedding_conv,self.bn2,self.relu2])
#        self.semantic_embedding=nn.Sequential(*[self.seperate_conv,self.relu1,self.embedding_conv,self.relu2])

        self.edge_layer = Edge_Module()
        for m in self.semantic_embedding:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
        #########################
        self.dynamic_seghead = DynamicSegHead()
        if cfg.MODEL_GLOBAL_ATTEN:
            self.global_atten=Global_Atten()
        if cfg.MODEL_USE_EDGE:
 
            self.inter_seghead = DynamicSegHead(in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3)
        elif cfg.MODEL_USE_EDGE_3:
            self.inter_seghead = DynamicSegHead(in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3,use_edge=True)

        else:
            self.inter_seghead = DynamicSegHead(in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3)

        if freeze_bn:
            self.freeze_bn()
    def forward(self,x=None,ref_scribble_label=None,previous_frame_mask=None,normalize_nearest_neighbor_distances=True,use_local_map=True,
                seq_names=None,gt_ids=None,k_nearest_neighbors=1,global_map_tmp_dic=None,local_map_dics=None,interaction_num=None,start_annotated_frame=None,
                frame_num=None):

        x,x1,x2,x3=self.extract_feature(x)
        _,_,x1 = torch.split(x1,split_size_or_sections=int(x.size(0)/3), dim=0)
        _,_,x2 = torch.split(x2,split_size_or_sections=int(x.size(0)/3), dim=0)
        _,_,x3 = torch.split(x3,split_size_or_sections=int(x.size(0)/3), dim=0)
        edge,edge_fea = self.edge_layer(x1,x2,x3) 
        if cfg.MODEL_USE_EDGE:
            input_edge=edge
        else:
            input_edge=None
        ref_frame_embedding, previous_frame_embedding,current_frame_embedding = torch.split(x,split_size_or_sections=int(x.size(0)/3), dim=0)

        dic= self.before_seghead_process(ref_frame_embedding,previous_frame_embedding,
                    current_frame_embedding,ref_scribble_label,previous_frame_mask,
                    normalize_nearest_neighbor_distances,use_local_map,
                    seq_names,gt_ids,k_nearest_neighbors,global_map_tmp_dic,local_map_dics,interaction_num,start_annotated_frame,frame_num,self.dynamic_seghead)
        
        return dic, global_map_tmp_dic,edge









    def extract_feature(self,x):
        x,x1,x2,x3=self.feature_extracter(x)
        x=self.semantic_embedding(x)
        return x,x1,x2,x3
    
#    def pred_label(self,x):
    def before_seghead_process(self,ref_frame_embedding=None,previous_frame_embedding=None, current_frame_embedding=None,
                ref_scribble_label=None,previous_frame_mask=None,normalize_nearest_neighbor_distances=True,use_local_map=True,
                seq_names=None,gt_ids=None,k_nearest_neighbors=1,global_map_tmp_dic=None,local_map_dics=None,interaction_num=None,start_annotated_frame=None,
                frame_num=None,dynamic_seghead=None):

        """return: feature_embedding,global_match_map,local_match_map,previous_frame_mask"""
    ###############
#        print('*'*10)
        ####
        ####
        global_map_tmp_dic=global_map_tmp_dic
        dic_tmp={}
        bs,c,h,w = current_frame_embedding.size()
        scale_ref_scribble_label=torch.nn.functional.interpolate(ref_scribble_label.float(),size=(h,w),mode='nearest')
        scale_ref_scribble_label = scale_ref_scribble_label.int()
        scale_previous_frame_label=torch.nn.functional.interpolate(previous_frame_mask.float(),size=(h,w),mode='nearest')
        scale_previous_frame_label=scale_previous_frame_label.int()
        #ref_frame_embedding = ref_frame_embedding.detach()
        #previous_frame_embedding=previous_frame_embedding.detach()
        if USE_CORRELATION_COST:
            n_chunks = 20
        else:
            n_chunks = 500
        ####
        ####
        for n in range(bs):
            seq_current_frame_embedding = current_frame_embedding[n]
            seq_ref_frame_embedding = ref_frame_embedding[n]
            seq_prev_frame_embedding = previous_frame_embedding[n]
            ########################Global dist map

            seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)
            seq_current_frame_embedding = seq_current_frame_embedding.permute(1,2,0)
            seq_ref_scribble_label = scale_ref_scribble_label[n].permute(1,2,0)
            ####
            t2 = time.time()
            ####

            nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(reference_embeddings=seq_ref_frame_embedding, query_embeddings=seq_current_frame_embedding, 
                reference_labels=seq_ref_scribble_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=10)
            if normalize_nearest_neighbor_distances:
                nn_features_n = (torch.sigmoid(nn_features_n) - 0.5) * 2

            ###
            t3=time.time()
#            print('global memory time:{}'.format(t3-t2))
            ###
            if global_map_tmp_dic is not None:
                if seq_names[n] not in global_map_tmp_dic:
                    global_map_tmp_dic[seq_names[n]]=torch.ones_like(nn_features_n).repeat(104,1,1,1,1)
                    if  torch.cuda.is_available():
                        global_map_tmp_dic[seq_names[n]]=global_map_tmp_dic[seq_names[n]].cuda()
                nn_features_n=torch.where(nn_features_n<=global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),nn_features_n,global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))
                
                global_map_tmp_dic[seq_names[n]][frame_num[n]]=nn_features_n.detach()
            
            ######
            t4=time.time()
            ######
            #########################TODO
#            if is_training and damage_initial_previous_frame_mask:
                # Damage the previous frame masks.
###                                                          dilate=False)
    
    
            #########################Local dist map
            seq_prev_frame_embedding = seq_prev_frame_embedding.permute(1,2,0)
            seq_previous_frame_label = scale_previous_frame_label[n].permute(1,2,0)
#            prev_frame_nn_features_n=torch.ones_like(nn_features_n)
#            prev_frame_nn_features_n=prev_frame_nn_features_n.cuda()

            if use_local_map:
                prev_frame_nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding=seq_prev_frame_embedding, 
                            query_embedding=seq_current_frame_embedding, prev_frame_labels=seq_previous_frame_label,
                                gt_ids=ref_obj_ids, max_distance=cfg.MODEL_MAX_LOCAL_DISTANCE)
            else:
               # prev_frame_nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding=seq_prev_frame_embedding, 
               #             query_embedding=seq_current_frame_embedding, prev_frame_labels=seq_previous_frame_label,
               #                 gt_ids=ref_obj_ids, max_distance=15)
                prev_frame_nn_features_n,_ = nearest_neighbor_features_per_object(reference_embeddings=seq_prev_frame_embedding, query_embeddings=seq_current_frame_embedding, 
                reference_labels=seq_previous_frame_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=20)
                prev_frame_nn_features_n = (torch.sigmoid(prev_frame_nn_features_n) - 0.5) * 2

            ###########
            t5=time.time()
#            print('local time:{}'.format(t5-t4))
            ###########

            #local_map_tmp_dic,local_map_dist_dic=local_map_dics
            #if seq_names[n] not in local_map_tmp_dic:
            #    local_map_tmp_dic[seq_names[n]]=torch.ones_like(prev_frame_nn_features_n).repeat(104,1,1,1,1)
            #    if  torch.cuda.is_available():
            #        local_map_tmp_dic[seq_names[n]]=local_map_tmp_dic[seq_names[n]].cuda()
            #prev_frame_nn_features_n=torch.where(prev_frame_nn_features_n<=local_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),prev_frame_nn_features_n,local_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))
            #
            #local_map_tmp_dic[seq_names[n]][frame_num[n]]=prev_frame_nn_features_n.detach()
            #local_map_dics=(local_map_tmp_dic,local_map_dist_dic)
            #
            #############
            if local_map_dics is not None:
                local_map_tmp_dic,local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    local_map_dist_dic[seq_names[n]] = torch.zeros(104,9)
                    if torch.cuda.is_available():
                        local_map_dist_dic[seq_names[n]]=local_map_dist_dic[seq_names[n]].cuda()
                if seq_names[n] not in local_map_tmp_dic:
                    local_map_tmp_dic[seq_names[n]] = torch.zeros_like(prev_frame_nn_features_n).unsqueeze(0).repeat(104,9,1,1,1,1)
                    if torch.cuda.is_available():
                        local_map_tmp_dic[seq_names[n]] = local_map_tmp_dic[seq_names[n]].cuda()
                local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]=1.0/(abs(frame_num[n]-start_annotated_frame))
                local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]=prev_frame_nn_features_n.squeeze(0).detach()
            #    prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n],:interaction_num]*local_map_dist_dic[seq_names[n]][frame_num[n],:interaction_num].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/(torch.sum(local_map_dist_dic[seq_names[n]][frame_num[n]][:interaction_num]))
            #    prev_frame_nn_features_n = torch.mean(prev_frame_nn_features_n,dim=0,keepdim=True)
                if interaction_num==1:
                    prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]
                    prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
                else:
#                    if local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]>local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-2]:
#                        prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]
#                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
#                    else:
#                        prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-2]
#                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
                    _,idx = torch.topk(local_map_dist_dic[seq_names[n]][frame_num[n]][:interaction_num],1)
                    prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][idx]
                        
                #####
                #######
                local_map_dics=(local_map_tmp_dic,local_map_dist_dic)
    
           ##########################
            previous_frame_to_cat = (seq_previous_frame_label.float()== ref_obj_ids.float())
    
            to_cat_current_frame_embedding = current_frame_embedding[n].unsqueeze(0).repeat((ref_obj_ids.size(0),1,1,1))
             
            to_cat_nn_feature_n = nn_features_n.squeeze(0).permute(2,3,0,1)
            to_cat_previous_frame = previous_frame_to_cat.unsqueeze(-1).permute(2,3,0,1).float()
            to_cat_prev_frame_nn_feature_n = prev_frame_nn_features_n.squeeze(0).permute(2,3,0,1)
            to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_prev_frame_nn_feature_n,to_cat_previous_frame),1)
         #   to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_prev_frame_nn_feature_n),1)
          #  to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n),1)
            if cfg.MODEL_GLOBAL_ATTEN:
                cat_global_=torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n),1)
                atten_maps = self.global_atten(cat_global_)
                atten_maps = torch.nn.functional.softmax(atten_maps,0)
                pred_=dynamic_seghead(to_cat,atten_maps)
            else:
                pred_=dynamic_seghead(to_cat)
            pred_ = pred_.permute(1,0,2,3)
            dic_tmp[seq_names[n]]=pred_

            ###
#            print('*'*10)
            ###
        if local_map_dics is None:
            return dic_tmp
        else:
            return dic_tmp,global_map_tmp_dic,local_map_dics

    def int_seghead(self,ref_frame_embedding=None,ref_scribble_label=None,prev_round_label=None,input_edge=None,normalize_nearest_neighbor_distances=True,global_map_tmp_dic=None,
                local_map_dics=None,interaction_num=None,seq_names=None,gt_ids=None,k_nearest_neighbors=1,frame_num=None,first_inter=True):
        dic_tmp={}
        if input_edge is not None and cfg.MODEL_USE_EDGE_3==False:
            input_edge = torch.sigmoid(input_edge)
#            input_edge = (input_edge>0.5)
        bs,c,h,w = ref_frame_embedding.size()
        scale_ref_scribble_label=torch.nn.functional.interpolate(ref_scribble_label.float(),size=(h,w),mode='nearest')
        scale_ref_scribble_label = scale_ref_scribble_label.int()
        if not first_inter:
            scale_prev_round_label=torch.nn.functional.interpolate(prev_round_label.float(),size=(h,w),mode='nearest')
            scale_prev_round_label = scale_prev_round_label.int()
        if USE_CORRELATION_COST:
            n_chunks = 20
        else:
            n_chunks = 500
        for n in range(bs):

            gt_id = torch.arange(0,gt_ids[n]+1)

#            gt_id = torch.from_numpy(gt_id)
            gt_id = gt_id.int()
            if torch.cuda.is_available():
                gt_id = gt_id.cuda()

            seq_ref_frame_embedding = ref_frame_embedding[n]

            ########################Global dist map
            seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)
            seq_ref_scribble_label = scale_ref_scribble_label[n].permute(1,2,0)
#            if first_inter:
            nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding=seq_ref_frame_embedding, 
                            query_embedding=seq_ref_frame_embedding, prev_frame_labels=seq_ref_scribble_label,
                                gt_ids=gt_id, max_distance=cfg.MODEL_MAX_LOCAL_DISTANCE)
#            else:

#                nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(reference_embeddings=seq_ref_frame_embedding, query_embeddings=seq_ref_frame_embedding,reference_labels=seq_ref_scribble_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=20)
#                nn_features_n = (torch.sigmoid(nn_features_n) - 0.5) * 2

            
#            nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(reference_embeddings=seq_ref_frame_embedding, query_embeddings=seq_ref_frame_embedding, 
#                reference_labels=seq_ref_scribble_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=20)
#            if normalize_nearest_neighbor_distances:
#                nn_features_n = (torch.sigmoid(nn_features_n) - 0.5) * 2
            ######################Global map update
            if seq_names[n] not in global_map_tmp_dic:
                global_map_tmp_dic[seq_names[n]]=torch.ones_like(nn_features_n).repeat(104,1,1,1,1)
                if  torch.cuda.is_available():
                    global_map_tmp_dic[seq_names[n]]=global_map_tmp_dic[seq_names[n]].cuda()
            nn_features_n_=torch.where(nn_features_n<=global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),nn_features_n,global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))
            
            global_map_tmp_dic[seq_names[n]][frame_num[n]]=nn_features_n_.detach()
            ##################Local map update
            ######

            #local_map_tmp_dic,local_map_dist_dic=local_map_dics
            #if seq_names[n] not in local_map_tmp_dic:
            #    local_map_tmp_dic[seq_names[n]]=torch.ones_like(nn_features_n).repeat(104,1,1,1,1)
            #    if  torch.cuda.is_available():
            #        local_map_tmp_dic[seq_names[n]]=local_map_tmp_dic[seq_names[n]].cuda()
            #local_map_dics=(local_map_tmp_dic,local_map_dist_dic)
            ######
            if local_map_dics is not None:
                local_map_tmp_dic,local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    local_map_dist_dic[seq_names[n]] = torch.zeros(104,9)
                    if torch.cuda.is_available():
                        local_map_dist_dic[seq_names[n]]=local_map_dist_dic[seq_names[n]].cuda()
                if seq_names[n] not in local_map_tmp_dic:
                    local_map_tmp_dic[seq_names[n]] = torch.ones_like(nn_features_n).unsqueeze(0).repeat(104,9,1,1,1,1)
                    if torch.cuda.is_available():
                        local_map_tmp_dic[seq_names[n]] = local_map_tmp_dic[seq_names[n]].cuda()
    #            local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]=0
                    

                local_map_dics=(local_map_tmp_dic,local_map_dist_dic)



            ##################

        #    for iii in ref_obj_ids:
        #        if iii not in torch.unique(seq_ref_scribble_label):
        #            nn_features_n[:,:,:,iii,:]=0.5
            to_cat_current_frame_embedding = ref_frame_embedding[n].unsqueeze(0).repeat((gt_id.size(0),1,1,1))
            to_cat_nn_feature_n = nn_features_n.squeeze(0).permute(2,3,0,1)
            

            scribble_mask_to_cat = (seq_ref_scribble_label.float()== gt_id.float())
            to_cat_scribble_mask_to_cat = scribble_mask_to_cat.unsqueeze(-1).permute(2,3,0,1).float()
            if not first_inter:
                seq_prev_round_label = scale_prev_round_label[n].permute(1,2,0)

                prev_round_to_cat = (seq_prev_round_label.float()==gt_id.float())
                to_cat_prev_round_to_cat = prev_round_to_cat.unsqueeze(-1).permute(2,3,0,1).float()
            else:
                to_cat_prev_round_to_cat = torch.zeros_like(to_cat_scribble_mask_to_cat)
                to_cat_prev_round_to_cat[0]=1.
            #    to_cat_prev_round_to_cat = to_cat_prev_round_to_cat*0.5

            #to_cat_background_feature = torch.cuda.FloatTensor()
            #for obj in gt_id:
            #    index = torch.masked_select(gt_id,gt_id!=obj)
            #    index = index.long()
            #    print(index)
            #    others = torch.index_select(to_cat_scribble_mask_to_cat,0,index)
            #    others,_ = torch.max(others,0,True)
            #    to_cat_background_feature=torch.cat((to_cat_background_feature,others),0)
            ##############Local dist map
            if cfg.MODEL_USE_EDGE_2:
               
                to_cat_edge = input_edge[n].unsqueeze(0).repeat((gt_id.size(0),1,1,1))
                ###
                round_=cfg.USE2ROUND
                to_cat_edge=(to_cat_edge>0.5)
                ###
                to_cat_edge = 0-to_cat_edge
                
                to_cat_scribble_mask_to_cat=torch.where(to_cat_scribble_mask_to_cat.float()==1.,to_cat_scribble_mask_to_cat.float(),to_cat_edge.float())
                to_cat = to_cat_scribble_mask_to_cat
                #####
                for i in range(round_):
                    for j in range(gt_id.size(0)):
                        ind=to_cat[j]
                        ind=ind.squeeze(0)
                        ind[:,1:] = torch.where((ind[:,1:]==0) * (ind[:,:-1]==1),torch.cuda.FloatTensor([1]),ind[:,1:])
                        ind[:,:-1] = torch.where((ind[:,1:]==1) * (ind[:,:-1]==0),torch.cuda.FloatTensor([1]),ind[:,:-1])
                        ind[1:,:] = torch.where((ind[1:,:]==0) * (ind[:-1,:]==1),torch.cuda.FloatTensor([1]),ind[1:,:])
                        ind[:-1,:] = torch.where((ind[1:,:]==1) * (ind[:-1,:]==0),torch.cuda.FloatTensor([1]),ind[:-1,:])
                        to_cat[j,0,:,:]=ind
                to_fill = torch.zeros_like(to_cat_scribble_mask_to_cat)
                to_cat_scribble_mask_to_cat = torch.where(to_cat.float()==1.,to_cat.float(),to_fill.float())
                                 

                #to_show = to_cat_scribble_mask_to_cat[2].squeeze(0).cpu().numpy()
                ##print(to_show.shape)
                #im = Image.fromarray(to_show.astype('uint8')).convert('P')
                #im.putpalette(_palette)
                #im.save(os.path.join(cfg.RESULT_ROOT, seq_names[n]+'_inter'+str(interaction_num)+'_frame'+str(frame_num[n])+'_showuse2.png'))
                             
                        
                #####
                    
                to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_scribble_mask_to_cat,to_cat_prev_round_to_cat),1)
                
                
                 
            
            elif cfg.MODEL_USE_EDGE:
                to_cat_edge=input_edge[n].unsqueeze(0).repeat((gt_id.size(0),1,1,1))
                ####
            #    ind = to_cat_edge>0.5 
############    ####
                to_cat_edge = 0-to_cat_edge
                to_scribble =to_cat_scribble_mask_to_cat
                to_cat_scribble_mask_to_cat=torch.where(to_cat_scribble_mask_to_cat.float()==1.,to_cat_scribble_mask_to_cat.float(),to_cat_edge.float())
                filter_= torch.zeros_like(to_cat_scribble_mask_to_cat)
                to_fill = torch.zeros_like(to_cat_scribble_mask_to_cat)
                dist=cfg.USE1DIST
                for i in range(gt_id.size(0)):

                    no_background = (to_scribble[i]==1)
                    no_background = no_background.squeeze(0)
                    if no_background.sum()<1:
                        continue
                    
                    
                    no_b=no_background.nonzero()
                     
                    (h_min,w_min),_ = torch.min(no_b,0)
                    (h_max,w_max),_ = torch.max(no_b,0)


                    filter_[i,0,max(h_min-dist,0):min(h_max+dist,h-1),max(w_min-dist,0):min(w_max+dist,w-1)]=1

                to_cat_scribble_mask_to_cat = torch.where(filter_.byte(),to_cat_scribble_mask_to_cat,to_fill)
#                print(to_cat_scribble_mask_to_cat.max())
#                print(to_cat_scribble_mask_to_cat.min())
                #to_show = to_cat_scribble_mask_to_cat[1].squeeze(0).cpu().numpy()
                ##print(to_show.shape)
                #im = Image.fromarray(to_show.astype('uint8')).convert('P')
                #im.putpalette(_palette)
                #im.save(os.path.join(cfg.RESULT_ROOT,'show_use1.png'))
               
               
                #print(to_cat_edge.size())
                to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_scribble_mask_to_cat,to_cat_prev_round_to_cat),1)

            else:


                to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_scribble_mask_to_cat,to_cat_prev_round_to_cat),1)
            #to_cat = torch.cat((to_cat_current_frame_embedding,to_cat_scribble_mask_to_cat,to_cat_prev_round_to_cat),1)
            if cfg.MODEL_USE_EDGE_3:
                pred_ = self.inter_seghead(to_cat,input_edge[n].unsqueeze(0).repeat((gt_id.size(0),1,1,1)))
            else:
                pred_ = self.inter_seghead(to_cat)
      
            pred_ = pred_.permute(1,0,2,3)
            dic_tmp[seq_names[n]]=pred_
        if local_map_dics is None:
            return dic_tmp
        else:
            return dic_tmp,local_map_dics










    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

_palette=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
