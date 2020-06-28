from IntVOS import nearest_neighbor_features_per_object,local_previous_frame_nearest_neighbor_features_per_object

import torch
a = torch.rand(104,104,100)
b = torch.rand(104,104,100)
label = torch.ones(104,104,1)
label[0] = label[0]*2
label[1] = label[1]*3
label =label.int()
gt_id = torch.Tensor([3])
gt_id = gt_id.int()
gt_ids = torch.Tensor([1,2,3])
gt_ids = gt_ids.int()
#nn  = local_previous_frame_nearest_neighbor_features_per_object(a,b,label,gt_ids,max_distance=15)
mm,rr = nearest_neighbor_features_per_object(a,b,label,1,gt_id,n_chunks=100)
#print(nn.size())
print(mm.size())
#print(rr)


