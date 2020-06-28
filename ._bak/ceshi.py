from IntVOS import nearest_neighbor_features_per_object

import torch
a = torch.rand(154,152,100)
b = torch.rand(154,152,100)
label = torch.ones(154,152,1)
label[0] = label[0]*2
label[1] = label[1]*3
label =label.int()
gt_ids = torch.Tensor([3])


nn,rr = nearest_neighbor_features_per_object(a,b,label,1,gt_ids,n_chunks=100)
print(nn.size())
print(rr)
