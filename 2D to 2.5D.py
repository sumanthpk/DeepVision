#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision
from revuresnet18 import revuresnet18
import os
import cv2
import sys
import warnings
warnings.filterwarnings('ignore')
# from parallel import DataParallelModel, DataParallelCriterion


# In[2]:


class Model2p5d(torch.nn.Module):
    def __init__(self):
        super(Model2p5d, self).__init__()
        self.model = models.resnet18(pretrained = True)
         
        module_list = list()
        
        in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        module_list.append(
            nn.Sequential(
                in_conv,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool
            )
        )
                
        module_list.append(self.model.layer1)
        module_list.append(self.model.layer2)
        module_list.append(self.model.layer3)
        module_list.append(self.model.layer4)
        self.encoder = nn.ModuleList(module_list)
        
        out_planes = [3,1,1]
        layer_names = ['normal','depth','sil']
        
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list2 = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list2.append(revresnet.layer1)
            module_list2.append(revresnet.layer2)
            module_list2.append(revresnet.layer3)
            module_list2.append(revresnet.layer4)
            module_list2.append(
                nn.Sequential(
                    revresnet.deconv1,
                    revresnet.bn1,
                    revresnet.relu,
                    revresnet.deconv2
                )
            )
            module_list2 = nn.ModuleList(module_list2)
            setattr(self, 'decoder_' + layer_name, module_list2)
            self.decoders[layer_name] = module_list2
        
    def forward(self,im):
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
            
        self.encoder_out = feat_maps[-1]
        
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


# In[ ]:


# # Preprocess image code

# import os
# import cv2
# superpath = "/home/ghostvortex/Documents/DL_Project/Dataset/"
# for folder in sorted(os.listdir(superpath))[1:]:
#     for folder_in in sorted(os.listdir(superpath+folder)):
#         for files in sorted(os.listdir(superpath+folder+"/"+folder_in)):
#             img = cv2.imread(superpath+folder+"/"+folder_in+"/"+files)
#             img = cv2.resize(img, (256,256))
#             cv2.imwrite(superpath+folder+"/"+folder_in+"/"+files, img)
            
            
# superpath = "/home/ghostvortex/Dataset/Dataset/"
# index = -1
# batch_size = 500
# for folder in sorted(os.listdir(superpath))[index+1:(index+int(batch_size/20)+1)]:
#     for folder_in in sorted(os.listdir(superpath+folder)):
#         for files in sorted(os.listdir(superpath+folder+"/"+folder_in)):
#             print(superpath+folder+"/"+folder_in+"/"+files)
#     break


# In[3]:


def loader(index, batch_size):
    superpath = "/home/ghostvortex/Dataset/Dataset/"
    input_img = []
    normal_img = []
    depth_img = []
    sil_img = []
    for folder in sorted(os.listdir(superpath))[index+1:(index+int(batch_size/20)+1)]:
        for folder_in in sorted(os.listdir(superpath+folder)):
            for files in sorted(os.listdir(superpath+folder+"/"+folder_in)):
                if "rgb" in files:
                    img = cv2.imread(superpath+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img).unsqueeze(0)
                    input_img.append(img)
                elif "depth" in files:
                    img = cv2.imread(superpath+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img[:,:,0]).unsqueeze(0)
                    depth_img.append(img)
                elif "normal" in files:
                    img = cv2.imread(superpath+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img).unsqueeze(0)
                    normal_img.append(img)
                elif "sil" in files:
                    img = cv2.imread(superpath+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img[:,:,0]).unsqueeze(0)
                    sil_img.append(img)
    input_img1 = torch.cat(input_img, dim=0)
    normal_img1 = torch.cat(normal_img, dim=0)
    depth_img1 = torch.cat(depth_img, dim=0)
    sil_img1 = torch.cat(sil_img, dim=0)
    return input_img1, normal_img1, depth_img1, sil_img1


# In[5]:


f = open('/home/ghostvortex/models/status-first-module.txt','w').close()


# In[6]:


class Trainer():
    def __init__(self,model):
        self.model = model
        
    def train(self, model):
        num_epochs = 5
        batch_size = 60                                        # Multiples of 20 (Very Important) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
#         parallel_model = DataParallelModel(model)             # Encapsulate the model
#         parallel_loss  = DataParallelCriterion(criterion)     # Encapsulate the loss function
#         model = nn.DataParallel(model)        

        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.to(device)
        
        f = open('/home/ghostvortex/models/status-first-module.txt','w').close()

        for epoch in range(num_epochs):
            ind = 0
            for i in range(int(2150/(batch_size/20))):        # Total number of objects (2150) / (Batch size / no.of views) [500/20]
                print("Epoch:", epoch,", Iteration:",i)
                f = open('/home/ghostvortex/models/status-first-module.txt','a')
                f.write("Epoch:  " + str(epoch) + ",  Iteration:" + str(i)+ "\n")
                f.close()
                inp_img, normal_img, depth_img, sil_img = loader(ind, batch_size)
                inp = inp_img.to(device)
                normal_img = normal_img.to(device)
                depth_img = depth_img.to(device)
                sil_img = sil_img.to(device)

                try:
                    output = model.forward(inp)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        sys.stdout.flush()
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad
                        torch.cuda.empty_cache()
                        output = model.forward(inp)
                    else: raise e

                loss_normal = criterion(output['normal'], normal_img)
                loss_depth = criterion(output['depth'], depth_img)
                loss_sil = criterion(output['sil'], sil_img)
                loss = loss_normal + loss_depth + loss_sil            

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ind += batch_size // 20

            print("Epoch: ", epoch , "Loss: ", loss.item())
            f = open('/home/ghostvortex/models/status-first-module.txt','a')
            f.write("\n"+"Epoch:  " + str(epoch) + ",  Loss:" + str(loss.item())+ "\n")
            f.close()
        f.close()
        return model

model = Model2p5d()
trainer = Trainer(model)
trained_model = trainer.train(model)
torch.save(trained_model.state_dict(),'/home/ghostvortex/models/first_module.pth')


# In[11]:


# weights = Model2p5d()
# weights.load_state_dict(torch.load('/home/ghostvortex/models/first_module.pth'))


# In[20]:


# import matplotlib.pyplot as plt
# test_img = cv2.imread('/home/ghostvortex/Dataset/Dataset/obj_rgb300/rgb_sil_depth_view1/03001627_c236deaff8c6fb0d29c9a7a92b0a566d_view000_rgb.png')
# transform = torchvision.transforms.ToTensor()
# test_img = transform(test_img).unsqueeze(0)
# output = weights(test_img)
# plt.imshow(output['sil'].detach().numpy().squeeze())
# plt.show()


# In[ ]:


# torch.cuda.max_memory_allocated()


# In[57]:


# img = cv2.imread("temp_results/rgb.png")
# img = cv2.resize(img, (256,256))
# import matplotlib.pyplot as plt
# plt.imshow(img)
# transform = torchvision.transforms.ToTensor()
# img = transform(img).unsqueeze(0)
# output = trained_model.forward(img)


# In[58]:


# sil = output['sil'].detach().numpy().squeeze()
# print(sil.shape)
# plt.imshow(sil)
# plt.show()


# In[59]:


# depth = output['depth'].detach().numpy().squeeze()
# print(depth.shape)
# plt.imshow(depth)
# plt.show()


# In[60]:


# normal = output['normal'].detach().numpy().squeeze()
# print(normal.shape)
# normal = normal.transpose(1,2,0)
# plt.imshow(normal)
# plt.show()


# In[61]:


# sil = np.resize(sil,(1,256,256))
# sil = np.vstack((sil,sil,sil))
# sil = sil.transpose(1,2,0)
# print(sil.shape)
# plt.imshow(sil)
# plt.show()


# In[41]:


# from PIL import Image
# import cv2
# import numpy as np
# img = cv2.imread("chair.png")
# img = cv2.resize(img, (256,256))
# transform = torchvision.transforms.ToTensor()
# img = transform(img).unsqueeze(0)
# train = Model2p5d()
# output = train.forward(img)
# print("Normal:", output['normal'].shape)
# print("Depth:", output['depth'].shape)
# print("Silhouette:", output['sil'].shape)

