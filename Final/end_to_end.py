import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torchvision
from revuresnet18 import revuresnet18
import matplotlib.pyplot as plt
from skimage import measure
import os
import cv2
import sys
import warnings
warnings.filterwarnings('ignore')

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

class Model3d(torch.nn.Module):
    def __init__(self):
        super(Model3d, self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(512, 200) #encode_dim =200 (latent vec dim)
        self.encoder = nn.Sequential(self.model)
        n_dims=200; 
        nf=512;
        self.decoder = nn.Sequential(
            
        nn.ConvTranspose3d(n_dims, nf, 4, stride=1, padding=0, dilation=1, groups=1, bias=True),
        nn.BatchNorm3d(nf, eps=1e-5, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(nf, nf//2, 4, stride=2, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm3d(nf//2, eps=1e-5, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(nf//2, nf//4, 4, stride=2, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm3d(nf//4, eps=1e-5, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(nf//4, nf//8, 4, stride=2, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm3d(nf//8, eps=1e-5, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(nf//8, nf//16, 4, stride=2, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm3d(nf//16, eps=1e-5, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(nf//16, 1, 4, stride=2, padding=1, dilation=1, groups=1, bias=True)
        )

    def forward(self,x):
        latent_vec = self.encoder(x.float())
        latent_vec = latent_vec.view(latent_vec.size(0), -1, 1, 1, 1)
        vox = self.decoder(latent_vec)
        return vox;

def to_obj_str(verts, faces):
    text = ""           
    for p in verts:
        text += "v "
        for x in p:
            text += "{} ".format(x)
            text += "\n"
    for f in faces:
        text += "f "
        for x in f:
            text += "{} ".format(x + 1)
            text += "\n"
            return text                
                
def save_iso_obj(df, path, th, shift=True):
    if th < np.min(df):
        df[0, 0, 0] = th - 1
    if th > np.max(df):
        df[-1, -1, -1] = th + 1
    spacing = (1 / 128, 1 / 128, 1 / 128)
    verts, faces, _, _ = measure.marching_cubes_lewiner(df, th, spacing=spacing)
    if shift:
        verts -= np.array([0.5, 0.5, 0.5])
    obj_str = to_obj_str(verts, faces)
    with open(path, 'w') as f:
        f.write(obj_str)
                
def vis_voxel(voxels, path, counter=0, th=0.25, use_sigmoid=True):
    m = nn.Sigmoid()
    voxels = m(voxels)
    voxels = voxels.detach().numpy().squeeze()
    save_iso_obj(voxels, path, th=th)

weights_2p5d = Model2p5d()
weights_2p5d.load_state_dict(torch.load('models/first_module.pth', map_location=torch.device('cpu')))
weights_25d = Model2p5d()
weights_25d.load_state_dict(torch.load('models/first_module_old.pth', map_location=torch.device('cpu')))
weights_3d = Model3d()
weights_3d.load_state_dict(torch.load('models/second_module.pth', map_location=torch.device('cpu')))

input_path = 'temp_ikea/2D_images/'
count = 0
for file in sorted(os.listdir(input_path)):
    count += 1
    img = cv2.imread(input_path+file)
    img = cv2.resize(img, (256,256))
    transform = torchvision.transforms.ToTensor()
    test_img = transform(img).unsqueeze(0)
    output = weights_2p5d(test_img)
    output1 = weights_25d(test_img) 
    normal = output1['normal'].detach().numpy().squeeze().transpose(1,2,0)
    depth = output['depth'].detach().numpy().squeeze()
    sil = output['sil'].detach().numpy().squeeze()
    fig = plt.figure()
    plt.imshow(output1['normal'].detach().numpy().squeeze().transpose(1,2,0))
    fig.savefig('2p5d_results/normal_'+str(count)+'.jpg')
    fig = plt.figure()
    plt.imshow(output['depth'].detach().numpy().squeeze())
    fig.savefig('2p5d_results/depth_'+str(count)+'.jpg')
    fig = plt.figure()
    plt.imshow(output['sil'].detach().numpy().squeeze())
    fig.savefig('2p5d_results/sil_'+str(count)+'.jpg')
    print(normal.shape)

# count = 0
# for i in range(20):
#     count += 1
#     normal_img = cv2.imread('2p5d_results/normal_'+str(count)+'.jpg')
#     depth_img = cv2.imread('2p5d_results/depth_'+str(count)+'.jpg')
#     sil_img = cv2.imread('2p5d_results/sil_'+str(count)+'.jpg')
#     normal_img = cv2.resize(normal_img, (256,256))
#     transform = torchvision.transforms.ToTensor()
#     normal_img = transform(normal_img).unsqueeze(0)
#     depth_img = cv2.resize(depth_img, (256,256))
#     transform = torchvision.transforms.ToTensor()
#     depth_img = transform(depth_img[:,:,0]).unsqueeze(0)
#     sil_img = cv2.resize(sil_img, (256,256))
#     transform = torchvision.transforms.ToTensor()
#     sil_img = transform(sil_img[:,:,0]).unsqueeze(0)
    
#     is_bg = sil_img <= 0 #self.silhou_thres
#     depth_img[is_bg] = 0
#     normal_img[is_bg.repeat(1, 3, 1, 1)] = 0 # NOTE: if old net2, set to white (100),
#     x = torch.cat((depth_img, normal_img), 1) # and swap depth and normal     
#     V = weights_3d(x)
#     vis_voxel(V,'3D_models/3D_'+str(count)+'.obj')