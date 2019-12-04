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

def loader(index, batch_size):
    superpath_rgb = "/home/ghostvortex/Dataset/Dataset/"
    superpath_3d = "/home/ghostvortex/3D/"
    voxel = []
    normal_img = []
    depth_img = []
    sil_img = []
    
    for folder in sorted(os.listdir(superpath_3d))[index+1:(index+int(batch_size/20)+1)]:
        for folder_in in sorted(os.listdir(superpath_3d+folder)):
            for files in sorted(os.listdir(superpath_3d+folder+"/"+folder_in)):
                # print(count, " ",superpath+folder+"/"+folder_in+"/"+files)
                if "rotvox" in files:
                    labels = np.load(superpath_3d+folder+"/"+folder_in+"/"+files)
                    label = np.where(labels['voxel']>0.5,1,0)
                    transform = torchvision.transforms.ToTensor()
                    label = transform(label).unsqueeze(0)
                    voxel.append(label)
    
    for folder in sorted(os.listdir(superpath_rgb))[index+1:(index+int(batch_size/20)+1)]:
        for folder_in in sorted(os.listdir(superpath_rgb+folder)):
            for files in sorted(os.listdir(superpath_rgb+folder+"/"+folder_in)):
                # print(superpath_rgb+folder+"/"+folder_in+"/"+files)
                if "depth" in files:
                    img = cv2.imread(superpath_rgb+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img[:,:,0]).unsqueeze(0)
                    depth_img.append(img)
                elif "normal" in files:
                    img = cv2.imread(superpath_rgb+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img).unsqueeze(0)
                    normal_img.append(img)
                elif "sil" in files:
                    img = cv2.imread(superpath_rgb+folder+"/"+folder_in+"/"+files)
                    transform = torchvision.transforms.ToTensor()
                    img = transform(img[:,:,0]).unsqueeze(0)
                    sil_img.append(img)
                    
    voxel1 = torch.cat(voxel, dim=0)
    normal_img1 = torch.cat(normal_img, dim=0)
    depth_img1 = torch.cat(depth_img, dim=0)
    sil_img1 = torch.cat(sil_img, dim=0)
    
    is_bg = sil_img1 <= 0 #self.silhou_thres
    depth_img1[is_bg] = 0
    normal_img1[is_bg.repeat(1, 3, 1, 1)] = 0 # NOTE: if old net2, set to white (100),
    x = torch.cat((depth_img1, normal_img1), 1) # and swap depth and normal             
    return voxel1, x

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

class Trainer():
    def __init__(self,model):
        self.model = model
        
    def train(self, model):
        num_epochs = 15 
        batch_size = 20
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.BCEWithLogitsLoss(reduction = 'elementwise_mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.to(device)
        
        f = open('/home/ghostvortex/models/status-second-module.txt','w').close()

        for epoch in range(num_epochs):
            ind = 0
            for i in range(int(300/(batch_size/20))):
                print("Epoch:", epoch,", Iteration:",i)
                # Load data and labels of that batch
                f = open('/home/ghostvortex/models/status-second-module.txt','a')
                f.write("Epoch:  " + str(epoch) + ",  Iteration:" + str(i)+ "\n")
                f.close()
                
                voxel, x = loader(ind, batch_size)
                label = voxel.to(device)
                inp = x.to(device)
                
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

                output = output.squeeze()
                loss = criterion(output.type(torch.FloatTensor),label.type(torch.FloatTensor)) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ind += batch_size // 20
            print("Epoch: ", epoch , "Loss: ", loss.item())
            f = open('/home/ghostvortex/models/status-second-module.txt','a')
            f.write("\n"+"Epoch:  " + str(epoch) + ",  Loss:" + str(loss.item())+ "\n")
            f.close()
        f.close()
        return model

def init_func(m, init_type= 'kaiming'):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.init.normal_(m.weight.data, 1.0, init_param)
        torch.init.constant_(m.bias.data, 0.0)

model = Model3d()
model.apply(init_func)
model = model.cuda()
trainer = Trainer(model)
trained_model = trainer.train(model)
torch.save(trained_model.state_dict(),'/home/ghostvortex/models/second_module.pth')
