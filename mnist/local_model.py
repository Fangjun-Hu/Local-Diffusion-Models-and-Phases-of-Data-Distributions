import numpy as np
from numpy.random import randn as randn
import torch
import torch.nn.functional as F
import torch.nn as nn

class LayerScheme():
    def __init__(self, A, R, w, h, c=1):
        self.A = A      # length of area A
        self.R = R      # Radius, might not be divisible by A
        self.w = w      # width, height, channels of original image
        self.h = h
        self.c = c      
        self.B = A + 2 * R                          # length of area B

        self.L = int(np.ceil(R/A))                  # layer range, lx, ly goes from -L to LL
        self.LL = min(self.L, int(np.ceil(h/A)-self.L-1))
        self.R_rounded = self.L*A        # Rounded radius, to be divisible by A
        self.D = self.R_rounded * 2 + A             # length B rounded to be divisible by A, will be the stride for sliding window

        self.rh = int(np.ceil(h/self.D))            # vertical num of patches
        self.rw = int(np.ceil(w/self.D))            # horizontal num of h

        self.default_pad_w = self.rw*self.D - w     # default padding to make w, h divisible by D
        self.default_pad_h = self.rh*self.D - h
        
        self.edge_B = self.R_rounded - R            # crop off edge to exclude unnecessary remainder
        self.edge_A = self.R_rounded
    def pull_layer_B(self, img, lx, ly):
        ''' 
        pick out B-related pixels in a layer and arrange in patches.
        :img: (batch, channel, h, w)
        :return: (batch, channel*B*B, rh*rw)
        '''

        # by cropping and padding, shift the image so that (lx, ly) is on topleft.
        img_shifted = F.pad(img, (- self.edge_B - lx * self.A, 
                                  self.default_pad_w + lx*self.A, 
                                  - self.edge_B - ly * self.A, 
                                  self.default_pad_h + ly*self.A))
        patches = F.unfold(img_shifted, self.B, stride = self.D).permute(0,2,1)

        return patches
    
    def pull_layer_A(self, img, lx, ly):
        ''' 
        pick out A-related pixels in a layer and arrange in patches.
        :img: (batch, channel, h, w)
        :return: (batch, channel*A*A, rh*rw)
        '''

        # by cropping and padding, shift the image so that (lx, ly) is on topleft.

        # print(- self.edge_A - lx * self.A, self.default_pad_w + lx * self.A)
        # print(- self.edge_A - ly * self.A, self.default_pad_h + ly * self.A)
        img_shifted = F.pad(img, (- self.edge_A - lx * self.A,
                                  self.default_pad_w + lx * self.A, 
                                  - self.edge_A - ly * self.A, 
                                  self.default_pad_h + ly * self.A))
        patches = F.unfold(img_shifted, self.A, stride=self.D).permute(0,2,1)

        return patches
    def push_layer_A(self, patches, lx, ly):
        '''
        given area A data as patches, put them back to their original places.
        :patches: (batch,  rh*rw, channel*A*A)
        :return: (batch, channel, h, w)
        '''
        # b = patches.shape[0]
        # patches = patches.reshape(b, self.rh, self.rw, self.c, self.A, self.A)
        # canvas = torch.zeros(b, self.c, self.h+self.default_pad_h, self.w+self.default_pad_w).to(DEVICE)
        # for i in range(self.rh):
        #     for j in range(self.rw):
        #         h_start = i*self.D+ly * self.A+self.edge_A
        #         w_start = j*self.D+lx * self.A+self.edge_A
        #         canvas[:, : ,h_start:h_start+self.A, w_start:w_start+self.A] = patches[:,i,j]
        # canvas = torchvision.transforms.functional.crop(canvas, 0, 0, self.h, self.w)
        # return canvas
        b = patches.shape[0]
        patches = patches.reshape(b, self.rh, self.rw, self.c, self.A, self.A)
        patches = F.pad(patches, (self.edge_A, self.edge_A, self.edge_A, self.edge_A))
        patches.permute(0, 3, 1, 4, 2, 5) 
        img = patches.reshape(b, self.c, self.h+self.default_pad_h, self.w+self.default_pad_w)
        img = F.pad(img, (lx * self.A,
                                  -self.default_pad_w - lx * self.A,
                                  ly * self.A,
                                  -self.default_pad_h - ly * self.A))
        return img
    
    # def pull_layer_B1(self, img, lx, ly):
    #     ''' 
    #     pick out B-related pixels in a layer and arrange in patches.
    #     :img: (batch, channel, h, w)
    #     :return: (batch, rh*rw, channel*B*B)
    #     '''
    #     img_shifted = F.pad(img, ( - lx * self.A, 
    #                               self.default_pad_w + lx*self.A, 
    #                                - ly * self.A, 
    #                               self.default_pad_h + ly*self.A))
    #     b = img.shape[0]
    #     patches = img_shifted.reshape(
    #                 b,               # batch
    #                 self.c,               # channels
    #                 self.rh, self.D,  # height → patches
    #                 self.rw, self.D   # width → patches
    #             )
    #     patches = patches.permute(0, 2, 4,  1,3, 5)
    #     patches = patches[...,self.edge_B:self.D-self.edge_B, self.edge_B:self.D-self.edge_B]
    #     patches = patches.reshape(b, -1, self.c * self.B * self.B)
    #     return patches
    
    # def pull_layer_A1(self, img, lx, ly):
    #     ''' 
    #     pick out A-related pixels in a layer and arrange in patches.
    #     :img: (batch, channel, h, w)
    #     :return: (batch, rh*rw, channel*A*A)
    #     '''

    #     img_shifted = F.pad(img, ( - lx * self.A, 
    #                               self.default_pad_w + lx*self.A, 
    #                               - ly * self.A, 
    #                               self.default_pad_h + ly*self.A))
    #     b = img.shape[0]
    #     patches = img_shifted.reshape(
    #                 b,               # batch
    #                 self.c,               # channels
    #                 self.rh, self.D,  # height → patches
    #                 self.rw, self.D   # width → patches
    #             )
    #     patches = patches.permute(0, 2, 4,  1,3, 5)
    #     patches = patches[...,self.edge_A:self.D-self.edge_A,self.edge_A:self.D-self.edge_A]
    #     patches = patches.reshape(b, -1, self.c * self.A * self.A)
    #     return patches
    # def push_layer_A1(self, patches, lx, ly):
    #     '''
    #     given area A data as patches, put them back to their original places.
    #     :patches: (batch, channel*A*A, rh*rw)
    #     :return: (batch, channel, h, w)
        # '''
        # patches = patches.permute(0,2,1)
        # fold = nn.Fold(output_size=(self.h+self.default_pad_h-self.edge_A, self.w+self.default_pad_w-self.edge_A), kernel_size=self.A, stride=self.D)
        # img_shifted = fold(patches)
        # #print(img_shifted.shape)
        # img = F.pad(img_shifted, (self.edge_A + lx * self.A,
        #                           -self.default_pad_w - lx * self.A,
        #                           self.edge_A + ly * self.A,
        #                           -self.default_pad_h - ly * self.A))
        # return img   


    


class GenLocal(nn.Module):
    def __init__(
        self,
        ls,
        hidden_dim=512,
        classes=None,
        class_embed_dim=128,
        layer_embed_dim=16
    ):
        super(GenLocal, self).__init__()
        self.input_dim = ls.B*ls.B*ls.c
        self.output_dim = ls.A*ls.A*ls.c
        self.P = ls.rh*ls.rw
        self.ls = ls
        if classes is not None:
            self.class_embed = nn.Sequential(
                nn.Embedding(classes, class_embed_dim),
                nn.LayerNorm(class_embed_dim),
            )
        self.layer_embed_x = nn.Sequential(
                nn.Embedding(self.ls.L + self.ls.LL + 1, layer_embed_dim),
                nn.LayerNorm(layer_embed_dim),
        )
        self.layer_embed_y = nn.Sequential(
                nn.Embedding(self.ls.L + self.ls.LL + 1, layer_embed_dim),
                nn.LayerNorm(layer_embed_dim),
        )
        self.fc1 = nn.Parameter(torch.empty(self.P, hidden_dim, self.input_dim + 2 + bool(classes) * class_embed_dim + 2 * layer_embed_dim))
        self.bias1 = nn.Parameter(torch.zeros(self.P, hidden_dim))

        self.activation = nn.Mish()
        self.norm = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Parameter(torch.empty(self.P, self.output_dim, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(self.P, self.output_dim))
        self.reset_parameters()  # call custom init

    def reset_parameters(self):
        # Loop over the length dimension and initialize per position
        for p in range(self.P):
            nn.init.kaiming_uniform_(self.fc1[p, :, :], a=np.sqrt(1))
            nn.init.kaiming_uniform_(self.fc2[p, :, :], a=np.sqrt(1))

        # # Optionally initialize biases
        # fan_in1 = self.fc1.shape[1]
        # fan_in2 = self.fc2.shape[1]
        # bound1 = 1 / np.sqrt(fan_in1)
        # bound2 = 1 / np.sqrt(fan_in2)
        # nn.init.uniform_(self.bias1, -bound1, bound1)
        # nn.init.uniform_(self.bias2, -bound2, bound2)


    def forward(self, patches, ti, t, lx, ly, cond=None):
        h = torch.cat([patches, ti.view(-1, 1, 1).repeat(1, self.P, 1)], dim=2)
        h = torch.cat([h, t.view(-1, 1, 1).repeat(1, self.P, 1)], dim=2)
        if hasattr(self, "class_embed"):
            if cond is not None:
                cond_emb = self.class_embed(cond).unsqueeze(1).repeat(1, self.P, 1)
            else:
                # Insert zeros if cond is None, using the correct embedding dimension
                cond_emb_dim = self.class_embed[0].normalized_shape[0]
                cond_emb = torch.zeros(h.shape[0], self.P, cond_emb_dim, device=h.device, dtype=h.dtype)
            h = torch.cat([h, cond_emb], dim=2)
        h = torch.cat([h, self.layer_embed_x(lx + self.ls.L).view(1, 1, -1).repeat(t.shape[0], self.P, 1)], dim=2)
        h = torch.cat([h, self.layer_embed_y(ly + self.ls.L).view(1, 1, -1).repeat(t.shape[0], self.P, 1)], dim=2)

        h = torch.einsum('bki,kji->bkj', h, self.fc1) + self.bias1
        h = self.activation(h)
        h = self.norm(h)
        h = torch.einsum('bkj,koj->bko', h, self.fc2) + self.bias2
        return h
    
class Gen(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28 * 1,
        hidden_dim=1024,
        classes=None,
        class_embed_dim=128,
    ):
        super(Gen, self).__init__()
        if classes is not None:
            self.class_embed = nn.Sequential(
                nn.Embedding(classes, class_embed_dim),
                nn.LayerNorm(class_embed_dim),
            )
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1 + bool(classes) * class_embed_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        # Slightly modified initialization
        nn.init.constant_(self.model[-1].weight, 0)
        nn.init.constant_(self.model[-1].bias, 0)
        # nn.init.normal_(self.model[-1].weight, std=0.02)
        # nn.init.zeros_(self.model[-1].bias)

    def forward(self, x, t, cond=None):
        x = torch.cat([x, t], dim=-1)
        if cond is not None:
            x = torch.cat([x, self.class_embed(cond)], dim=-1)
        return self.model(x)