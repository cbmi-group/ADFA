import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from .functions import *
from .coordconv import CoordConv2d
from .soft_ot import TopK_custom
import torch.nn.functional as F
from .eca import eca_layer

class ADFA(nn.Module):
    def __init__(self, model, data_loader, gamma_d, device):
        super(ADFA, self).__init__()
        self.device = device
        
        self.C   = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3

        self.Descriptor = Descriptor(self.gamma_d).to(device)
        self._init_centroid(model, data_loader)
        self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()
        self.Topk = TopK_custom(self.K, epsilon=0.1, max_iter = 200)
        
        self.C = self.C.transpose(-1, -2).detach()
        self.C = nn.Parameter(self.C, requires_grad=False)

    def forward(self, p):
        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')
        kl = F.kl_div(phi_p.softmax(dim=-1).log(), self.C.transpose(-1, -2).softmax(dim=-1), reduction='none')
        kldist=torch.sum(kl,(1,2),keepdim=True).detach()
        
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True) 
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, self.C)
        dist     = features + centers - f_c 
        dist = dist*kldist
        dist_top = self.Topk(dist)
        dist_top = dist_top.view(dist.size())
        dist = torch.sum(dist_top*dist,dim=-1,keepdim=True)

        score = 0
        loss = 0
        if self.training:
            loss = torch.mean(dist)
        else:
            score = rearrange(torch.sqrt(dist), 'b (h w) c -> b c h w', h=self.scale)

        return loss, score

    def _init_centroid(self, model, data_loader):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        h = []
        h += [model.layer1[-1].register_forward_hook(hook)]
        h += [model.layer2[-1].register_forward_hook(hook)]
        h += [model.layer3[-1].register_forward_hook(hook)]

        for i, (x, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            _ = model(x)
            self.scale = outputs[0].size(2)
            phi_p = self.Descriptor(outputs)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)
            outputs = []
        
        for i in h:
            i.remove()


class Descriptor(nn.Module):
    def __init__(self, gamma_d):
        super(Descriptor, self).__init__()
        dim = 1792 
        self.eca = eca_layer(dim,k_size=9)
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)

    def forward(self, p):
        sample = None
        for o in p :  #concatenate
            o = F.avg_pool2d(o, 3, 1, 1)
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)
        phi_p = self.layer(sample)
        phi_p = 0.1*self.eca(phi_p)+phi_p
        return phi_p
