import random
import argparse
import torch
import torch.optim as optim
import warnings
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
from utils.dataset import MedicalDataset
from utils.functions import *
from utils.adfa import *


warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
CLASS_NAMES = ['brainMRI','covid19','BUSI','cervical']

def parse_args():
    parser = argparse.ArgumentParser('CFA configuration')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--size', type=int, choices=[224, 256], default=224)
    parser.add_argument('--gamma_d', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--class_name', type=str, default='all')
    
    return parser.parse_args()


def run(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    class_names = CLASS_NAMES if args.class_name == 'all' else [args.class_name]

    total_roc_auc = []

    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    for class_name in class_names:
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        print(' ')
        print('%s | newly initialized...' % class_name)
        
        train_dataset    = MedicalDataset(dataset_path  = args.data_path, 
                                        class_name    =     class_name, 
                                        resize        =            256,
                                        cropsize      =      args.size,
                                        is_train      =           True)
        
        test_dataset     = MedicalDataset(dataset_path  = args.data_path, 
                                        class_name    =     class_name, 
                                        resize        =            256,
                                        cropsize      =      args.size,
                                        is_train      =          False)


        train_loader   = DataLoader(dataset         = train_dataset, 
                                    batch_size      =             10, 
                                    pin_memory      =          True,
                                    shuffle         =          True,
                                    drop_last       =          False,)

        test_loader   =  DataLoader(dataset        =   test_dataset, 
                                    batch_size     =              10, 
                                    pin_memory     =           True,)

        # load model
        model = wide_resnet50_2(pretrained=True, progress=True)
        model = model.to(device)
        model.eval()

        loss_fn = ADFA(model, train_loader, args.gamma_d, device)
        loss_fn = loss_fn.to(device)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
        
        epochs = args.epoch
        params = [{'params' : loss_fn.parameters()},]
        optimizer     = optim.AdamW(params        = params, 
                                    lr            = 1e-3,
                                    weight_decay  = 5e-4,
                                    amsgrad       = True )

        r'TRAIN PHASE'
        for epoch in tqdm(range(epochs), '%s -->'%(class_name)):
            loss_fn.train()
            for (x, _) in train_loader:
                optimizer.zero_grad()
                _ = model(x.to(device)) 
                loss,_ = loss_fn(outputs)
                loss.backward()
                optimizer.step()
                del outputs
                outputs = []
        
        r'TEST PHASE'
        test_imgs = list()
        gt_list = list()
        heatmaps = None
        loss_fn.eval()
        for x, y in test_loader:
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

            _ = model(x.to(device))
            _, score = loss_fn(outputs)
            heatmap = score.cpu().detach()
            heatmap = torch.mean(heatmap, dim=1) 
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            del outputs
            outputs = []

        heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
        heatmaps = gaussian_smooth(heatmaps, sigma=4)
        scores = rescale(heatmaps)
        
        r'Image-level AUROC'
        img_roc_auc = cal_img_roc(scores, gt_list)       
        print('image ROCAUC: %.3f'% (img_roc_auc))
        total_roc_auc.append(img_roc_auc)

if __name__ == '__main__':
    seeds=[1024,213,317]
    for seed in seeds:
        print('seed=%d'%seed)
        run(seed)
