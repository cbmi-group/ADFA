import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class MedicalDataset(Dataset):
    def __init__(self, dataset_path, class_name='breakHis', is_train=True, resize=256, cropsize=224):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y = self.load_dataset_folder()

        if class_name in['cervical','covid19']:
            self.transform_x =   T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        else:
            self.transform_x =   T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.5,0.5,0.5],
                                                        std=[0.5,0.5,0.5])])


    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith(('.png','jpeg'))])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
