import os
from torch.utils.data import Dataset
import torch
import cv2
from torchvision.transforms import transforms as T
from PIL import Image
import albumentations as A


class CarDataset(Dataset):
    def __init__(self, root, transform, mean, std):
        super(CarDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.mean = mean
        self.std = std

        self.filenames = os.listdir(os.path.join(self.root, 'images'))
        self.labels = os.listdir(os.path.join(self.root, 'labels'))

    def __getitem__(self, index):
        image_name = self.filenames[index]
        label_name = self.labels[index]

        image = cv2.imread(os.path.join(self.root,'images',image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root,'labels',label_name))

        aug = self.transform(image=image, mask=mask)
        image = Image.fromarray(aug['image'])
        mask = aug['mask']

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image = t(image)
        mask = torch.from_numpy(mask).to(torch.int64)
        mask = mask[:, :, 0]

        return image, mask

    def __len__(self):
        return len(self.filenames)


def load_data(batch_size, size):
    mean = [0.5048, 0.4892, 0.4739]
    std = [0.2709, 0.2673, 0.2681]
    train_transform = A.Compose([A.Resize(size, size, interpolation=cv2.INTER_NEAREST),
                    A.VerticalFlip(), # X轴水平翻转
                    A.HorizontalFlip(), # Y轴水平翻转
                    A.GridDistortion(p=0.2), # 网格失真
                    A.GaussNoise(), # 高斯噪声
                    A.RandomBrightnessContrast((0, 0.5), (0, 0.5))]) # 随机对比度
    test_transform = A.Resize(size, size, interpolation=cv2.INTER_NEAREST)

    train_loader = torch.utils.data.DataLoader(CarDataset('./dataset/train', train_transform,mean,std), batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(CarDataset('./dataset/test', test_transform,mean,std), batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_data(1, 256)
    for i, (X, y) in enumerate(train_loader):
        print(X.shape,y.shape)
        break
