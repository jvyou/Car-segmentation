import os
import shutil
import random
import torch
from torchvision import io


def car_to_dataset():
    # 汽车数据集转换为语义分割数据集
    images_path = 'Car segmentation/images'
    labels_path = 'Car segmentation/masks'
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists(os.path.join('dataset', 'train')):
        os.mkdir(os.path.join('dataset', 'train'))
    if not os.path.exists(os.path.join('dataset', 'test')):
        os.mkdir(os.path.join('dataset', 'test'))
    if not os.path.exists(os.path.join('dataset', 'train', 'images')):
        os.mkdir(os.path.join('dataset', 'train', 'images'))
    if not os.path.exists(os.path.join('dataset', 'train', 'labels')):
        os.mkdir(os.path.join('dataset', 'train', 'labels'))
    if not os.path.exists(os.path.join('dataset', 'test', 'images')):
        os.mkdir(os.path.join('dataset', 'test', 'images'))
    if not os.path.exists(os.path.join('dataset', 'test', 'labels')):
        os.mkdir(os.path.join('dataset', 'test', 'labels'))

    image_name = os.listdir(images_path)
    length = len(image_name)
    train_list = random.sample(range(length),int(length * 0.8))
    train_set = set(train_list)
    test_list = [i for i in range(length) if i not in train_set]

    for i in train_list:
        shutil.copy(os.path.join(images_path,image_name[i]),os.path.join('dataset', 'train','images'))
        shutil.copy(os.path.join(labels_path,image_name[i]),os.path.join('dataset', 'train','labels'))
    for i in test_list:
        shutil.copy(os.path.join(images_path,image_name[i]),os.path.join('dataset', 'test','images'))
        shutil.copy(os.path.join(labels_path,image_name[i]),os.path.join('dataset', 'test','labels'))

    with open(os.path.join('dataset', 'train.txt'), 'w') as f:
        for i in train_list:
            f.write(str(i))
            f.write("\n")

    with open(os.path.join('dataset', 'test.txt'), 'w') as f:
        for i in test_list:
            f.write(str(i))
            f.write("\n")


def get_mean_std(path):
    length = len(os.listdir(path))
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for name in os.listdir(path):
        img = io.read_image(os.path.join(path, name)).type(torch.float32) / 255
        for i in range(3):
            means[i] += img[i, :, :].mean()
            stds[i] += img[i, :, :].std()

    print("means:{}".format(means.div_(length)), "stds:{}".format(stds.div_(length)))


if __name__ == '__main__':
    car_to_dataset()
    get_mean_std('dataset/train/images')
