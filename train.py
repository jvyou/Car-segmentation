import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.dataLoader import load_data
from utils.model import UNet
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum() / float(correct.numel()))
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for classes in range(0, n_classes):
            true_class = (pred_mask == classes)
            true_label = (mask == classes)

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def train(net, epochs, train_iter, test_iter, device, loss, optimizer, scheduler, model_path, auto_save):
    train_acc_list = []
    train_miou_list = []
    train_loss_list = []

    test_acc_list = []
    test_miou_list = []
    test_loss_list = []

    net = net.to(device)

    for epoch in range(epochs):

        net.train()
        train_acc = 0
        train_miou = 0
        train_loss = 0
        train_len = 0
        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc="train epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                scheduler.step()
                train_len += len(y)
                train_acc += pixel_accuracy(y_hat, y)
                train_miou += mIoU(y_hat, y)
                train_loss += l.detach()
                pbar.set_postfix({'loss': "{:.4f}".format(train_loss / train_len),
                                  'acc': "{:.4f}".format(train_acc / train_len),
                                  'miou': "{:.4f}".format(train_miou / train_len)})
                pbar.update(1)
            train_acc_list.append(train_acc / train_len)
            train_miou_list.append(train_miou / train_len)
            train_loss_list.append(train_loss.cpu().numpy() / train_len)

        net.eval()
        test_acc = 0
        test_miou = 0
        test_loss = 0
        test_len = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc="test epoch {}/{}".format(epoch + 1, num_epochs)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_len += len(y)
                test_acc += pixel_accuracy(y_hat, y)
                test_miou += mIoU(y_hat, y)
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    pbar.set_postfix({'loss': "{:.4f}".format(test_loss / test_len),
                                      'acc': "{:.4f}".format(test_acc / test_len),
                                      'miou': "{:.4f}".format(test_miou / test_len)})
                    pbar.update(1)
            test_acc_list.append(test_acc / test_len)
            test_miou_list.append(test_miou / test_len)
            test_loss_list.append(test_loss.cpu().numpy() / test_len)

        if (epoch + 1) % auto_save == 0:
            torch.save(net.state_dict(), model_path)

    plt.plot([i+1 for i in range(len(train_acc_list))], train_acc_list, 'bo--', label="train_acc")
    plt.plot([i+1 for i in range(len(test_acc_list))], test_acc_list, 'ro--', label="test_acc")
    plt.title("train_acc vs test_acc")
    plt.ylabel("acc")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig('logs/acc.png')
    plt.show()

    plt.plot([i+1 for i in range(len(train_miou_list))], train_miou_list, 'bo--', label="train_miou")
    plt.plot([i+1 for i in range(len(test_miou_list))], test_miou_list, 'ro--', label="test_miou")
    plt.title("train_miou vs test_miou")
    plt.ylabel("miou")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig('logs/miou.png')
    plt.show()

    plt.plot([i+1 for i in range(len(train_loss_list))], train_loss_list, 'bo--', label="train_loss")
    plt.plot([i+1 for i in range(len(test_loss_list))], test_loss_list, 'ro--', label="test_loss")
    plt.title("train_loss vs test_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig('logs/loss.png')
    plt.show()


if __name__ == '__main__':
    batch_size = 2 # 批量大小
    crop_size = 256 # 裁剪大小
    in_channels = 3 # 输入图像通道
    classes_num = 5 # 输出标签类别
    num_epochs = 100 # 总轮次
    auto_save = 10 # 自动保存的间隔轮次
    lr = 1e-3 # 学习率
    weight_decay = 1e-4 # 权重衰退
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 选择设备

    train_loader, test_loader = load_data(batch_size, crop_size)

    net = UNet(classes_num) # 定义模型
    model_path = 'model_weights/UNet.pth'

    loss = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay) # 定义优化器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(train_loader))

    print("训练开始")
    time_start = time.time()
    train(net, num_epochs, train_loader, test_loader, device=device, loss=loss, optimizer=optimizer,scheduler=scheduler, model_path=model_path, auto_save=auto_save)
    torch.save(net.state_dict(), model_path)
    time_end = time.time()
    seconds = time_end - time_start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("训练结束")
    print("本次训练时长为：%02d:%02d:%02d" % (h, m, s))
