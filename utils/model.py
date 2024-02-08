import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.enc_blk11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.enc_blk12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.enc_blk21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.enc_blk22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.enc_blk31 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.enc_blk32 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.enc_blk41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.enc_blk42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.enc_blk51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.enc_blk52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_blk11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.dec_blk12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_blk21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.dec_blk22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_blk31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(128)
        self.dec_blk32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_blk41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        self.dec_blk42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(64)

        # Output Layer
        self.out_layer = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        enc11 = self.relu(self.bn1(self.enc_blk11(x)))
        enc12 = self.relu(self.bn2(self.enc_blk12(enc11)))
        pool1 = self.pool(enc12)

        enc21 = self.relu(self.bn3(self.enc_blk21(pool1)))
        enc22 = self.relu(self.bn4(self.enc_blk22(enc21)))
        pool2 = self.pool(enc22)

        enc31 = self.relu(self.bn5(self.enc_blk31(pool2)))
        enc32 = self.relu(self.bn6(self.enc_blk32(enc31)))
        pool3 = self.pool(enc32)

        enc41 = self.relu(self.bn7(self.enc_blk41(pool3)))
        enc42 = self.relu(self.bn8(self.enc_blk42(enc41)))
        pool4 = self.pool(enc42)

        enc51 = self.relu(self.bn9(self.enc_blk51(pool4)))
        enc52 = self.relu(self.bn10(self.enc_blk52(enc51)))

        up1 = self.upconv1(enc52)
        up11 = torch.cat([up1, enc42], dim=1)
        dec11 = self.relu(self.bn11(self.dec_blk11(up11)))
        dec12 = self.relu(self.bn12(self.dec_blk12(dec11)))

        up2 = self.upconv2(dec12)
        up22 = torch.cat([up2, enc32], dim=1)
        dec21 = self.relu(self.bn13(self.dec_blk21(up22)))
        dec22 = self.relu(self.bn14(self.dec_blk22(dec21)))

        up3 = self.upconv3(dec22)
        up33 = torch.cat([up3, enc22], dim=1)
        dec31 = self.relu(self.bn15(self.dec_blk31(up33)))
        dec32 = self.relu(self.bn16(self.dec_blk32(dec31)))

        up4 = self.upconv4(dec32)
        up44 = torch.cat([up4, enc12], dim=1)
        dec41 = self.relu(self.bn17(self.dec_blk41(up44)))
        dec42 = self.relu(self.bn18(self.dec_blk42(dec41)))

        out = self.out_layer(dec42)

        return out
