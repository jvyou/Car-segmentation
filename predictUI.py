import cv2
import sys
import torch
import numpy as np
from PIL import Image
from utils.model import UNet
from torchvision import transforms
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import *
from screen import Ui_MainWindow
import tkinter as tk
from tkinter import filedialog #获取文件


class Main(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.photo)
        self.pushButton_2.clicked.connect(self.devio)
        self.pushButton_3.clicked.connect(self.exit)
        self.flag = 1

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])

        self.net = UNet(5)
        self.net.load_state_dict(torch.load('model_weights/UNet.pth'))

        # 显示封面
        pix = QtGui.QPixmap('R-C.jpg')
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)

    def photo(self):
        root = tk.Tk()
        root.withdraw()
        Filepath = filedialog.askopenfilename() # 获取文件路径

        if (Filepath[-1] == 'g' and Filepath[-2] == 'n' and Filepath[-3] == 'p') \
                or (Filepath[-1] == 'g' and Filepath[-2] == 'p' and Filepath[-3] == 'j'):

            # 读取图片并做预测
            img = cv2.imread(Filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(img)
            img = self.transform(img)
            img = img.unsqueeze(0)
            pred = torch.argmax(self.net(img), dim=1)
            pred = pred.detach().numpy()
            pred = pred.reshape(256, 256)
            img = self.label2bgr(pred)
            img = cv2.resize(img, (401, 401), interpolation=cv2.INTER_LINEAR)

            # 显示
            q_img = QImage(img.data, img.shape[0], img.shape[1], img.shape[0] * 3, QImage.Format_RGB888)
            pix = QPixmap(q_img).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pix)
            self.label.setScaledContents(True)

        else:
            reply = QMessageBox.information(self, '标题', '请选择图片文件！',
                                            QMessageBox.Ok)  # 信息框

    def devio(self):
        root = tk.Tk()
        root.withdraw()
        Filepath = filedialog.askopenfilename()

        pix = QtGui.QPixmap('R-C.jpg')
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)

        if Filepath[-1] == '4' and Filepath[-2] == 'p' and Filepath[-3] == 'm':
            self.flag = 1
            cap = cv2.VideoCapture(Filepath)

            while cap.isOpened() and self.flag:
                ret, frame = cap.read()

                if not ret:
                    break

                # 做预测
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frame = frame.unsqueeze(0)
                pred = torch.argmax(self.net(frame), dim=1)
                pred = pred.detach().numpy()
                pred = pred.reshape(256, 256)
                frame = self.label2bgr(pred)
                frame = cv2.resize(frame, (401, 401), interpolation=cv2.INTER_LINEAR)

                # 显示
                temp_imgSrc = QImage(frame[:], frame.shape[1], frame.shape[0], frame.shape[1] * 3,
                                     QImage.Format_RGB888)
                pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap(pixmap_imgSrc))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()

        else:
            reply = QMessageBox.information(self, '标题', '请选择视频文件！',
                                            QMessageBox.Ok)  # 信息框

        pix = QtGui.QPixmap('R-C.jpg')
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)

    def exit(self):
        if self.flag:
            self.flag = 0
            reply = QMessageBox.information(self, '标题', '退出成功！',
                                            QMessageBox.Ok)  # 信息框
        else:
            reply = QMessageBox.information(self, '标题', '还未读入视频！',
                                            QMessageBox.Ok)  # 信息框

    def label2bgr(self,pred):
        frame = np.zeros((256, 256, 3)).astype(np.uint8)
        frame[pred == 0] = (68, 1, 84)
        frame[pred == 1] = (58, 82, 139)
        frame[pred == 2] = (32, 144, 140)
        frame[pred == 3] = (94, 201, 97)
        frame[pred == 4] = (253, 231, 36)
        return frame


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 使窗体按照Qt设计显示
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
