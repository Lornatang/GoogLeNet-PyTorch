import json
import sys
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *

from googlenet_pytorch import GoogLeNet


parser = argparse.ArgumentParser("GoogLeNet Classifier Tool")
parser.add_argument("-a", "--arch", type=str, default='GoogLeNet',
                    help="Weight of the model loaded by default.")
parser.add_argument("-r", "--resume", type=str, default="",
                    help="Loading custom weight.")
parser.add_argument("-s", "--image_size", type=int, default=None,
                    help="Size of classified image. (default=None).")
parser.add_argument("-l", "--labels_map", type=str, default="./labels_map.txt",
                    help="Image tag. (default='./labels_map.txt').")
parser.add_argument("-n", "--num_classes", type=int, default=1000,
                    help="Number of categories of images. (default=1000).")
parser.add_argument("-p", "--echo", type=bool, default=False,
                    help="Show pop ups or not? (default: False)")
args = parser.parse_args()


def classifier(image_path):
    # Open image
    img = Image.open(image_path)
    img = tfms(img).unsqueeze(0)

    # Classify with GoogLeNet
    with torch.no_grad():
        logits = model(img)
    preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

    for idx in preds:
        label = labels_map[idx]
        probability = torch.softmax(logits, dim=1)[0, idx].item()
    return label, probability


class Picture(QWidget):
    def __init__(self):
        super(Picture, self).__init__()

        self.resize(1000, 1000)
        self.setWindowTitle("Classifier tool")

        self.label = QLabel(self)
        self.label.setFixedSize(args.image_size, args.image_size)
        self.label.move(300, 300)
        self.label.setStyleSheet(
            "QLabel{background:white;}"
            "QLabel{color:rgb(0,0,0);font-size:18px;font-weight:bold;font-family:宋体;}"
        )

        # add open image button
        self.btn_open_img = QPushButton(self)
        self.btn_open_img.setText("Open image")
        self.btn_open_img.move(10, 30)
        self.btn_open_img.clicked.connect(self.open_image)

        # add open popup window button
        self.btn_open_echo = QPushButton(self)
        self.btn_open_echo.setText("open popup window")
        self.btn_open_echo.move(10, 200)
        self.btn_open_echo.clicked.connect(self.open_echo)

        # add close popup window button
        self.btn_close_echo = QPushButton(self)
        self.btn_close_echo.setText("close popup window")
        self.btn_close_echo.move(10, 300)
        self.btn_close_echo.clicked.connect(self.close_echo)

    def open_image(self):
        img_name, _ = QFileDialog.getOpenFileName(self, "Open image", "", "*.jpg;;*.png;;All Files(*)")
        img = QtGui.QPixmap(img_name).scaled(args.image_size, args.image_size)
        self.label.setPixmap(img)
        text, prob = classifier(img_name)
        print("------------------------------")
        print(f"Label: {text:<75}")
        print(f"Probability: {prob:.6f}.")
        print("------------------------------")
        if args.echo:
            self.echo(str(text), prob)

    def open_echo(self):
        args.echo = True

    def close_echo(self):
        args.echo = False

    def echo(self, text, prob):
        QMessageBox.information(
            self, "Message",
            f"Label :{str(text):<75}\nProbability: {prob:.6f}")


if __name__ == "__main__":
    model = GoogLeNet.from_pretrained(args.arch)
    model.eval()
    if args.image_size is None:
        args.image_size = 224
    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load class names
    labels_map = json.load(open(args.labels_map))
    labels_map = [labels_map[str(i)] for i in range(args.num_classes)]

    app = QtWidgets.QApplication(sys.argv)
    my = Picture()
    my.show()
    sys.exit(app.exec_())
