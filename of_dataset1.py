from oneflow.utils.data import Dataset
from flowvision import transforms as ft
import oneflow as of
import json
import cv2
import numpy as np

M1 = cv2.getRotationMatrix2D(center=(112, 112), angle=90, scale=1)
M2 = cv2.getRotationMatrix2D(center=(112, 112), angle=-90, scale=1)
M3 = cv2.getRotationMatrix2D(center=(112, 112), angle=180, scale=1)
var_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
class MyDataset(Dataset):
    def __init__(self, json_path='train_list.json', if_train=True):
        super(MyDataset, self).__init__()
        # 读取json文件
        file = open(json_path)
        infos = json.load(file)
        annotations = infos['annotations']
        self.train_list = annotations
        self.len = len(annotations)
        self.if_train = if_train
        self.transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           ])
    def __getitem__(self, item):
        file1 = self.train_list[item]  # 一个字典
        img_file = file1['filename']  # 图片路径
        class1 = int(file1['label'])  # 图片类别
        img = cv2.imread(img_file)
        if self.if_train:
            t11 = of.randint(0, 21, size=[1]).item()
            t12 = of.randint(1, 21, size=[1]).item()
            t13 = of.randint(1, 21, size=[1]).item()
        else:
            t11 = 20
            t12 = of.randint(1, 21, size=[1]).item()
            t13 = of.randint(1, 21, size=[1]).item()

        if t11 <= 18:
            t3 = of.randint(0, 7, size=[1]).item()
            img = AddGaussianNoise(img, std=var_list[t3])
            img = cv2.blur(img, ksize=(9, 9))
        else:
            img = cv2.blur(img, ksize=(9, 9))
        # 旋转
        if t12 <= 2:
            img = cv2.warpAffine(img, M1, (224, 224))
        elif t12 <= 4:
            img = cv2.warpAffine(img, M2, (224, 224))
        elif t12 <= 6:
            img = cv2.warpAffine(img, M3, (224, 224))
        else:
            pass
        # 翻转
        if t13 <= 2:
            img = cv2.flip(img, 0)
        elif t13 <= 4:
            img = cv2.flip(img, 1)
        elif t13 <= 6:
            img = cv2.flip(img, -1)
        else:
            pass
        img = self.transforms(img)
        return img, of.tensor(class1, dtype=of.int)
    def __len__(self):
        return self.len


def AddGaussianNoise(array, mean=0, std=0.02):

    noise = np.random.normal(mean, std, array.shape)
    array = array / 255 + noise
    array[array < 0] = 0
    array[array > 1] = 1
    return np.uint8(array * 255)


if __name__ == '__main__':
    a = MyDataset()
    b, c = a.__getitem__(1)
    print(b.size())
    print(c)
