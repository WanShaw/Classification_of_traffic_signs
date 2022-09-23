# 多模型投票文件
from of_DenseNet import DenseNet161_pre
import flowvision.transforms as ft
import cv2
import json
import oneflow as of

M1 = cv2.getRotationMatrix2D(center=(112, 112), angle=90, scale=1)
M2 = cv2.getRotationMatrix2D(center=(112, 112), angle=-90, scale=1)
M3 = cv2.getRotationMatrix2D(center=(112, 112), angle=180, scale=1)

file = open('submit_example.json')
infos = json.load(file)
annotations = infos['annotations']

size = 224
transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
# 我使用四个模型进行投票，行21-行40换成你存储的权重路径
model1 = DenseNet161_pre(num_classes=10, pretrained=False)
model2 = DenseNet161_pre(num_classes=10, pretrained=False)
model3 = DenseNet161_pre(num_classes=10, pretrained=False)
model4 = DenseNet161_pre(num_classes=10, pretrained=False)

model1.load_state_dict(of.load('model1/161_45_acc0.9868'))
model1.to('cuda')
model1.eval()

model2.load_state_dict(of.load('model2/161_ls_43_acc0.9868'))
model2.to('cuda')
model2.eval()

model3.load_state_dict(of.load('model3/161_ls_48_acc0.9868'))
model3.to('cuda')
model3.eval()

model4.load_state_dict(of.load('model4/161_pre_47_0.990'))
model4.to('cuda')
model4.eval()

result = {}
ann = []
ii = 0
nn1 = 0
# 处理字典
def f(label1):
    temp = -1
    re = []
    for k in label1.keys():
        if label1[k] > temp:
            re = []
            temp = label1[k]
            re.append(k)
        elif label1[k] == temp:
            re.append(k)
        else:
            pass
    return re


for temp in annotations:
    print(ii)
    ii += 1
    label1 = {}
    label2 = []
    imgfile = temp['filename']
    img = cv2.imread(imgfile)
    img = cv2.blur(img, ksize=(9, 9))
    # 五张图片，我只弄了五张图片，你可以继续增加
    img1 = cv2.warpAffine(img, M1, (224, 224))
    img1 = transforms(img1)
    img1 = of.reshape(img1, (1, 3, size, size))
    img1 = img1.to('cuda')
    #
    img2 = cv2.warpAffine(img, M2, (224, 224))
    img2 = transforms(img2)
    img2 = of.reshape(img2, (1, 3, size, size))
    img2 = img2.to('cuda')
    #
    img3 = cv2.warpAffine(img, M3, (224, 224))
    img3 = transforms(img3)
    img3 = of.reshape(img3, (1, 3, size, size))
    img3 = img3.to('cuda')
    #
    img4 = transforms(img)
    img4 = of.reshape(img4, (1, 3, size, size))
    img4 = img4.to('cuda')
    #
    img5 = cv2.flip(img, 0)
    img5 = transforms(img5)
    img5 = of.reshape(img5, (1, 3, size, size))
    img5 = img5.to('cuda')
    input1 = of.cat([img1, img2, img3, img4, img5], dim=0)  # 一下五个batch_size
    # 生成二十个结果
    with of.no_grad():
        #
        pred1 = model1(input1)
        _, indices1 = of.topk(pred1, k=1, dim=1)  # 5 * 1
        for index in indices1:
            label2.append(index.item())
        #
        pred2 = model2(input1)
        _, indices2 = of.topk(pred2, k=1, dim=1)  # 5 * 1
        for index in indices2:
            label2.append(index.item())
        #
        pred3 = model3(input1)
        _, indices3 = of.topk(pred3, k=1, dim=1)  # 5 * 1
        for index in indices3:
            label2.append(index.item())
        #
        pred4 = model4(input1)
        _, indices4 = of.topk(pred4, k=1, dim=1)  # 5 * 1
        for index in indices4:
            label2.append(index.item())
    #
    for re in label2:
        if re in label1.keys():
            label1[re] += 1
        else:
            label1[re] = 1

    re1 = f(label1)

    if len(re1) == 1:
        ann.append({'filename': imgfile, 'label': int(re1[0])})
    else:
        print('model4')
        nn1 += 1
        re1 = indices4[3, 0]
        ann.append({'filename': imgfile, 'label': int(re1)})

print(nn1)
result['annotations'] = ann
with open("bestmodel.json", "w", encoding='utf-8') as f:  # 设置'utf-8'编码
    f.write(json.dumps(result, ensure_ascii=False, indent=4))
