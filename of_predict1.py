from of_DenseNet import DenseNet161_pre
import flowvision.transforms as ft
import cv2
import json
import oneflow as of
import glob

pth_paths = glob.glob('161_*')
n = len(pth_paths)
print('num_pth', n)
file = open('submit_example.json')
infos = json.load(file)
annotations = infos['annotations']

size = 224
transforms = ft.Compose([
            ft.ToTensor(),
            ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

len1 = len(annotations)
print('总数量:', len1)
model = DenseNet161_pre(num_classes=10, pretrained=False)
i = 1
for pth_path in pth_paths:
    print(i)
    i += 1
    model.load_state_dict(of.load(pth_path))
    model.to('cuda')
    model.eval()
    result = {}
    ann = []

    for temp in annotations:
        imgfile = temp['filename']
        img = cv2.imread(imgfile)
        img = cv2.blur(img, ksize=(9, 9))
        img = transforms(img)
        img = of.reshape(img, (1, 3, size, size))
        img = img.to('cuda')
        with of.no_grad():
            pred = model(img)
        _, indices = of.topk(pred, k=1, dim=1)
        re = indices.item()
        ann.append({'filename': imgfile, 'label': int(re)})

    result['annotations'] = ann
    with open("{}.json".format(pth_path), "w", encoding='utf-8') as f:  # 设置'utf-8'编码
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
