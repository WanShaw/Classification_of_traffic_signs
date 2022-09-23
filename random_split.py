import random
import json
# 读取json文件
file = open('train1.json')
infos = json.load(file)
annotations = infos['annotations']
r = 0.8
d1 = {}
d2 = {}
train_list = []
val_list = []

for t in annotations:
    a = random.random()
    if a <= r:
        train_list.append(t)
    else:
        val_list.append(t)
d1['annotations'] = train_list
with open('train_list1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d1, ensure_ascii=False))
d2['annotations'] = val_list
with open('val_list1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d2, ensure_ascii=False))