# 删除json文件中的错误标签图片，已经划分后的json文件
import json

delete_lsit = ['train/M1/00010.jpg', 'train/M1/00046.jpg', 'train/M1/00047.jpg', 'train/M1/00116.jpg',
               'train/M1/00122.jpg', 'train/M1/00123.jpg', 'train/M1/00157.jpg', 'train/M1/00199.jpg',
               'train/M1/00231.jpg', 'train/M1/00225.jpg', 'train/M1/00248.jpg', 'train/GuideSign/00426.jpg',
               'train/GuideSign/00003.jpg', 'train/GuideSign/00686.jpg', 'train/GuideSign/00054.jpg',
               'train/GuideSign/00177.jpg', 'train/GuideSign/00372.jpg', 'train/GuideSign/00548.jpg',
               'train/GuideSign/00667.jpg', 'train/GuideSign/00795.jpg', 'train/GuideSign/00859.jpg',
               'train/GuideSign/01064.jpg', 'train/GuideSign/01084.jpg', 'train/GuideSign/01095.jpg',
               'train/GuideSign/01218.jpg', 'train/M4/00165.jpg', 'train/M4/00342.jpg',
               'train/M4/00428.jpg', 'train/M4/00438.jpg', 'train/M4/00439.jpg','train/M4/00546.jpg',
               'train/M4/00499.jpg', 'train/M4/00524.jpg', 'train/M4/00592.jpg', 'train/M4/00594.jpg',
               'train/M4/00817.jpg', 'train/M4/01074.jpg', 'train/M4/01305.jpg', 'train/M4/01332.jpg',
               'train/M4/01821.jpg', 'train/M4/01929.jpg', 'train/M4/02138.jpg', 'train/M4/02196.jpg',
               'train/M4/02197.jpg', 'train/M4/02198.jpg', 'train/M4/02580.jpg', 'train/M4/02601.jpg',
               'train/M4/02688.jpg', 'train/M4/02892.jpg', 'train/M4/03252.jpg', 'train/M5/00016.jpg',
               'train/M5/00039.jpg', 'train/M5/00114.jpg', 'train/M5/00134.jpg', 'train/M5/00214.jpg',
               'train/M5/00209.jpg', 'train/M6/00030.jpg', 'train/M7/00435.jpg', 'train/P1/00098.jpg',
               'train/P12/00001.jpg', 'train/P12/00002.jpg', 'train/P12/00012.jpg', 'train/P12/00013.jpg',
               'train/P12/00016.jpg', 'train/P12/00046.jpg', 'train/P12/00053.jpg', 'train/P12/00075.jpg',
               'train/P12/00093.jpg', 'train/P12/00095.jpg', 'train/P12/00096.jpg', 'train/P12/00098.jpg',
               'train/P12/00099.jpg', 'train/W1/00111.jpg',
               ]

file1 = open('train_list.json')
infos1 = json.load(file1)
ann1 = infos1['annotations']

file2 = open('val_list.json')
infos2 = json.load(file2)
ann2 = infos2['annotations']
print(len(ann1) + len(ann2))

d1 = {}
d2 = {}
train_list = []
val_list = []
sum1 = 0
temp = []
for t in ann1:
    file_temp = t['filename']
    if file_temp in delete_lsit:
        temp.append(file_temp)
        sum1 += 1
        continue
    else:
        train_list.append(t)

sum2 = 0
for t in ann2:
    file_temp = t['filename']
    if file_temp in delete_lsit:
        temp.append(file_temp)
        sum2 += 1
        continue
    else:
        val_list.append(t)
print(sum1)
print(sum2)

d1['annotations'] = train_list
with open('train_list1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d1, ensure_ascii=False, indent=4))
d2['annotations'] = val_list
with open('val_list1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d2, ensure_ascii=False, indent=4))
