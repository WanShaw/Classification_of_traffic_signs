from of_dataset2 import MyDataset
from of_DenseNet import DenseNet161_pre
from oneflow.utils import data
import oneflow as of
import oneflow.nn as nn


device = 'cuda'
model = DenseNet161_pre(num_classes=10, pretrained=True)
model.to(device)
json_file = 'train_list1.json'

train_dataset = MyDataset(json_path=json_file)
val_dataset = MyDataset(json_path='val_list1.json', if_train=False)

train = data.DataLoader(train_dataset, batch_size=20, shuffle=True)
val = data.DataLoader(val_dataset, shuffle=False, batch_size=1)

opt1 = of.optim.Adam(model.parameters(), lr=3e-4)

lossf = nn.CrossEntropyLoss().to('cuda')

epochs1 = 50
acc = 0.97
for epoch in range(1, epochs1 + 1):
    model.train()
    print('epoch:', epoch)
    for data, label in train:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = lossf(pred, label)
        print('loss:', loss.item())
        opt1.zero_grad()
        loss.backward()
        opt1.step()

    model.eval()
    with of.no_grad():
        sum1 = 0
        for data, label in val:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            _, top1 = of.topk(pred.softmax(dim=1), k=1)
            if top1.item() == label:
                sum1 += 1
        acc1 = sum1 / val_dataset.__len__()
        print('the arr of val is {}'.format(acc1))
        if acc1 > acc:
            of.save(model.state_dict(), './0722/161_pre_%02d_%0.3f' %(epoch, acc1))
