from of_dataset1 import MyDataset
from of_DenseNet import DenseNet161_pre
from oneflow.utils import data
import oneflow as of
import oneflow.nn as nn

device = 'cuda'
model = DenseNet161_pre(pretrained=True)
model.to(device)
train_dataset = MyDataset(json_path='train_list1.json')
val_dataset = MyDataset(json_path='val_list1.json', if_train=False)
train = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
val = data.DataLoader(val_dataset, shuffle=False, batch_size=20)
val_len = val_dataset.__len__()
yuzhi = val_len * 0.97
opt1 = of.optim.Adam(model.parameters(), lr=0.00002)

lossf = nn.CrossEntropyLoss().to('cuda')
epochs1 = 50
for epoch in range(1, epochs1 + 1):
    model.train()
    print('epoch:', epoch)
    for data, label in train:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = lossf(pred.view(-1, 10), label.view(-1, 1))
        print('loss:', loss.item())
        loss.backward()
        opt1.step()
        opt1.zero_grad()

    model.eval()
    with of.no_grad():
        sum1 = 0
        for data, label in val:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            _, top1 = of.topk(pred.softmax(dim=1), dim=1, k=1)
            sum1 = sum1 + (top1.eq(label.view(-1, 1))).sum().item()
        print('the acc of val is {:0.4f}'.format(sum1.item() / val_len))
        if sum1 >= yuzhi:
            of.save(model.state_dict(), './0808/161_{:02d}_acc{:0.4f}'.format(epoch, sum1.item() / val_len))
