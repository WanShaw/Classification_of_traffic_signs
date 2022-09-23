# Classification-of-traffic-signs
北交大深度学习暑期争霸赛-高级赛道
# 数据集下载地址
https://www.datafountain.cn/competitions/569/datasets
# 代码运行环境
Oneflow=0.8.0+cu112

flowvision=0.2.0
# 运行介绍
## 生成验证集和训练集
运行random_split.py文件
## 删除json文件中的错误标签图片
运行json_delete.py文件
## 代码训练
根据细节的不同，分成了三个训练文件，分别为of_train1.py、of_train2.py、of_train3.py

细节上有数据增强的方式不同，具体见of_dataset1.py、of_dataset2.py

还有是否进行标签平滑，参考of_train1.py文件的行20-行31
## 结果预测
运行of_predict.py文件

## 模型的读取和保存，在运行时，需要自行进行修改，再此不在详细说明。

# 写在最后
第一次写github仓库，多少会出现问题，请见谅，有兴趣和问题，请提issue。

感谢大家的注意，谢谢。
