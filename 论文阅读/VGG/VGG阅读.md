# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 

 

## 42 Matrix

|      | 定义                    | 特点                                                   |
| ---- | ----------------------- | ------------------------------------------------------ |
| 背景 | ImageNet Challenge 2014 |                                                        |
| 对象 | IMAGE                   | LARGE-SCALE                                            |
| 问题 | RECOGNITION             |                                                        |
| 方法 | CONVOLUTIONAL NETWORKS  | VERY DEEP using very small (3 × 3) convolution filters |

## 逻辑树

![VGG思维导图](D:\work_DL\论文阅读\VGG\VGG思维导图.PNG)

## 算法

```python
inputs = array([256,224,224,3])
net=Conv2d(inputs,3,3,64,relu)
#256,224,224,64
net=Conv2d(inputs,3,3,64,relu)
#256,224,224,64
net=Maxpool(net)
#256,112,112,64
net=Conv2d(inputs,3,3,128,relu)
#256,112,112,128
net=Conv2d(inputs,3,3,128,relu)
#256,112,112,128
net=Maxpool(net)
#256,56,56,128
net=Conv2d(inputs,3,3,256,relu)
#256,56,56,256
net=Conv2d(inputs,3,3,256,relu)
#256,56,56,256
net=Conv2d(inputs,3,3,256,relu)
#256,56,56,256
net=Maxpool(net)
#256,28,28,256
net=Conv2d(inputs,3,3,512,relu)
#256,28,28,512
net=Conv2d(inputs,3,3,512,relu)
#256,28,28,512
net=Conv2d(inputs,3,3,512,relu)
#256,28,28,512
net=Maxpool(net)
#256,14,14,512
net=Conv2d(inputs,3,3,512,relu)
#256,14,14,512
net=Conv2d(inputs,3,3,512,relu)
#256,14,14,512
net=Conv2d(inputs,3,3,512,relu)
#256,14,14,512
net=Maxpool(net)
#256,7,7,512
net=Flatten(net)
#256,25088
net=FullyConected(net,4096)
#256,4096
net=FullyConected(net,4096)
#256,4096
net=FullyConected(net,1000)
#256,1000
outputs=Softmax(net)

loss = cross_entropy(outputs, labels)
train = Momentum(learning rate = 0.01,momentum = 0.9)
#The learning rate was initially set to 10−2, and then decreased by a factor of 10 

```

## 实验结果

| 数据库                            | 指标              | 性能     |
| --------------------------------- | ----------------- | -------- |
| ILSVRC-2012 and ILSVRC-2013-top-1 | val. error        | 23.7%    |
| ILSVRC-2012 and ILSVRC-2013-top-5 | val. error        | 6.8%     |
| ILSVRC-2014 localisation-top-5    | val. error        | 26.9%    |
| VOC-2007                          | mean AP           | 89.7     |
| VOC-2012                          | mean AP           | 89.3     |
| Caltech-101                       | mean class recall | 92.7±0.5 |
| Caltech-101                       | mean class recall | 86.2±0.3 |

