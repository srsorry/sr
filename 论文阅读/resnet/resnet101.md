```python
inputs = array([N,224,224,3])
#N,224,224,3
net=Conv2d(inputs,7,7,64,relu,stride=2)
#N,112,112,64
net=Maxpool(3,3,stride=2)
#N,56,56,64
net = Bottleneck(net,64,64,256)
#N,56,56,256
net = Bottleneck(net,64,64,256)
#N,56,56,256
net = Bottleneck(net,64,64,256)
#N,56,56,256
net = Maxpool(net)
#N,28,28,256
net = Bottleneck(net,128,128,512)
#N,28,28,512
net = Bottleneck(net,128,128,512)
#N,28,28,512
net = Bottleneck(net,128,128,512)
#N,28,28,512
net = Bottleneck(net,128,128,512)
#N,28,28,512
net = Maxpool(net)
#N,14,14,512
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Bottleneck(net,256,256,1024)
#N,14,14,1024
net = Maxpool(net)
#N,7,7,1024
net = Bottleneck(net,512,512,2048)
#N,7,7,2048
net = Bottleneck(net,512,512,2048)
#N,7,7,2048
net = Bottleneck(net,512,512,2048)
#N,7,7,2048
net = Averagepool(net)
#N,?,?,2048
net=Flatten(net)
#N,?
net=FullyConected(net,1000)
#N,1000

outputs=Softmax(net)
#N,1,1
```

![resnet网络结构](C:\Users\17204\Desktop\论文阅读\resnet网络结构.PNG)