```python
inputs = array([N,224,224,3])
net=Conv2d(inputs,3,3,64,relu)
#N,224,224,64
net=Conv2d(net,3,3,64,relu)
#N,224,224,64
pool1=Maxpool(net)
#N,112,112,64
net=Conv2d(pool1,3,3,128,relu)
#N,112,112,128
net=Conv2d(net,3,3,128,relu)
#N,112,112,128
pool2=Maxpool(net)
#N,56,56,128
net=Conv2d(pool2,3,3,256,relu)
#N,56,56,256
net=Conv2d(net,3,3,256,relu)
#N,56,56,256
net=Conv2d(net,3,3,256,relu)
#N,56,56,256
pool3=Maxpool(net)
#N,28,28,256
net=Conv2d(pool3,3,3,512,relu)
#N,28,28,512
net=Conv2d(net,3,3,512,relu)
#N,28,28,512
net=Conv2d(net,3,3,512,relu)
#N,28,28,512
pool4=Maxpool(net)
#N,14,14,512
net=Conv2d(pool4,3,3,512,relu)
#N,14,14,512
net=Conv2d(net,3,3,512,relu)
#N,14,14,512
net=Conv2d(net,3,3,512,relu)
#N,14,14,512
pool5=Maxpool(net)
#N,7,7,512
net = Conv2d(pool5,7,7,4096,relu,dropout_ratio=0.5)
#N,7,7,4096
net = Conv2d(net,1,1,4096,relu,dropout_ratio=0.5)
#N,7,7,4096
score_fr=Conv2d(net,1,1,60)
#N,7,7,60
fcn_32s_pre = DeConv2d(score_fr,n,n,60,stride=32)
#N,224,224,60
fcn_32s_output = Softmax(fcn_32s_pre)
#N,224,224,1
upscore2 = DeConv2d (score_fr,4,4,60,stride=2)
#N,14,14,60
score_pool4 = Conv2d(pool4,1,1,60)
#N,14,14,60
fuse_pool4 = concat(score_pool4,upscore2)
#N,14,14,60
net =DeConv2d (net,N,N,60,stride=16)
#N,224,224,60
fcn_16s_output = Softmax(net)
#N,224,224,1
upscore_pool4 = DeConv2d (fuse_pool4,4,4,60,stride=2)
#N,28,28,60
score_pool3 = Conv2d(pool3,1,1,60)
#N,28,28,60
fuse_pool3 = concat(score_pool3,upscore_pool4)
#N,28,28,60
net = DeConv2d (fuse_pool3,16,16,60,stride=8)
#N,224,224,60
fcn_8s_output = Softmax(net)
#N,224,224,1
```

