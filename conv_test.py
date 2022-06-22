import torch

x = torch.rand(1, 1, 5, 5)


#--------------------------------------------#
# 1x1Conv分支
# 中心为1x1Conv的卷积核，周围为0的卷积相当于恒等映射
#--------------------------------------------#
r = torch.rand(1)
print(r)        # tensor([0.7974])

conv1 = torch.nn.Conv2d(1, 1, 1, bias=False)
kernel1 = torch.tensor([[[[r]]]])
conv1.weight.data = kernel1
y1 = conv1(x)

conv2 = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
kernel2 = torch.tensor([[[[0, 0, 0],
                          [0, r, 0],
                          [0, 0, 0]]]])
conv2.weight.data = kernel2
y2 = conv2(x)

print(y1==y2)
# tensor([[[[True, True, True, True, True],
#           [True, True, True, True, True],
#           [True, True, True, True, True],
#           [True, True, True, True, True],
#           [True, True, True, True, True]]]])


#--------------------------------------------#
# identity分支
# 中心为1，周围为0的卷积相当于恒等映射
#--------------------------------------------#
conv3 = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
kernel3 = torch.tensor([[[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0.]]]])
conv3.weight.data = kernel3
y3 = conv3(x)
print(x==y3)
# tensor([[[[True, True, True, True, True],
        #   [True, True, True, True, True],
        #   [True, True, True, True, True],
        #   [True, True, True, True, True],
        #   [True, True, True, True, True]]]])