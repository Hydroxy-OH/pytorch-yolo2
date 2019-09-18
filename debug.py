from __future__ import print_function

import os

import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from darknet import Darknet
from torch.autograd import Variable
from utils import convert2cpu, image2torch

cfgfile = "face4.1re_95.91.cfg"
weightfile = "face4.1re_95.91.conv.15"
imgpath = "data/train/images/10002.png"
labpath = imgpath.replace("images", "labels").replace(
    "JPEGImages", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
label = torch.zeros(50 * 5)
if os.path.getsize(labpath):
    tmp = torch.from_numpy(np.loadtxt(labpath))
    #tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width))
    #tmp = torch.from_numpy(read_truths(labpath))
    tmp = tmp.view(-1)
    tsz = tmp.numel()
    #print("labpath = %s , tsz = %d" % (labpath, tsz))
    if tsz > 50 * 5:
        label = tmp[0:50 * 5]
    elif tsz > 0:
        label[0:tsz] = tmp
label = label.view(1, 50 * 5)

model = Darknet(cfgfile)
region_loss = model.loss
model.load_weights(weightfile)

print("--- bn weight ---")
print(model.models[0][1].weight)
print("--- bn bias ---")
print(model.models[0][1].bias)
print("--- bn running_mean ---")
print(model.models[0][1].running_mean)
print("--- bn running_var ---")
print(model.models[0][1].running_var)

model.train()
m = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-2,
                      momentum=0.9, weight_decay=0.1)

img = Image.open(imgpath)
img = image2torch(img)
img = Variable(img.cuda())

target = Variable(label)

print("----- img ---------------------")
print(img.data.storage()[0:100])
print("----- target  -----------------")
print(target.data.storage()[0:100])

optimizer.zero_grad()
output = m(img)
print("----- output ------------------")
print(output.data.storage()[0:100])
exit()

loss = region_loss(output, target)
print("----- loss --------------------")
print(loss)

save_grad = None


def extract(grad):
    global saved_grad
    saved_grad = convert2cpu(grad.data)


output.register_hook(extract)
loss.backward()

saved_grad = saved_grad.view(-1)
for i in range(saved_grad.size(0)):
    if abs(saved_grad[i]) >= 0.001:
        print("%d : %f" % (i, saved_grad[i]))

print(model.state_dict().keys())
# print(model.models[0][0].weight.grad.data.storage()[0:100])
# print(model.models[14][0].weight.data.storage()[0:100])
weight = model.models[13][0].weight.data
grad = model.models[13][0].weight.grad.data
mask = torch.abs(grad) >= 0.1
print(weight[mask])
print(grad[mask])

optimizer.step()
weight2 = model.models[13][0].weight.data
print(weight2[mask])
