#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import cuda
from chainer import function


class Linear(L.Linear):
    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear(x, self.W, self.b)


def linear(x, W, b=None):
    if b is None:
        return GuidedLinearFunction()(x, W)
    else:
        return GuidedLinearFunction()(x, W, b)


class GuidedLinearFunction(F.connection.linear.LinearFunction):
    def backward(self, inputs, grad_outputs):
        grad = super().backward(inputs, grad_outputs)
        if len(grad) == 3:
            gx, gW, gb = grad
            gx = F.relu(gx).data
            return gx, gW, gb
        else:
            gx, gW = grad
            return gx, gW


class GuidConvolution2DFunction(F.connection.convolution_2d.Convolution2DFunction):
    def backward(self, inputs, grad_outputs):
        grad = super().backward(inputs, grad_outputs)
        if len(grad) == 3:
            gx, gW, gb = grad
            gx = F.relu(gx).data
            return gx, gW, gb
        else:
            gx, gW = grad
            return gx, gW


def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, **kwargs):
    func = GuidConvolution2DFunction(stride, pad, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)


class Convolution2D(L.Convolution2D):
    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d(x, self.W, b=self.b, stride=self.stride, pad=self.pad)


class VGGNet(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1=Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        self.cam = h
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        return h


class VGGNet2(chainer.Chain):
    def __init__(self):
        super(VGGNet2, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        self.cam = h
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h
