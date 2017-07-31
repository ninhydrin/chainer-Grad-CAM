#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import chainer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from VGGNet import VGGNet
from chainer import cuda
from chainer import serializers
from chainer import Variable
import chainer.functions as F

def image_convert(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('image', help="Path to image")
    parser.add_argument('--label', type=int, default=0, help="Categories you want to visualize")
    args = parser.parse_args()

    category_label = args.label
    parent, img_name = os.path.split(args.image)
    save_name = "result/{}".format(img_name)

    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    raw_img = cv2.imread(args.image).astype(np.float32)
    img = raw_img - mean
    img = cv2.resize(img, (224, 224)).transpose((2, 0, 1))
    img = img[np.newaxis]

    vgg = VGGNet()
    serializers.load_hdf5('VGG.model', vgg)

    input_img = Variable(img)
    pred = vgg(input_img)
    probs = F.softmax(pred).data[0]
    top5 = np.argsort(probs)[::-1][:5]
    pred.zerograd()
    pred.grad = np.zeros([1, 1000], dtype=np.float32)
    pred.grad[0, top5[category_label]] = 1

    words = open('data/synset_words.txt').readlines()
    words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
    words = np.asarray(words)

    probs = np.sort(probs)[::-1][:5]
    for w, p in zip(words[top5], probs):
        print('{}\tprobability:{}'.format(w, p))
    print("your choice ", words[top5[category_label]][1])
    pred.backward(True)

    feature, grad = vgg.cam.data[0], vgg.cam.grad[0]
    cam = np.ones(feature.shape[1:], dtype=np.float32)
    weights = grad.mean((1, 2))*1000
    for i, w in enumerate(weights):
        cam += feature[i] * w
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    guid = np.maximum(input_img.grad[0], 0).transpose(1, 2, 0)
    guided_cam = image_convert(guid * heatmap[:, :, np.newaxis])
    guided_bp = image_convert(guid)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    image = img[0, :].transpose(1, 2, 0)
    image -= np.min(image)
    image = np.minimum(image, 255)
    cam_img = np.float32(heatmap) + np.float32(image)
    cam_img = 255 * cam_img / np.max(cam_img)

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax1.imshow(raw_img[:, :, ::-1].astype(np.uint8))
    ax2.imshow(guided_cam)
    ax3.imshow(heatmap[:, :, ::-1])
    ax4.imshow(cam_img[:, :, ::-1].astype(np.uint8))
    fig.savefig(save_name)
