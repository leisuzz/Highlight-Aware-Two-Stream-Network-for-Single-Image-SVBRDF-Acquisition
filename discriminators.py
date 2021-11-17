import os, time, sys
import itertools
import pickle

import tensorflow as tf
import numpy as np
from keras.layers import Dense, Activation, Conv2D,  Flatten, BatchNormalization


class GlobalDiscriminator():
    def __init__(self, input_shape, arc='places2'):
        super(GlobalDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]


        self.conv1 = Conv2D(self.img_c, kernel_size=(5,5), stride=(2,2), padding=2)
        self.conv1 = Conv2D(64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')

        self.conv2 = Conv2D(128, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.conv3 = Conv2D(256, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        self.conv4 = Conv2D(512, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')

        if arc == 'celeba':
            in_features = 512 * (self.img_h//32) * (self.img_w//32)
            self.flatten5 = Flatten()
            self.linear5 = Dense(in_features, input_shape=(1024,))
            self.act5 = Activation('relu')

        elif arc == 'places2':
            self.conv5 = Conv2D(512, kernel_size=(5,5), stride=(2,2), padding=2)
            self.bn5 = BatchNormalization()
            self.act5 = Activation('relu')
            in_features = 512 * (self.img_h//64) * (self.img_w//64)
            self.flatten6 = Flatten()
            self.linear6 = Dense(in_features, input_shape=(1024,))
            self.act6 = Activation('relu')
        else:
            raise ValueError('Unsupported architecture \'%s\'.' % self.arc)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        if self.arc == 'celeba':
            x = self.act5(self.linear5(self.flatten5(x)))
        elif self.arc == 'places2':
            x = self.bn5(self.act6(self.conv5(x)))
            x = self.act6(self.linear6(self.flatten6(x)))
        return x

class LocalDiscriminator():
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]
        self.conv1 = Conv2D(self.img_c, kernel_size=(5,5), stride=(2,2), padding=2)
        self.conv1 = Conv2D(64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(128, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.conv3 = Conv2D(256, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        self.conv4 = Conv2D(512, kernel_size=(5,5), stride=(2,2), padding=2)
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')

        in_features = 512 * (self.img_h//32) * (self.img_w//32)
        self.flatten4 = Flatten()
        self.linear4 = Dense(in_features, input_shape=(1024,))
        self.act5 = Activation('relu')

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.act5(self.linear4(self.flatten4(x)))
        return x    