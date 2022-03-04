"""
Dataset:
!wget http://users.itk.ppke.hu/~nagad2/resources/electrical_devs/ElectricDevices_TRAIN.dms
!wget http://users.itk.ppke.hu/~nagad2/resources/electrical_devs/ElectricDevices_TEST.dms
TODO finish implementation from scratch
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

with open('ElectricDevices_TEST.dms', 'r') as f:
    test_file = f.read()

with open('ElectricDevices_TRAIN.dms', 'r') as f:
    train_file = f.read()

train_file = train_file.split('\n')
train_data = np.zeros((len(train_file), 96))
train_labels = np.zeros((len(train_file), 1))

for ind,t in enumerate(train_file):
    d = t.split(',')
    if d[0] != "":
        train_labels[ind] = int(d[0])-1
        train_data[ind,:] = np.asarray(d[1:])

test_file = test_file.split('\n')
test_data = np.zeros((len(test_file), 96))
test_labels = np.zeros((len(test_file), 1))

for ind,t in enumerate(test_file):
    d = t.split(',')
    if d[0] != "":
        test_labels[ind] = int(d[0])-1
        test_data[ind, :] = np.asarray(d[1:])