#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python /home/ec2-user/cifar/vgg_3_dropout.py >/home/ec2-user/cifar/vgg_3_dropout.py.log </dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python /home/ec2-user/cifar/vgg_3_data_augmentation.py >/home/ec2-user/cifar/vgg_3_data_augmentation.py.log </dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python /home/ec2-user/cifar/vgg_3_dropout_augmentation.py >/home/ec2-user/cifar/vgg_3_dropout_augmentation.py.log </dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python /home/ec2-user/cifar/vgg_3_dropout_augmentation_batchnorm.py >/home/ec2-user/cifar/vgg_3_dropout_augmentation_batchnorm.py.log </dev/null 2>&1 &
