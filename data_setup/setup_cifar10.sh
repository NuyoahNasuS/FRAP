#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_cifar10.sh <data_dir>"
    exit 1
fi

echo "Downloading CIFAR10"
mkdir -p $1/CIFAR10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz -C $1/CIFAR10/

echo "Downloading CIFAR10v2"
mkdir -p $1/CIFAR10V2
wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_train.npz
mv cifar102_train.npz $1/CIFAR10V2/
wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_test.npz
mv cifar102_test.npz $1/CIFAR10V2/

echo "Downloading CIFAR10C"
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C $1/

