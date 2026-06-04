#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_cifar100.sh <data_dir>"
    exit 1
fi

echo "Downloading CIFAR100"
mkdir -p $1/CIFAR100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-100-python.tar.gz -C $1/CIFAR100/

echo "Downloading CIFAR100C"
wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf "CIFAR-100-C.tar" -C  $1/
rm -rf "CIFAR-100-C.tar"
