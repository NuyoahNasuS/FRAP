#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_MNIST.sh <data_dir>"
    exit 1
fi

# MNIST: 
mkdir -p $1/MNIST
cd $1/MNIST
echo "Downloading MNIST datasets"
wget -nc https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -nc https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget -nc https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget -nc https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip -f *.gz
cd $1

# Q-MNIST
mkdir -p $1/Q-MNIST
cd $1/Q-MNIST
echo "Downloading Q-MNIST datasets"
wget -nc https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz
wget -nc https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz
wget -nc https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz
wget -nc https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz
gunzip -f *.gz
cd $1

# SVHN
echo "Downloading SVHN datasets"
mkdir -p SVHN
cd SVHN
wget -nc http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -nc http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -nc http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
cd $1

# USPS
wget https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl

