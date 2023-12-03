#! /usr/bin/env bash

set verbose
set -o errexit

export CC=gcc
export CXX=g++

conda create -y -n chap_amr \
    python=3.10 \
    pytorch \
    torchvision \
    torchaudio \
    pytorch-cuda=12.1 \
    pyg \
    numpy \
    numba \
    matplotlib \
    seaborn \
    pandas \
    nltk \
    scikit-learn \
    ipython \
    ipywidgets \
    networkx \
    rich \
    black \
    isort \
    dill \
    tensorboard \
    -c pytorch -c pyg -c nvidia

conda activate chap_amr

# pip install -r requirements.txt

# sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev


mkdir .dependencies
cd .dependencies
git clone --depth 1 -b 20220623.1 https://github.com/abseil/abseil-cpp.git
git clone --depth 1 -b 3.4.0 https://gitlab.com/libeigen/eigen.git
git clone --depth 1 -b v2.10.2 https://github.com/pybind/pybind11.git
cd ..

# git clone --depth 1 -b v0.1.97 https://github.com/google/sentencepiece.git
# cd sentencepiece
# mkdir build
# cd build
# cmake ..
# make -j
# sudo make install
# sudo ldconfig -v

cd src/models/components/masking
rm -rf tmp
mkdir tmp
cd tmp
cmake ..
cmake --build . -- -j
mv ./*.so ..
cd ../../../../../
# test masking
# if you are installing on SLURM, do the following commands on worker nodes instead of the login nodes
# otherwise, you will get glibc error.
# python -m src.models.components.constrained_decoding
