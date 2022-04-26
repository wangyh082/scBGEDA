# scBGEDA: Deep Single-cell Clustering Analysis via Dual Denoising Autoencoder with Bipartite Graph Ensemble Clustering

## Architecture
![Image text](https://github.com/wangyh082/scBGEDA/blob/main/frame.jpg)

## Requirements

scBGEDA is written in Python3 and requires the following dependencies to be installed:

Tensorflow 1.14

Keras 2.2

## Data availability

The real single-cell RNA-seq datasets and source codes for our analyses are freely available at (https://figshare.com/account/home#/projects/137832).

## Usage

The dataset "Adam" is given as an example. 

First, you can run the following code in your command lines:

python SDDA.py 

Then you can will obtain ten mat files in ten random seeds. 

Next, you can run the main.m using Matlab to get the clustering result for "Adam" dataset. 

Finally, you can achieve the median values of ARI and NMI, respectively.

## Contact

If you have any suggestions or questions, please email me at wangyh082@hebut.edu.cn.


