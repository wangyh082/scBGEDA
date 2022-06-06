# scBGEDA: Deep Single-cell Clustering Analysis via Dual Denoising Autoencoder with Bipartite Graph Ensemble Clustering

# Contents
- [Overview](#overview)
- [Architecture](#Architecture)
- [Installation](#Installation)
- [Data availability](#Data-availability)
- [Usage](#Usage)
- [Key Functions](#Key-Functions)
- [Results](#Results)
- [Contact](#Contact)

# Overview

Single-cell RNA sequencing (scRNA-seq) is an increasingly popular technique for transcriptomic analysis of gene expression at the single-cell level. Cell-type clustering is the first crucial task in the analysis of scRNA-seq data to facilitate accurate identification of cell types and to study the characteristics of their transcripts. Recently, several computational models based on a deep autoencoder and the ensemble clustering have been developed to analyze scRNA-seq data. However, current deep autoencoders are not sufficient to learn the latent representations of scRNA-seq data, and obtaining consensus partitions from these feature representations remains under-explored. To address this challenge, we propose a single-cell deep clustering model via a dual denoising autoencoder with bipartite graph ensemble clustering called scBGEDA, to identify specific cell populations in single-cell transcriptome profiles. First, a single-cell dual denoising  autoencoder network is proposed to project the data into a compressed low-dimensional space and that can learn feature representation via explicit modeling of synergistic optimization of ZINB reconstruction loss and denoising reconstruction loss. Then, a bipartite graph ensemble clustering algorithm is designed to exploit the relationships between cells and the learned latent embedded space by means of a graph-based consensus function. Multiple comparison experiments were conducted on fifteen scRNA-seq datasets from different sequencing platforms using a variety of clustering metrics. The experimental results indicated that scBGEDA outperforms other state-of-the-art methods on these datasets, and also demonstrated scalability to large scale scRNA-seq datasets. Moreover, scBGEDA was able to identify cell-type specific marker genes and provide functional genomic analysis by quantifying the influence of genes on cell clusters, bringing new insights to identify cell types and characterize the scRNA-seq data from different perspectives.

# Architecture
![Image text](https://github.com/wangyh082/scBGEDA/blob/main/frame.jpg)

The overall workflow of the scBGEDA pipeline, comprising three components: the data preprocessing mechanism, the single-cell dual denoising autoencoder network, and the bipartite graph ensemble clustering method.

# Installation

1. Requirements:

```
[python 3.6+]
[tensorflow 2.6.0]
[keras 2.6.0]
[numpy 1.19.5]
[jgraph 0.2.1]
[scipy 1.5.4]
[scanpy 1.7.2]
[pathos 0.2.8]
[tqdm 4.64.0]
[python-dateutil 2.8.2]
```

2. Installation:

To meet the requirements, we recommend user to use [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment:
```
# Download scBGEDA from GitHub and create an environment:

git clone https://github.com/wangyh082/scBGEDA.git
conda create -n scBGEDA python=3.6
```

```
# To activate environment:

conda activate scBGEDA
```

```
# To install all required packages:

pip install -r requirements.txt
```

```
# To exit environment:

conda deactivate
```

# Data availability

The real single-cell RNA-seq datasets and source codes for our analyses are freely available at (https://figshare.com/articles/software/scBGEDA/19657911).

# Usage

## Command Lines

```  
SDDA.py [-h] [--dataname DATANAME] [--highly_genes HIGHLY_GENES]
               [--random_seed RANDOM_SEED] [--dims DIMS] [--alpha ALPHA]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--pretrain_epoch PRETRAIN_EPOCH] [--noise_sd NOISE_SD]
               [--gpu_option GPU_OPTION]

optional arguments:
  -h, --help Show this help message and exit
  --dataname DATANAME The input dataname
  --highly_genes HIGHLY_GENES The number of highly variable genes
  --random_seed RANDOM_SEED The random seeds which are used to select the cells randomly to ensure the fairness
  --dims DIMS The dimensions of the hidden layers
  --alpha ALPHA The hyperparameter to control the relative impact of two decoders
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --pretrain_epoch PRETRAIN_EPOCH 
  --noise_sd NOISE_SD
  --gpu_option GPU_OPTION
```  

## Examples:
The parameters including "dataname", "highly_genes", "random_seed", "dims", "alpha", "learning_rate", "batch_size", "pretrain_epoch", "noise_sd", "gpu_option" can be set as you like in your command lines.

We set default settings for each parameter, and if the parameter is not set to the given value, then it will use the default settings. 

Take the dataset "Adam"  as an example.

Do not use the default value:

python SDDA.py --dataname Adam --highly_genes 2000 --random_seed "1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000" --dims "256, 32" --alpha 0.001 --learning_rate 0.0001 --batch_size 256 --pretrain_epoch 1000 --noise_sd 1.5 --gpu_option “2"

Using the default value：

python SDDA.py --dataname Adam

After that, to generate a more efficient clustering result, use the following code:


Output explanation:
The final output reports the clustering performance and the median values of ARI and NMI is provided, respectively.

# Key Functions

The key functions of the source code and their detailed description.

| Function     | Description                                   |
| ------------ | --------------------------------------------- |
| preprocess.py| Function of the first module of scBGEDA       |
| SDDA.py      | Function of the second module of scBGEDA      |
| network.py   | Single-cell Dual Denoising Autoencoder Network|
| loss.py      | the loss functions of the network             |
| utils.py     | the utility functions of the network          |
| main.m       | Main function of the third module of scBGEDA  |
| BGEC.m       | Bipartite Graph Ensemble Clustering           |
| rand_index.m | Computing ARI values after clustering         |
| computeNMI.m | Computing NMI values after clustering         |

# Results
Multiple comparison experiments were conducted on fifteen scRNA-seq datasets from different sequencing
platforms using a variety of clustering metrics. The experimental results indicated that scBGEDA
outperforms other state-of-the-art methods on these datasets, and also demonstrated scalability to large
scale scRNA-seq datasets. 

# Contact

If you have any suggestions or questions, please email me at wangyh082@hebut.edu.cn.


