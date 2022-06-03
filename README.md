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
[IPython 7.16.3]
[tensorflow 2.6.2]
[keras 2.6.0]
[scanpy 1.7.2]
[scipy 1.5.4]
[tqdm 4.64.0]
[packaging 21.3]
[python-dateutil 2.8.2]
[jgraph 0.2.1]
[pillow 8.4.0]
[sklearn]
```

2. Installation:
2.1 Build from sources
To meet the requirements, we recommend user to use either (1) [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment:
```
# Download SapCell from GitHub and install all required packages:

git clone https://github.com/BiomedicalMachineLearning/Spacell.git
cd scBGEDA
conda env create -f requirements.yml
```

```
# To activate environment:

conda activate SpaCell
```

```
# To exit environment:

conda deactivate
```

# Data availability

The real single-cell RNA-seq datasets and source codes for our analyses are freely available at (https://figshare.com/articles/software/scBGEDA/19657911).

# Usage

The dataset "Adam" is given as an example. 

First, you can run the following code in your command lines:

python SDDA.py 

Then you can will obtain ten mat files in ten random seeds. 

Next, you can run the main.m using Matlab to get the clustering result for "Adam" dataset. 

Finally, you can achieve the median values of ARI and NMI, respectively.
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


