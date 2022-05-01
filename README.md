# scBGEDA: Deep Single-cell Clustering Analysis via Dual Denoising Autoencoder with Bipartite Graph Ensemble Clustering

Single-cell RNA sequencing (scRNA-seq) is an increasingly popular technique for the transcriptomic analysis of gene expression at the single-cell level. Cell clustering is the first crucial task in the analysis of scRNA-seq data to facilitate more accurate identification of cell types and to study the characteristics of transcripts. Recently, several computational models based on the deep autoencoder have been developed to cluster cell types from a mass of heterogeneous cells. However, it is still challenging to provide robust and accurate clustering of scRNA-seq data due to the unstable feature representative information of the deep autoencoder. To address this challenge, we propose a deep single-cell clustering model via dual denoising autoencoder with bipartite graph ensemble clustering called scBGEDA, to identify cell populations in single-cell transcriptome profiles. In the first, single-cell dual denoising  autoencoder network is proposed to project the data onto a compressed low-dimensional space, which can learn feature representation via explicit modelling of synergistic optimization of ZINB reconstruction loss and denoising reconstruction loss. After that, a bipartite graph ensemble clustering algorithm is designed to exploit the relationships between cells and the learned latent embedded space under the graph-based consensus function.Multiple comparison experiments are conducted on fifteen scRNA-seq datasets from different sequencing platforms on a variety of clustering metrics. The experimental results indicate that scBGEDA outperforms other state-of-the-art methods on those datasets, and it has also demonstrated sufficient capacity for analysis of large scale scRNA-seq datasets. Moreover, scBGEDA can identify cell-type specific markers and function genomic analysis by quantifying the influence of genes on cell clusters, which can bring new insights into identifying cell types and characterizing the scRNA-seq data from different perspectives.

## Architecture
![Image text](https://github.com/wangyh082/scBGEDA/blob/main/frame.jpg)

## Requirements

scBGEDA is written in Python3 and requires the following dependencies to be installed:

Tensorflow 1.14

Keras 2.2

## Data availability

The real single-cell RNA-seq datasets and source codes for our analyses are freely available at (https://figshare.com/articles/software/scBGEDA/19657911).

## Usage

The dataset "Adam" is given as an example. 

First, you can run the following code in your command lines:

python SDDA.py 

Then you can will obtain ten mat files in ten random seeds. 

Next, you can run the main.m using Matlab to get the clustering result for "Adam" dataset. 

Finally, you can achieve the median values of ARI and NMI, respectively.
## Key Functions

The key functions of the source code and their detailed description.

| Function     | Description                                   |
| ------------ | --------------------------------------------- |
| main.m       | Main function of our method.                  |
| InfFS_U.m    | Unsupervised Graph-based Feature Ranking      |
| ComputeGM.m  | Computing the graph-based linking matrix      |
| Preprocess.m | Generating basic partitions                   |
| exMeasure.m  | Computing NMI and ARI values after clustering |

## Contact

If you have any suggestions or questions, please email me at wangyh082@hebut.edu.cn.


