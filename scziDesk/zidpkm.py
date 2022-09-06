import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
from preprocess import *
from network import *
from utils import *
import argparse
import time
import scib
import umap
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpu_id = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

if __name__ == "__main__":
    random_seed = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]
    #default parameter#
    distribution="ZINB"
    self_training=True
    dims=[500,256,64,32]
    highly_genes=500
    alpha=1e-3
    gamma=1e-3
    learning_rate=1e-4
    batch_size=256
    update_epoch=10
    pretrain_epoch=1000
    funetrain_epoch=2000
    t_alpha=1.0
    noise_sd=1.5
    error=0.001
    gpu_option="3"
    dataset_list=[
        "Adam", "Bach", "Chen", "Klein","Muraro", "Plasschaert", "Pollen", "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Heart", "Quake_Smart-seq2_Limb_Muscle",
       "Quake_Smart-seq2_Lung", "Quake_Smart-seq2_Trachea",  "Quake_10x_Bladder", "Quake_10x_Limb_Muscle","Quake_10x_Spleen", "Quake_10x_Trachea", "Romanov","Tosches_turtle","Wang_Lung", "Young"]
    for name in dataset_list:

        print(name)

        import pandas as pd
        filepath = '/home/scBGEDA/dataset/' + name
        X, Y = prepro(filepath + '/data.h5')
        X = np.ceil(X).astype(np.int)
        count_X = X

        adata = sc.AnnData(X)
        adata.obs['Group'] = Y
        adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
        X = adata.X.astype(np.float32)
        Y = np.array(adata.obs["Group"])
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
        size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
        cluster_number = int(max(Y) - min(Y) + 1)

        result = []

        for seed in random_seed:
            start = time.time()
            np.random.seed(seed)
            tf.compat.v1.reset_default_graph()
            chencluster = autoencoder(name, distribution, self_training, dims, cluster_number, t_alpha,
                                        alpha, gamma, learning_rate, noise_sd)
            chencluster.pretrain(X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option)

            chencluster.funetrain(X, count_X, size_factor, batch_size, funetrain_epoch, update_epoch, error)
            end = time.time()
            print(end-start)
            pre = np.array(chencluster.latent_repre)
            import scipy.io as scio
            scio.savemat(
                '/home/scBGEDA/Results/scziDesk/temp/{}_seed_{}.mat'.format(name,seed),
                {'data': pre, 'label': chencluster.Y_pred})
            nmi = np.round(normalized_mutual_info_score(Y,chencluster.Y_pred), 5)
            ari = np.round(adjusted_rand_score(Y, chencluster.Y_pred), 5)
            result.append([name,seed, end-start, nmi, ari])
        output = np.array(result)
        output = pd.DataFrame(output,columns=["dataset name", "seed",
                                                    "time", "NMI","ARI"])
        output.to_csv(r'/home/scBGEDA/Results/scziDesk/metrics/{}.csv'.format(name))
        print(output)















