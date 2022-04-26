import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from preprocess import *
from network import *
from utils import *


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def train_DualSCDC(x_nomalized,count_X,size_factor,y,seed,name):
                batch_size = 256
                if x_nomalized.shape[0]>1e4:
                    batch_size = 512
                pretrain_epoch=1000
                gpu_option="2"
                np.random.seed(seed)
                alpha_list=[1e-5]
                for alpha in alpha_list:
                    tf.compat.v1.reset_default_graph()
                    ae= autoencoder(dims=[x_nomalized.shape[1],256,32],alpha=alpha)
                    ae.pretrain(x_nomalized, count_X, size_factor, batch_size, pretrain_epoch, gpu_option)
                    pre=np.array(ae.latent_repre)
                    import scipy.io as scio
                    scio.savemat(
                        '/home/mdata/{}_seed_{}_{}_{}.mat'.format(name, seed, x_nomalized.shape[1], alpha),
                        {'data': pre, 'label': y})
                    del ae





if __name__ == "__main__":
    from pathos.multiprocessing import ProcessingPool as Pool
    import os

    dataset_list=[
        "Adam"]
    dataset_list=list(set(dataset_list))
    print(dataset_list)

    highly_genes=2000
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')
    n_feature_list = [500]
    for name in dataset_list:

        print(name)

        import pandas as pd

        filepath = '/home/dataset/' + name
        if os.path.exists(filepath + '/data.h5'):
            x_raw, y = prepro(filepath + '/data.h5')
            x_raw = np.ceil(x_raw).astype(np.int)
            count_X = x_raw.copy()
        
            adata = sc.AnnData(x_raw.copy())
            adata.obs['Group'] = y



            adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
            x_nomalized = adata.X.astype(np.float32)
            print(type(x_nomalized))
            x_normalized_DC = x_nomalized[1]

            Y = np.array(adata.obs["Group"])
            high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
            count_X = count_X[:, high_variable]
            size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
            cluster_number = int(max(Y) - min(Y) + 1)

            seed_list = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]
            x_train = [x_nomalized.copy() for i in range(len(seed_list))]
            count_x_train = [count_X.copy() for i in range(len(seed_list))]
            size_factor_train = [size_factor.copy() for i in range(len(seed_list))]
            y_train = [Y for i in range(len(seed_list))]
            namelist = [name for i in range(len(seed_list))]
            with Pool(len(seed_list) + 3) as p:
                rs = p.map(train_DualSCDC, x_train, count_x_train, size_factor_train, y_train, seed_list, namelist)















