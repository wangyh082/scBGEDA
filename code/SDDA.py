import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from preprocess import *
from network import *
from utils import *
import argparse

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


def train_DualSCDC(x_nomalized, count_X, size_factor, y, seed, name, dims, alpha, learning_rate, noise_sd, batch_size, pretrain_epoch, gpu_option):
    if x_nomalized.shape[0] > 1e4:
        batch_size = 2*batch_size
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    temp_dims = args.dims.split(',')
    temp = [x_nomalized.shape[1]]
    temp.extend(temp_dims)
    dimsNew = list(map(int,temp))
    ae = autoencoder(dimsNew, alpha, learning_rate, noise_sd)
    ae.pretrain(x_nomalized, count_X, size_factor, batch_size, pretrain_epoch, gpu_option)
    pre = np.array(ae.latent_repre)
    import scipy.io as scio
    scio.savemat(
        '../mdata/{}_seed_{}.mat'.format(name, seed),
        {'data': pre, 'label': y})
    del ae


if __name__ == "__main__":
    from pathos.multiprocessing import ProcessingPool as Pool
    import os

    random_seed_list = "1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000"
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default="Adam", type=str)
    parser.add_argument("--highly_genes", default=2000, type=int)
    parser.add_argument("--random_seed", default=random_seed_list)
    parser.add_argument("--dims", default="256,32")
    parser.add_argument("--alpha", default=0.00001, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pretrain_epoch", default=1000, type=int)
    parser.add_argument("--noise_sd", default=1.5, type=float)
    parser.add_argument("--gpu_option", default="2")

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    import pandas as pd

    name = args.dataname
    dims = args.dims
    alpha = args.alpha
    learning_rate = args.learning_rate
    noise_sd = args.noise_sd
    batch_size = args.batch_size
    pretain_epoch =  args.pretrain_epoch
    gpu_option = args.gpu_option

    filepath = '../dataset/' + name
    if os.path.exists(filepath + '/data.h5'):
        x_raw, y = prepro(filepath + '/data.h5')
        x_raw = np.ceil(x_raw).astype(np.int)
        count_X = x_raw.copy()
        adata = sc.AnnData(x_raw.copy())
        adata.obs['Group'] = y

        adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        x_nomalized = adata.X.astype(np.float32)
        x_normalized_DC = x_nomalized[1]

        Y = np.array(adata.obs["Group"])
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
        size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
        cluster_number = int(max(Y) - min(Y) + 1)

        temp_random = args.random_seed
        temp_random1 = temp_random.split(',')
        seed_list = list(map(int,temp_random1))
        x_train = [x_nomalized.copy() for i in range(len(seed_list))]
        count_x_train = [count_X.copy() for i in range(len(seed_list))]
        size_factor_train = [size_factor.copy() for i in range(len(seed_list))]
        y_train = [Y for i in range(len(seed_list))]
        namelist = [name for i in range(len(seed_list))]
        dimslist = [dims for i in range(len(seed_list))] 
        alphalist = [alpha for i in range(len(seed_list))] 
        learning_ratelist = [learning_rate for i in range(len(seed_list))] 
        noise_sdlist = [noise_sd for i in range(len(seed_list))]
        batch_sizelist = [batch_size for i in range(len(seed_list))]
        pretrain_epochlist = [pretain_epoch for i in range(len(seed_list))]
        gpu_optionlist = [gpu_option for i in range(len(seed_list))]
        with Pool(len(seed_list) + 3) as p:
            rs = p.map(train_DualSCDC, x_train, count_x_train, size_factor_train, y_train, seed_list, namelist, dimslist, alphalist, learning_ratelist, noise_sdlist, batch_sizelist, pretrain_epochlist, gpu_optionlist)
