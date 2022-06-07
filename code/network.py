import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import GaussianNoise, Dense, Activation
from sklearn.cluster import KMeans
from loss import *
import os


class autoencoder(object):
    def __init__(self, dims, alpha, learning_rate, noise_sd, init='glorot_uniform', act='relu'):
        self.dims = dims
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act

        self.n_stacks = len(self.dims) - 1

        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))

        self.h = self.x
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)

        for i in range(self.n_stacks - 1):
            self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
            self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
            self.h = Activation(self.act)(self.h)

        self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)

        self.h1 = self.latent
        for i in range(self.n_stacks - 1, 0, -1):
            self.h1 = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(self.h1)
        self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h1)
        self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(
            self.h1)
        self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h1)
        self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
        self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)

        self.h2 = self.latent
        for i in range(self.n_stacks - 1, 0, -1):
            self.h2 = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init,
                            name='decoder_recon_%d' % i)(self.h2)
        self.recon=Dense(units=self.dims[0], activation='relu', kernel_initializer=self.init,
                            name='decoder_recon_x')(self.h2)
        self.pretrain_loss=self.likelihood_loss+self.alpha*tf.keras.losses.MSE( self.mean, self.recon)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.pretrain_loss)

    def pretrain(self, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("begin the pretraining")
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        self.sess = tf.Session(config=config_)
        self.sess.run(init)

        self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        pre_index = 0
        for ite in range(pretrain_epoch):
            while True:
                if (pre_index + 1) * batch_size > X.shape[0]:
                    last_index = np.array(list(range(pre_index * batch_size, X.shape[0])) + list(
                        range((pre_index + 1) * batch_size - X.shape[0])))
                    _, likelihood_loss, latent = self.sess.run([self.pretrain_op, self.likelihood_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[last_index],
                            self.x: X[last_index],
                            self.x_count: count_X[last_index]})
                    self.latent_repre[last_index] = latent
                    pre_index = 0
                    break
                else:
                    _, likelihood_loss, latent = self.sess.run(
                        [self.pretrain_op, self.likelihood_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x: X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x_count: count_X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)]})
                    self.latent_repre[(pre_index * batch_size):((pre_index + 1) * batch_size)] = latent
                    pre_index += 1


