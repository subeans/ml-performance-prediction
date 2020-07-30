"""Benchmark the training of VGG16 convpolutional neural
network on synthetic data. Returns images per time as
result metric.
"""

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
from utils_tf import merge_timeline
from utils_tf import nets_vggsmall as vggsmall

class VGGsmall(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self, args):
        """Initialize VGG16
        Args:
            args: Input arguments
        """

        self.imgwidth = args.imgsize
        self.imgheight = args.imgsize
        self.numclasses = args.numclasses
        self.precision = args.precision
        self.batchsize = args.batchsize
        self.optimizer = args.optimizer
        self.devlist = args.devlist
        self.comment = args.comment


    def create_benchmark_op(self):
        """Create benchmark operation
        Returns:
            train_op: Operation for training VGG-16 for one step
            g: VGG-16 graph
        """

        # Generate synthetic data
        datatype = eval('np.float%d' %(self.precision))

        g = tf.Graph()
        with g.as_default():
            batch_data = np.zeros(
                    [self.batchsize,self.imgwidth,self.imgheight,3],
                    dtype=datatype)
            batch_label = slim.one_hot_encoding(
                    np.zeros(self.batchsize,dtype=np.int32),
                    self.numclasses)

            # Define model from slim.nets.vgg
            model = vggsmall.vgg_small

            # Run data through model
            prediction, _ = model(
                    batch_data,
                    num_classes=self.numclasses)

            # Get loss
            loss = tf.losses.softmax_cross_entropy(
                    batch_label,
                    prediction)

            # Define train_op
            #SGD, RMSPROP, Adadelta, Adagrad, Momentum, Adam
            if self.optimizer=='sgd':
                opt = tf.train.GradientDescentOptimizer(0.01)
            elif self.optimizer=='rmsprop':
                opt = tf.train.RMSPropOptimizer(0.001, 0.9)
            elif self.optimizer=='adadelta':
                opt = tf.train.AdadeltaOptimizer(0.001, 0.9)
            elif self.optimizer=='adagrad':
                opt = tf.train.AdagradOptimizer(0.001)
            elif self.optimizer=='adam':
                opt = tf.train.AdamOptimizer(0.001, 0.9)
            elif self.optimizer=='momentum':
                opt = tf.train.MomentumOptimizer(0.001,0.9)
            

            train_op = slim.learning.create_train_op(
                    loss,
                    optimizer = opt,
                    summarize_gradients=True)
        return train_op, g

    def generate_logtext(self, timeUsed, mem):
        """Function that generates comma separated text for a logfile"""

        logtext = ('VGG-small, %d, %d, %d, %.3f, %.3f, %.3f, %s\n'
                %(self.imgwidth,
                self.precision,
                self.batchsize,
                timeUsed*1000,
                self.batchsize/timeUsed,
                mem/1e6,
                self.comment))
        return logtext
