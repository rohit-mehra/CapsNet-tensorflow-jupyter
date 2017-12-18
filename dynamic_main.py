"""Implementaion with Dynamic Loop in tensorflow
Test acc: 99.38% """

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 0. Training flags:
train_round = 1
ckpt = './checkpoints/capsnet_dynamic.ckpt'  # Checkpoint file
resume = False  # True if you have the above file
do_test_first = False  # do test eval
save_after = 430  # and validate # checkpointing global step

batch_size = 128
epochs = 300

# 1. Optional: safe max
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.93

# 2. Optional: reproduce
np.random.seed(27)
tf.set_random_seed(27)

# 3. Data: temporary
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True, reshape=False)
input_size = 28

# 4. Architecture parameters: hardcoded as in paper
pc_units = 32
pc_dims = 8
pc_capsules = pc_units * 6 * 6 # right now hard-coded

dc_units = 10
dc_dims = 16
epsilon = 1e-8

# to output activities of local feature detectors
conv_0_params = {'filters': 256, 'kernel_size': 9, 'strides': 1,
                 'padding': 'valid', 'activation': tf.nn.relu
                }


# to output primary capsules
conv_pc_params = {'filters': pc_units * pc_dims, 'kernel_size': 9, 'strides': 2,
                  'padding': 'valid', 'activation': tf.nn.relu # <-------- doubt/ works well with relu/ in paper not mentioned
                 }

# arch params
arch_params = {'inp_dims': input_size,
               'pc_dims': pc_dims,
               'out_dims': dc_dims,
               'classes': dc_units,
               'm_plus': 0.9,
               'm_minus': 0.1,
               'lambda_val': 0.5,  # down weight of the loss for absent digit classes # <------------
               'epsilon': epsilon,
               'recon_loss_w': 0.0005, # reconstruction loss weight # <------------
               'num_iters': 3,
               'lr':1e-3,
              }


# 5. Activation function: squash
def squash(in_tensor, axis=-1, name='squash'):
    """Squashing Function: Squash magnitude along an axis but preserve orientation"""
    with tf.name_scope(name):
        norm_square = tf.reduce_sum(tf.square(in_tensor), axis=axis, keep_dims=True, name='norm_square')  # [N, 1, J, 1, 1] if axis = -2
        squash_factor = norm_square / (1 + norm_square) # scalar factor to squash magnitudes b/w 0-1
        orientation = in_tensor / tf.sqrt(norm_square + epsilon) # unchanged
        out_tensor = squash_factor * orientation
        return out_tensor


# 6. Agreement
def prediction_agreement(next_layer_out_predictions, next_layer_outs, name='agree_by_dot'):
    """Agreement: Explain away this layer values,
       by agreeing predictions of this layer for next layer outputs with next layers real outputs."""
    with tf.name_scope(name):

        # Tile next_layer_outs for each i from I=1152, as we have to see that from 1152 capsules which agree max.
        next_layer_outs = tf.tile(next_layer_outs, multiples=[1, 1152, 1, 1, 1])  # [N, 1152, 10, 16, 1]

        # [N, 1152, 10, 1, 16] x [N, 1152, 10, 16, 1] = [N, 1152, 10, 1, 1]
        agreement = tf.matmul(next_layer_out_predictions, next_layer_outs, transpose_a=True)

        return agreement


# 7. Route
def routing(capsule_outs, num_iters=3):
    """Routing: Learn routing weights and route, dynamic connection between two capsule layers."""

    batch_size = tf.shape(capsule_outs)[0] # will know only during run

    with tf.name_scope('predict_next_layer_outs'):

        # Tensor containg all transformation matrices W_ij
        W_IJ = tf.Variable(tf.random_normal([1, pc_capsules, dc_units, pc_dims, dc_dims], stddev=0.01))  # [1, 1152, 10, 8, 16]
        W_IJ = tf.tile(W_IJ, multiples=[batch_size, 1, 1, 1, 1])  # [N, 1152, 10, 8, 16] <-- IJ[8, 16]

        # Predict Next Layer's Output
        next_layer_out_predictions = tf.matmul(W_IJ, capsule_outs, transpose_a=True)

    with tf.name_scope('routing'):
        # [N, 1152, 10, 1, 1] Constant Value Tensor
        prior_IJ = tf.zeros([batch_size, pc_capsules, dc_units, 1, 1],
                                        dtype=np.float32, name='routing_prior')

        def execute(prior_IJ, next_layer_out_predictions, next_layer_outs, counter):

            c_IJ = tf.nn.softmax(prior_IJ, dim=2) # [N, 1152, 10, 1, 1]
            next_layer_outs = tf.reduce_sum((c_IJ * next_layer_out_predictions), axis=1, keep_dims=True)  # [N, 1, 10, 16, 1]
            next_layer_outs = squash(next_layer_outs, axis=-2)
            routing_weights = tf.add(prior_IJ, prediction_agreement(next_layer_out_predictions, next_layer_outs))
            return routing_weights, next_layer_out_predictions, next_layer_outs, tf.add(counter, 1)

        def check(prior_IJ, next_layer_out_predictions, next_layer_outs, counter):
            return tf.less(counter, num_iters)

        counter = tf.constant(1)
        # dummy Constant Value Tensor to hold digit caps outputs
        next_layer_outs = tf.zeros([batch_size, 1, dc_units, dc_dims, 1], dtype=np.float32, name='dc_out') # [N, 1, 10, 16, 1]
        _, _, next_layer_outs, counter = tf.while_loop(check, execute, [prior_IJ, next_layer_out_predictions, next_layer_outs, counter])

        return next_layer_outs  # [N, 1, 10, 16, 1]


class CapsNet:
    def __init__(self, conv_0_params, conv_pc_params, inp_dims=28, pc_dims=8, out_dims=16, classes=10, num_iters=3,
                 recon_loss_w=0.0005, epsilon=1e-9, m_plus=0.9, m_minus=0.1, lambda_val=0.5, lr=1e-3, scope='CapsNet'):

        print('Learning Rate={}, m_plus={}, m_minus={}, lambda_val={}'.format(lr,
                                                                              m_plus,
                                                                              m_minus,
                                                                              lambda_val))

        with tf.variable_scope(scope):
            self.X = tf.placeholder(tf.float32,
                                    shape=[None, inp_dims, inp_dims, 1],
                                    name='inputs')
            self.Y = tf.placeholder(tf.float32,
                                    shape=[None, classes],
                                    name='one_hot_labels')

            self.is_training = tf.placeholder_with_default(True, shape=[], name='bool_for_masking')

            # Get feature maps for basic features
            with tf.variable_scope('conv_0_layer'):
                self.conv_0 = tf.layers.conv2d(self.X, **conv_0_params)  # [N, 20, 20, 256]

            # Get Primary Capsules
            with tf.variable_scope('conv_pc_layer'):

                # raw caps
                self.conv_pc = tf.layers.conv2d(self.conv_0, **conv_pc_params)   # [N, 6, 6, 256]
                batch_size = tf.shape(self.conv_pc)[0]
                self.conv_pc = tf.reshape(self.conv_pc, shape=[batch_size, -1, pc_dims])  # [N, 1152, 8]

                # activated caps
                self.pcaps = squash(self.conv_pc) # squash 1152 capsules along axis=-1
                self.pcaps = tf.reshape(self.pcaps, shape=[batch_size, -1, 1, pc_dims, 1])

            with tf.variable_scope('digit_caps'):

                # tile pcaps activations w.r.t each unit in digit caps
                self.tiled_pcaps = tf.tile(self.pcaps, multiples=[1, 1, classes, 1, 1])
                self.digit_caps_out = tf.squeeze(routing(self.tiled_pcaps, num_iters), axis=[1, -1], name='distributed_prediction') # [N, 10, 16]

            # [N, 10, 16] --> [N, 10] length of vectors for actual predictions
            self.pred_lengths = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps_out) + epsilon, axis=2, keep_dims=False))

            # [N, 10] --> # [N,]
            self.predictions = tf.squeeze(tf.argmax(self.pred_lengths,
                                                    axis=1, output_type=tf.int32, name='predictions'))
            self.one_hot_predictions = tf.one_hot(self.predictions, depth=classes, name='one_hot_predictions')

            # [N, 10] * [N, 10] = [N, 10]
            self.m_plus = self.Y * tf.square(tf.maximum(0., m_plus - self.pred_lengths))  # [N, 10]
            self.m_minus = lambda_val * (1 - self.Y) * tf.square(tf.maximum(0., self.pred_lengths - m_minus))  # [N, 10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(self.m_plus + self.m_minus, axis=-1), name='margin_loss')

            self.recon_target = tf.cond(self.is_training,
                                        lambda: self.Y, # Train True use Labels
                                        lambda: self.one_hot_predictions,
                                        name='recon_target')

            with tf.variable_scope('masking'):
                # WAY 1:
                self.masked_pred = tf.matmul(self.digit_caps_out, tf.reshape(self.recon_target, shape=[-1, 10, 1]),
                                             transpose_a=True, name='masked_pred')  # [N, 16, 10] x [N, 10, 1] = [N, 16, 1]
#                 # WAY 2:
#                 self.masked_pred = tf.multiply(self.digit_caps_out,
#                                                tf.reshape(self.recon_target, shape=[-1, 10, 1]),
#                                                name='masked_pred') # [N, 10, 16] x [N, 10, 1] = [N, 10, 16]

            with tf.variable_scope('decoder'):
                # [N, 16, 1] --> [N, 16]
                self.decoder_inp = tf.reshape(self.masked_pred,
                                         shape=[-1, out_dims]) # WAY 2: shape = [-1, out_dims * classes]
                self.fc1 = tf.layers.dense(self.decoder_inp,
                                           units=512,
                                           activation=tf.nn.relu,
                                           name='fc1')
                self.fc2 = tf.layers.dense(self.fc1,
                                           units=1024,
                                           activation=tf.nn.relu,
                                           name='fc2')

                self.pred_X = tf.layers.dense(self.fc2,
                                              units=inp_dims * inp_dims,
                                              activation=tf.nn.sigmoid,
                                              name='pred_X')  # [N, 28*28]

                self.pred_image = tf.reshape(
                    self.pred_X, shape=[-1, inp_dims, inp_dims, 1])

            # Backward
            self.reconstruction_loss = tf.reduce_sum(tf.square(tf.reshape(self.X,
                                                                          shape=[-1, inp_dims * inp_dims]) - self.pred_X),
                                                     name='reconstruction_loss')

            self.loss = tf.add(self.margin_loss, recon_loss_w * self.reconstruction_loss,
                               name='loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step)

            # function variables/ops:
            # [N, 10] --> # [N,]
            self.true = tf.squeeze(tf.argmax(self.Y, axis=1,
                                             output_type=tf.int32, name='true_values'))

            self.correct = tf.cast(tf.equal(self.true,
                                            self.predictions), dtype=tf.float32)
            self.acc = tf.reduce_mean(self.correct) * 100
            # meta variables:
            self.tvars = tf.trainable_variables()

    def predict(self, xs, ys=None, is_training=True, get_recon_images=False, sess=None):
        """Returns Predicted Number and Reconstructed Image."""
        sess = sess or tf.get_default_session()

        if is_training and ys is None and get_recon_images:
            print("Would need true one hot encoded labels..for masking..")
            return

        if ys is None:
                ys = np.zeros([10, 10], dtype=np.float32) # dummy

        if get_recon_images:
            return sess.run([self.predictions, self.pred_image], feed_dict={self.X: xs,
                                                                            self.Y: ys,
                                                                            self.is_training: is_training})
        else:
            return sess.run(self.predictions, feed_dict={self.X: xs})

    def accuracy(self, xs, ys, sess=None):
        """Predicts and returns accuracy at current state."""
        sess = sess or tf.get_default_session()
        return sess.run(self.acc, feed_dict={self.X: xs, self.Y: ys})

    def learn(self, xs, ys, is_training=True, val_xs=None, val_ys=None, sess=None):
        """Train Step"""
        sess = sess or tf.get_default_session()

        if val_xs is not None and val_ys is not None:
            val_acc = self.accuracy(val_xs, val_ys, sess=sess)
            return val_acc
        else:
            train_acc, loss, _ = sess.run([self.acc, self.loss, self.train_op], feed_dict={
                                          self.X: xs, self.Y: ys, self.is_training: is_training})
            return train_acc, loss


# Training:
"""
55,000 data points of training data (mnist.train),
10,000 points of test data (mnist.test)
and 5,000 points of validation data (mnist.validation).
"""

test_batch_size = 100

tf.reset_default_graph()

caps_net = CapsNet(conv_0_params, conv_pc_params, **arch_params)
print('Network Built..')

saver = tf.train.Saver(var_list=caps_net.tvars)
file = open('avg_log_{}.csv'.format(train_round), 'w')
file.write('step,avg_train_acc,step_val_acc\n')

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    if resume:
        saver.restore(sess, ckpt)
        print('RESUMED..')

    if do_test_first:
        ta = 0
        total_steps = len(mnist.test.images) // test_batch_size

        for step in range(1, total_steps + 1):
            val_x, val_y = mnist.test.next_batch(test_batch_size)
            test_batch_accuracy = caps_net.accuracy(val_x, val_y, sess)
            ta += test_batch_accuracy
            print('{},{:>3.3f}'.format(step, test_batch_accuracy))

        print('Test Accuracy: ', ta / total_steps)

    print('*' * 90)
    print('Training..learning_rate = {}, total_epochs = {}'.format(arch_params['lr'], epochs))
    total_steps_approx = save_after * epochs
    val_acc_base = -np.inf
    ta = 0
    for step in range(1, total_steps_approx + 1):

        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=True)
        global_step = sess.run(caps_net.global_step)

        if step % save_after == 0:
            print('Validation in progress..')
            va = 0
            total_steps = len(mnist.validation.images) // test_batch_size
            for v in range(total_steps):
                val_x, val_y = mnist.validation.next_batch(test_batch_size, shuffle=True)
                val_accuracy = caps_net.accuracy(val_x, val_y, sess)
                va += val_accuracy

            print('Validation log in..avg_log_{}.csv'.format(train_round))
            va = va / total_steps
            file.write('{},{:>3.3f},{:>3.3f}\n'.format(step, ta / save_after,
                                                       va))
            file.flush()
            ta = 0

            if va > val_acc_base:
                saver.save(sess, ckpt)
                print('SAVED..old best val acc = {}, new best val acc = {}'.format(val_acc_base, va))
                val_acc_base = va

        else:
            train_accuracy, train_loss = caps_net.learn(batch_x, batch_y,
                                                        sess=sess)
            print('{},{:>3.3f},{:>3.3f}'.format(step, train_accuracy,
                                                train_loss))
            ta += train_accuracy

file.close()
