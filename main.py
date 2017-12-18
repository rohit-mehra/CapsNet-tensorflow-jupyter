"""Simplest Capsule Implementation Based on my understanding of the Paper: https://arxiv.org/abs/1710.09829
Referred to clear Doubts: https://github.com/naturomics/CapsNet-Tensorflow
Tried to comment as much as possible, let me know if you find any error.

Test Acc = 99.37%
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Optional
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.93
np.random.seed(99)
tf.set_random_seed(99)

# Data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True, reshape=False)

# Train Params
train_round = 0
ckpt = './checkpoints/capsnet.ckpt'  # Checkpoint file
resume = True  # True if you have the above file
do_test_first = True  # do test eval
save_after = 430  # and validate # checkpointing global step

lr = 1e-3  # default Adam

# For separate margin loss
m_plus = 0.9
m_minus = 0.1
lambda_val = 0.5  # down weight of the loss for absent digit classes # <------------
epsilon = 1e-9

# reconstruction loss weight
recon_loss_w = 0.0005  # <------------

batch_size = 128
epochs = 300
routing_iter = 3  # number of iterations in routing algorithm


# 1. Squashing
def squash(in_tensor, axis=-1, name="squash"):
    """Squashing Function
    Args:
        (S) in_tensor: Eg. A 5-D input tensor of a capsule layer, shape [N, 1, J, next_D, 1],
    Returns:
        (V) out_tensor: A 5-D tensor with the same shape as input tensor of capsule layer but squashed along axis.
    """
    with tf.name_scope(name):
        norm_square = tf.reduce_sum(tf.square(in_tensor), axis=axis, keep_dims=True,
                                    name='norm_square')  # [N, 1, J, 1, 1] if axis = -2
        squash_factor = norm_square / (1 + norm_square)

        return squash_factor * (in_tensor / tf.sqrt(norm_square + epsilon))


# 2. Agreement
def prediction_agreement(current_active_out, prediction, name):
    with tf.name_scope(name, default_name="agree"):
        # Tile current_active_out to get I=1152
        V_J = tf.tile(current_active_out, multiples=[
                      1, 1152, 1, 1, 1])  # [N, 1152, 10, 16, 1]

        # [N, 1152, 10, 1, 16] x [N, 1152, 10, 16, 1] = [N, 1152, 10, 1, 1]
        agreement = tf.matmul(prediction, V_J, transpose_a=True)
        return agreement


# 3. Routing Algorithm
def routing(prediction, num_iters, prior_IJ):

    with tf.variable_scope('routing'):
        for r in range(int(num_iters)):
            with tf.variable_scope('routing_iter_' + str(r)):
                # Step 4:
                # [N, 1152, 10, 1, 1]
                c_IJ = tf.nn.softmax(prior_IJ, dim=2, name='C_' + str(r))
                # Step 5:
                weighted_unactive_out = tf.reduce_sum((c_IJ * prediction),
                                                      axis=1,
                                                      keep_dims=True, name='VP_' + str(r))  # [N, 1, 10, 16, 1]
                # Step 6:
                current_active_out = squash(
                    weighted_unactive_out, axis=-2, name='squash_' + str(r))  # [N, 1, 10, 16, 1]
                # Step 7:
                prior_IJ += prediction_agreement(current_active_out,
                                                 prediction, name='agreement_' + str(r))

        return current_active_out  # [N, 1, 10, 16, 1]


# attribs of caps layer = W_IJ, prior_IJ, Conv2dunits
# We Know I how to Know J?
# Wij = [8 x 16]

# I see this as information in 4D encoded ---> distributed represenation in 3D,
# We can use it to flatten as well by setting next_capsules = 1, and next_D = vector dimension
# Naive Flattening won't be required.
def capsule_layer_prediction(in_tensor, num_iters=3, D=8, capsules=32, next_D=16, next_capsules=10, kernel_size=9,
                             strides=2, name='caps_conv2d'):

    batch_size = tf.shape(in_tensor)[0]
    with tf.variable_scope(name):
        # Optimized way from: https://github.com/naturomics/CapsNet-Tensorflow
        capsule = tf.layers.conv2d(in_tensor,
                                   capsules * D,
                                   kernel_size,
                                   strides,
                                   activation=tf.nn.relu)  # NHWC*D # [N, 6, 6, 32*8]
        # output of capsule units
        U_I = tf.reshape(capsule, shape=[batch_size, -1, D])
        U_I = squash(U_I, name='pc_out')  # squash 1152 capsules along axis=-1
        U_I = tf.expand_dims(U_I, -1)
        U_I = tf.expand_dims(U_I, 2)  # [N, 1152, 1, 8, 1]
        # [N, 1152, 10, 8, 1] # Technically this is the end of primary capsules
        U_I = tf.tile(U_I, multiples=[1, 1, 10, 1, 1])

        # capsule_shape[1] # 1152 = 32 * 6 * 6
        I = 1152  # <-------- can be calculated dynamically
        J = next_capsules  # 10

        # W_IJ, shared weights
        W_IJ = tf.Variable(tf.random_normal(
            [1, I, J, D, next_D], stddev=0.01))  # [1, 1152, 10, 8, 16]
        # [N, 1152, 10, 8, 16] <-- IJ[8, 16]
        W_IJ = tf.tile(W_IJ, multiples=[batch_size, 1, 1, 1, 1])

        # prediction vectors, prediction u_j = [16D]
        # [N, 1152, 10, 16, 8] x [N, 1152, 1, 8, 1] = [N, 1152, 10, 16, 1]
        prediction_vectors = tf.matmul(W_IJ, U_I, transpose_a=True)

    prior_IJ = tf.Variable(
        np.zeros([1, I, J, 1, 1]), trainable=False, dtype=np.float32)  # nijkl
    # [N, 1152, 10, 1, 1]
    prior_IJ = tf.tile(prior_IJ, multiples=[batch_size, 1, 1, 1, 1])

    activations = routing(prediction_vectors, num_iters, prior_IJ)

    with tf.control_dependencies([activations]):  # Sanity
        return tf.squeeze(activations)  # [N, 10, 16]


# Network Arch:
class CapsNet:
    def __init__(self, inp_dim=28, num_iters=3, pred_vec_len=16, lr=1e-3,
                 classes=10, m_plus=0.9, m_minus=0.1, lambda_val=0.5, scope="CapsNet"):

        print("Learning Rate={}, m_plus={}, m_minus={}, lambda_val={}".format(lr,
                                                                              m_plus,
                                                                              m_minus,
                                                                              lambda_val))

        with tf.variable_scope(scope):
            self.X = tf.placeholder(
                tf.float32, shape=[None, inp_dim, inp_dim, 1], name='inputs')
            self.Y = tf.placeholder(
                tf.float32, shape=[None, 10], name='one_hot_labels')

            self.is_training = tf.placeholder_with_default(
                True, shape=[], name="bool_for_masking")

            with tf.variable_scope('conv_layer'):
                self.conv1 = tf.layers.conv2d(self.X,
                                              filters=256,
                                              kernel_size=9,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.nn.relu,
                                              name='conv1')  # [N, 20, 20, 256]

            # Capsule Layer Out
            self.digit_caps_out = tf.squeeze(capsule_layer_prediction(in_tensor=self.conv1,
                                                            num_iters=num_iters,
                                                            next_D=pred_vec_len),  # [N, 1, 10, 16, 1]
                                   name='distributed_prediction')  # [N, 10, 16]

            # [N, 10, 16] --> [N, 10] length vectors for actual predictions
            self.pred_length = tf.sqrt(tf.reduce_sum(
                tf.square(self.digit_caps_out) + epsilon, axis=2, keep_dims=False))

            # [N, 10] --> # [N,]
            self.predictions = tf.squeeze(tf.argmax(self.pred_length,
                                                    axis=1, output_type=tf.int32, name='predictions'))
            self.one_hot_predictions = tf.one_hot(
                self.predictions, depth=classes, name='one_hot_predictions')

            # [N, 10] * [N, 10] = [N, 10]
            self.m_plus = self.Y * \
                tf.square(tf.maximum(0., m_plus - self.pred_length))  # [N, 10]
            self.m_minus = lambda_val * \
                (1 - self.Y) * tf.square(tf.maximum(0.,
                                                    self.pred_length - m_minus))  # [N, 10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(
                self.m_plus + self.m_minus, axis=-1), name='margin_loss')

            self.recon_target = tf.cond(self.is_training,
                                        lambda: self.Y,  # Train True use Labels
                                        lambda: self.one_hot_predictions,
                                        name='recon_target')

            with tf.variable_scope('masking'):
                self.masked_pred = tf.matmul(self.digit_caps_out, tf.reshape(self.recon_target, shape=[-1, 10, 1]),
                                             transpose_a=True, name='masked_pred')  # [N, 16, 10] x [N, 10, 1] = [N, 16, 1]
            with tf.variable_scope('decoder'):
                # [N, 16, 1] --> [N, 16]
                decoder_inp = tf.reshape(self.masked_pred,
                                         shape=[-1, pred_vec_len])
                self.fc1 = tf.layers.dense(decoder_inp,
                                           units=512,
                                           activation=tf.nn.relu,
                                           name='fc1')
                self.fc2 = tf.layers.dense(self.fc1,
                                           units=1024,
                                           activation=tf.nn.relu,
                                           name='fc2')

                self.pred_X = tf.layers.dense(self.fc2,
                                              units=inp_dim * inp_dim,
                                              activation=tf.nn.sigmoid,
                                              name='pred_X')  # [N, 28*28]

                self.pred_image = tf.reshape(
                    self.pred_X, shape=[-1, inp_dim, inp_dim, 1])

            # Backward
            self.reconstruction_loss = tf.reduce_sum(tf.square(tf.reshape(self.X,
                                                                          shape=[-1, inp_dim * inp_dim]) - self.pred_X),
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
            ys = np.zeros([10, 10], dtype=np.float32)  # dummy

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

caps_net = CapsNet(lr=lr)
print('Network Built..')

tvars = tf.trainable_variables()
saver = tf.train.Saver(var_list=tvars)

file = open('avg_log_{}.csv'.format(train_round), 'w')
file.write('step,avg_train_acc,step_val_acc\n')

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    if resume:
        saver.restore(sess, ckpt)
        print("RESUMED..")

    if do_test_first:
        print('TESTING..')

        ta = 0
        total_steps = len(mnist.test.images) // test_batch_size

        for step in range(1, total_steps + 1):
            val_x, val_y = mnist.test.next_batch(test_batch_size)
            test_batch_accuracy = caps_net.accuracy(val_x, val_y, sess)
            ta += test_batch_accuracy
            print('{},{:>3.3f}'.format(step, test_batch_accuracy))

        print('Test Accuracy: ', ta / total_steps)

    print('*' * 90)
    print('Training..learning_rate = {}, total_epochs = {}'.format(lr, epochs))
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
