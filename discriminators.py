import tensorflow as tf
import numpy as np



def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)

def conv_layer(x, filter_shape, stride):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=xavier_init(),
        trainable=True)
    return tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding='SAME')


def dilated_conv_layer(x, filter_shape, dilation):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=xavier_init(),
        trainable=True)
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')


def deconv_layer(x, filter_shape, output_shape, stride):
    filters = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=xavier_init(),
        trainable=True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1])


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.compat.v1.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    x = tf.constant(x)
    shapes = tf.shape(x)
    dim = tf.TensorShape(shapes).as_list()[-1]
    beta = tf.compat.v1.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.compat.v1.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.compat.v1.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.compat.v1.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)


def flatten_layer(x):
    x = tf.constant(x)
    input_shape = tf.shape(x)
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    x = tf.constant(x)
    shapes = tf.shape(x)
    in_dim = tf.TensorShape(shapes).as_list()[-1]
    W = tf.compat.v1.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

class Discriminator:
    def __init__(self, x, mask, local_x, global_completion, local_completion, is_training, batch_size):
        self.batch_size = batch_size
        self.imitation = self.generator(x * (1 - mask), is_training)  # generator will change to input
        self.completion = self.imitation * mask + x * (1 - mask)
        self.real = self.discriminator(x, local_x, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, reuse=True)
        self.g_loss = self.calc_g_loss(x, self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.g_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.d_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.compat.v1.variable_scope('global'):
                with tf.compat.v1.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.compat.v1.variable_scope('local'):
                with tf.compat.v1.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.compat.v1.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.compat.v1.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = full_connection_layer(output, 1)
               
        return output


    def calc_g_loss(self, x, completion):
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)


    def calc_d_loss(self, real, fake):
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha