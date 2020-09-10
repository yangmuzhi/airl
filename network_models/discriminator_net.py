import tensorflow as tf



def d_net(inputs):
    x = tf.layers.dense(inputs=inputs, units=20, activation=tf.nn.leaky_relu, name='layer1')
    x = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
    x = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
    prob = tf.layers.dense(inputs=x, units=1, activation=tf.sigmoid, name='prob')
    return prob