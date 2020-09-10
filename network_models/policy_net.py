import tensorflow as tf


class Policy_net:
    def __init__(self, name, ob_space, act_space):
        """
        :param name: string
        :param env: gym env
        """

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=(None, ob_space), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

