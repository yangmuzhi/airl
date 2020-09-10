import tensorflow as tf
from .base import Base

class Discriminator(Base):
    def __init__(self, save_path, obs_dims, n_actions):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        self.n_actions = n_actions
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, obs_dims])
            action = tf.placeholder(dtype=tf.float32, shape=[None])
            self.action = tf.one_hot(self.action, depth=n_actions)
            self.state_action = tf.concat([self.state, self.action], axis=1)

            self.label = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='label')

        with tf.variable_scope('network') as network_scope:
            self.prob = self.construct_network(input=self.state_action)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.losses.log_loss(self.label, self.prob))

        # rewards
        self.rewards = tf.log(self.prob) - tf.log(1-self.prob) 

        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)
        super().__init__(save_path=save_path, rnd=1234)

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        state = np.concatenate([expert_s, agent_s])
        action = np.concatenate([expert_a, agent_a])
        label = np.concatenate([np.ones([expert_a.shape[0], 1]), np.zeros([agent_a.shape[0], 1])])
        return self.sess.run([self.train_op, self.loss], feed_dict={self.state: state, 
                                                         self.action: action, self.label: label})


    def get_rewards(self, agent_s, agent_a):
        agent_a = to_categorical(agent_a, self.n_actions)
        return self.sess.run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
