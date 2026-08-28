import tensorflow as tf
from utils import weight_variable_glorot


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, name=''):
        self.name = name
        self.vars = {}
        self.issparse = True
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class GraphSAGE():
    """GraphSAGE layer with mean aggregator."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, name='', aggregator_type='mean'):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['self_weights'] = weight_variable_glorot(input_dim, output_dim, name='self_weights')
            self.vars['neigh_weights'] = weight_variable_glorot(input_dim, output_dim, name='neigh_weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.aggregator_type = aggregator_type

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            # Mean aggregation: A * x
            neigh = tf.sparse_tensor_dense_matmul(self.adj, x)
            # Self + neighbor transformation
            from_self = tf.matmul(x, self.vars['self_weights'])
            from_neigh = tf.matmul(neigh, self.vars['neigh_weights'])
            outputs = self.act(from_self + from_neigh)
        return outputs


class GraphAttention():
    """Graph Attention Network layer."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, name=''):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['a_self'] = weight_variable_glorot(output_dim, 1, name='a_self')
            self.vars['a_neigh'] = weight_variable_glorot(output_dim, 1, name='a_neigh')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])

            # Attention coefficients
            f_self = tf.matmul(x, self.vars['a_self'])  # (N, 1)
            f_neigh = tf.matmul(x, self.vars['a_neigh'])  # (N, 1)
            logits = f_self + tf.transpose(f_neigh)  # (N, N) broadcast

            # Mask with adjacency (dense version for attention)
            adj_dense = tf.sparse_tensor_to_dense(self.adj)
            neg_inf = -1e9 * tf.ones_like(adj_dense)
            masked_logits = tf.where(tf.not_equal(adj_dense, 0), logits, neg_inf)

            coefs = tf.nn.softmax(tf.nn.leaky_relu(masked_logits))
            coefs = tf.nn.dropout(coefs, 1 - self.dropout)

            outputs = self.act(tf.matmul(coefs, x))
        return outputs


class InnerProductDecoder():
    """Decoder using inner product for link prediction."""

    def __init__(self, input_dim, num_r, act=tf.nn.sigmoid, name=''):
        self.name = name
        self.act = act
        self.num_r = num_r

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            R = inputs[:self.num_r, :]  # drug embeddings
            D = inputs[self.num_r:, :]  # side-effect embeddings
            outputs = self.act(tf.matmul(R, tf.transpose(D)))
            outputs = tf.reshape(outputs, [-1])
        return outputs


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors."""
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)
