"""
 Another transformer decoders using template as input. Attention is all you need.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.util import nest
from tensorflow.contrib import rnn
#from texar.core import layers

import math
import collections
import numpy as np

#from texar.modules.networks.networks import FeedForwardNetwork
#from texar.modules.decoders.transformer_decoders import TransformerDecoderOutput
#from texar.utils import beam_search
#from texar.modules.embedders import embedder_utils
#from texar.modules.embedders import position_embedders
#from texar.utils import utils
#from texar.utils.shapes import shape_list

class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",\
            ("output_logits", "sample_ids"))):
    """the output logits and sampled_ids"""
    pass

def shape_list(x):
    """Returns the tensor shape.

    Returns static shape when possible.

    Returns:

        - If the rank of :attr:`x` is unknown, returns the dynamic shape: \
        `tf.shape(x)`
        - Otherwise, returns a list of dims, each of which is either an `int` \
        whenever it can be statically determined, or a scalar Tensor.
    """
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.
    Args:
     length: a Scalar.
    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)

def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.
    This does not actually save any computation.
    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative values
            indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative values
            indicate unlimited.
    Returns:
        a `Tensor` with shape [1, 1, length, length].
        [batch_size, num_heads, queri_len, queri_len]
    """
    band = _ones_matrix_band_part(
            length,
            length,
            max_backward,
            max_forward,
            out_shape=[1, 1, length, length])
    return -1e18 * (1.0 - band)

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].
    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len, memory_length
    """
    ret = memory_padding * -1e18
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def multihead_attention(queries,
                        memory_attention_bias=None,
                        memory=None,
                        num_heads=8,
                        num_units=None,
                        dropout_rate=0,
                        cache=None,
                        scope='multihead_attention',
                        is_training = True):
    '''Applies multihead attention.
    Args:
      queries: A 3d tensor with shape of [batch, length_query, depth_query].
      keys: A 3d tensor with shape of [batch, length_key, depth_key].
      num_units: A scalar indicating the attention size,
        equals to depth_query if not given.
      dropout_rate: A floating point number.
      num_heads: An int. Number of heads with calculating attention.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      A 3d tensor with shape of (batch, length_query, num_units)
    '''
    #pylint: disable=too-many-locals
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number"
                             "of attention heads (%d)." % (\
                            num_units, num_heads))
        if memory is None:
            #'self attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
            V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
            #Q = queries
            #K = queries
            #V = queries
            if cache is not None:
                # 'decoder self attention when dynamic decoding'
                K = tf.concat([cache['self_keys'], K], axis=1)
                V = tf.concat([cache['self_values'], V], axis=1)
                cache['self_keys'] = K
                cache['self_values'] = V
        else:
            # 'encoder decoder attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            if cache is not None:
                K, V = tf.cond(
                    tf.equal(tf.shape(cache["memory_keys"])[1], 0),
                    true_fn=lambda: \
                        [tf.layers.dense(memory, num_units, \
                            use_bias=False, name='k'), \
                        tf.layers.dense(memory, num_units, \
                            use_bias=False, name='v')],
                    false_fn=lambda: \
                        [cache["memory_keys"], cache["memory_values"]])
            else:
                K, V = [tf.layers.dense(memory, num_units, \
                            use_bias=False, name='k'),
                        tf.layers.dense(memory, num_units, \
                            use_bias=False, name='v')]

        #Q = tf.layers.dropout(Q, rate=dropout_rate)
        #K = tf.layers.dropout(K, rate=dropout_rate)
        # = tf.layers.dropout(V, rate=dropout_rate)

        Q_ = _split_heads(Q, num_heads)
        K_ = _split_heads(K, num_heads)
        V_ = _split_heads(V, num_heads)
        #[batch_size, num_heads, seq_length, memory_depth]
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head**-0.5

        logits = tf.matmul(Q_, K_, transpose_b=True)
        if memory_attention_bias is not None:
            logits += memory_attention_bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        attn_map = weights
        #attn_map = tf.Variable(weights, name="attn_map")
        #print("weights:",weights)
        #add by qyh, save attention
        #np_weights = weights.eval()
        #np_input = queries.eval()
        #np.save("/home/qiyunhan/Text_Infilling-master/text_infilling/atten_visu/inputs.npy",np_input)
        #np.save("")
        weights = tf.layers.dropout(weights, rate=dropout_rate, training = is_training)
        outputs = tf.matmul(weights, V_)

        outputs = _combine_heads(outputs)
        outputs = tf.layers.dense(outputs, num_units,\
            use_bias=False, name='output_transform')
        #(batch_size, length_query, attention_depth)
    return outputs, attn_map

def layer_normalize(inputs,
                    epsilon=1e-8,
                    scope='ln',
                    reuse=None):
    '''Applies layer normalization. averaging over the last dimension
    Args:
        inputs: A tensor with 2 or more dimensions, where the first
            dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing
            ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
    Returns:
        A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        filters = inputs.get_shape()[-1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        
        scale = tf.get_variable('layer_norm_scale',\
            [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable('layer_norm_bias',\
            [filters], initializer=tf.zeros_initializer())
        
        norm_x = (inputs - mean) * tf.rsqrt(variance + epsilon)
        outputs = norm_x * scale + bias
        #outputs = norm_x
    return outputs


def _split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads, becomes dimension 1).
        must ensure x.shape[-1] can be deviced by num_heads.any
    """
    depth = x.get_shape()[-1]
    splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
        num_heads, depth // num_heads])
    return tf.transpose(splitted_x, [0, 2, 1, 3])

def _combine_heads(x):
    """
    input: [batch, num_heads, seq_len, dim]
    output:[batch, seq_len, num_heads*dim]
    """
    t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
    num_heads, dim = t.get_shape()[-2:]
    return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])



def _ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    """Matrix band part of ones."""
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = tf.constant(band, tf.float32)
    else:
        band = tf.matrix_band_part(tf.ones([rows, cols]),
                                   tf.cast(num_lower, tf.int64),
                                   tf.cast(num_upper, tf.int64))
        if out_shape:
            band = tf.reshape(band, out_shape)
    return band

'''
def get_initializer(hparams=None):
    """Returns an initializer instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None

    if is_str(hparams["type"]):
        kwargs = hparams["kwargs"]
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()
        modules = ["tensorflow.initializers", "tensorflow.keras.initializers",
                   "tensorflow", "texar.custom"]
        try:
            initializer = utils.get_instance(hparams["type"], kwargs, modules)
        except TypeError:
            modules += ['tensorflow.contrib.layers']
            initializer_fn = utils.get_function(hparams["type"], modules)
            initializer = initializer_fn(**kwargs)
    else:
        initializer = hparams["type"]
    return initializer
'''

class SinusoidsSegmentalPositionEmbedder(object):
    def __init__(self, hparams=None):
        self._hparams = hparams

    def default_hparams(self):
        """returns a dictionary of hyperparameters with default values
        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale. The number of different
        timescales is equal to channels/2.
        """
        hparams = {
            'name': 'sinusoid_segmental_posisiton_embedder',
            'min_timescale': 1.0,
            'max_timescale': 1.0e4,
            'trainable': False,
            'base': 256,
        }
        return hparams

    def _build(self, length, channels, segment_ids, offsets):
        """
        :param length: an int
        :param channels: an int
        :param segment_id: [batch_size, length]
        :param segment_offset: [batch_size, length]
        :return: [batch_size, length, channels]
        """
        # TODO(wanrong): check if segment_ids is of shape [batch_size, length]
        position = tf.to_float(tf.add(tf.multiply(tf.cast(1, tf.int64), segment_ids),
                                      offsets))
        num_timescales = channels // 2
        min_timescale = 1.0
        max_timescale = 1.0e4
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 2) * inv_timescales
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        signal = tf.reshape(signal, shape=[-1, length, channels])
        return signal

def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)

def conv(h, kernel,if_kernel):
    #h size = bs * 50 * dim, kernel = 3*3
    print("if_kernel",if_kernel)
    kernel_ = tf.nn.softmax(kernel, dim = -1)
    def conv_(h,kernel_):
        #kernel_combine = get_combine_kernel(kernel)#3*3*dim
        batch_size = h.shape[0]
        dim = h.shape[-1]
        length = h.shape[1]#50
        conv_all = np.array([])
        flag = 0
        h_expd = np.concatenate((h[:,0:1,:],h,h[:,-2:-1,:]),axis = 1)#bs*52*dim
        for i in range(length):
            t = h_expd[:,i:i+3,:]#bs*3*dim
            if if_kernel == 1:
                t = t[:,0,:] * kernel_[0][0] + t[:,1,:] * kernel_[0][1] + t[:,2,:] * kernel_[0][2]#bs*dim
            else:
                t = t[:,0,:] /3 + t[:,1,:] /3 + t[:,2,:] /3#bs*dim
            t = np.expand_dims(t,axis = 1)#bs*1*dim
            if flag == 0:
                conv_all = t
                flag = 1
            else:
                conv_all = np.concatenate((conv_all,t),axis = 1)
        return conv_all#bs*50*dim
    return tf.py_func(conv_, [h,kernel_], tf.float32)

def distance_scale(template, history):
    #template: bs*50*dim
    #history: bs*50*dim
    #cos distance
    batch_size = shape_list(template)[0]
    length = shape_list(template)[1]
    norm1 = tf.norm(template, axis = -1)#bs*50
    norm1 = tf.reshape(norm1, (batch_size,length,1))#bs*50*1
    norm2 = tf.norm(history, axis = -1)#bs*50
    norm2 = tf.reshape(norm2, (batch_size,length,1))#bs*50*1
    score = []
    for b in range(50):
        cos_dis = tf.matmul(template[:,b:b+1,:], history[:,b:b+1,:], transpose_b=True) #bs*1*1
        if b == 0:
            score = cos_dis
        else:
            score = tf.concat((score, cos_dis), axis = 1)
    #score bs*50*1
    score = score / norm1 / norm2#bs*50*1
    score = tf.reduce_mean(score,axis = 1)#bs*1
    return score

def attn_scale(template, history):
    #template: bs*50*dim
    #history: bs*50*dim
    #attention between day
    print("template:",template)
    history = tf.reduce_mean(history,axis = 1)#bs*dim
    template = tf.reduce_mean(template, axis = 1)#bs*dim
    print("template after:", template)
    batch_size = shape_list(template)[0]
    dim = shape_list(template)[1]
    with tf.variable_scope("attention_between_day"):
        weight_f = tf.Variable(tf.random_normal(shape=[dim,dim],mean=0,stddev=1), name = "feature")
        #history = tf.layers.dense(history, dim, use_bias=False, name='feature')#bs*dim
        history = tf.matmul(history, weight_f)
        template = tf.matmul(template, weight_f)
        #template = tf.layers.dense(template, dim, use_bias=False, name='feature')#bs*dim
        weight = tf.Variable(tf.random_normal(shape=[dim,dim],mean=0,stddev=1), name = "trans_attn")
        attn = tf.matmul(history, weight)#bs*dim
        attn = tf.matmul(attn, template, transpose_b=True)#bs*bs
        attn = tf.reshape(tf.diag_part(attn), (batch_size, 1))#bs*1
    return attn

def RNN_attn_scale(template, history, cell):
    #template: bs*50*dim
    #history: bs*50*dim
    #attention between day
    batch_size = shape_list(template)[0]
    length = shape_list(template)[1]
    dim = shape_list(template)[2]
    with tf.variable_scope("RNN", reuse = tf.AUTO_REUSE):
        history = tf.reshape(history, (batch_size, 50, dim))
        template = tf.reshape(template, (batch_size, 50, dim))
        history_o, history_s = tf.nn.dynamic_rnn(cell,history, dtype = tf.float32)#bs*50*dim, bs*dim
        template_o, template_s = tf.nn.dynamic_rnn(cell,template, dtype = tf.float32)#bs*50*dim, bs*dim
        weight_f = tf.get_variable(name = "feature", shape =[dim,dim],  initializer=tf.random_normal_initializer(mean=0, stddev=1) )
        #history = tf.layers.dense(history, dim, use_bias=False, name='feature')#bs*dim
        history_s = tf.matmul(history_s, weight_f)
        template_s = tf.matmul(template_s, weight_f)
        #template = tf.layers.dense(template, dim, use_bias=False, name='feature')#bs*dim
        weight = tf.get_variable(name = "trans_attn", shape =[dim,dim],  initializer=tf.random_normal_initializer(mean=0, stddev=1))
        attn = tf.matmul(history_s, weight)#bs*dim
        attn = tf.matmul(attn, template_s, transpose_b=True)#bs*bs
        attn = tf.reshape(tf.diag_part(attn), (batch_size, 1))#bs*1
        return attn


class Bi_LSTM(object):
    def __init__(self, hidden_dim):
        self._hidden_dim = hidden_dim
        with tf.variable_scope("bi-LSTM",reuse = tf.AUTO_REUSE):
            self.lstm_fw_cell = rnn.BasicLSTMCell(self._hidden_dim, forget_bias=1.0)
            self.lstm_bw_cell = rnn.BasicLSTMCell(self._hidden_dim, forget_bias=1.0)
        with tf.variable_scope("LSTM",reuse = tf.AUTO_REUSE):
            self.lstm_cell = rnn.BasicLSTMCell(self._hidden_dim, forget_bias=1.0)

    def _built(self, inputs):
        with tf.variable_scope("bi-LSTM", reuse = tf.AUTO_REUSE):
            outputs,states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell, inputs, dtype=tf.float32)
            outputs_fw = outputs[0]
            outputs_bw = outputs[1]
            out = tf.concat((outputs_fw, outputs_bw), axis = -1)#bs*98*2dim
            out = tf.layers.dense(out, self._hidden_dim, use_bias = False, name = "dim_transfer")#bs*98*dim
            return out
    def _built_RNN(self, inputs):
        with tf.variable_scope("LSTM", reuse = tf.AUTO_REUSE):
            out, _ = tf.nn.dynamic_rnn(self.lstm_cell, inputs, dtype=tf.float32)#input bs*98*dim, output bs*98*dim
            return out




class TemplateTransformerDecoder(object):
    """decoder for transformer: Attention is all you need
    """
    def __init__(self, embedding=None, vocab_size=None, hparams=None, dis_matrix = None, user_embedding = None, day_embedder = None):
        #ModuleBase.__init__(self, hparams)
        self._vocab_size = vocab_size
        self._embedding = embedding
        self._user_embedding = user_embedding
        self._day_embedding = day_embedder
        self._hparams = hparams
        self.sampling_method = self._hparams["sampling_method"]


        if self._vocab_size is None:
            self._vocab_size = self._embedding.get_shape().as_list()[0]

        self.output_layer = \
            self.build_output_layer(shape_list(self._embedding)[-1])
        #self.conv_kernel = tf.Variable(tf.random_normal(shape=[3,3,self._hparams["num_units"],self._hparams["num_units"]],mean=0,stddev=1))
        self.conv_kernel = tf.Variable(tf.random_normal(shape=[1,3],mean=0,stddev=3))
        #self.conv_kernel = self.get_conv_kernel(shape_list(self._embedding)[-1])
        #_, self.mask_train = self.get_mask(self._vocab_size, self._hparams["batch_size_train"])
        #self.mask_test,_ = self.get_mask(self._vocab_size, self._hparams["batch_size_test"])
        if self._hparams["position_embedder"]["name"] == 'sinusoids':
            self.position_embedder = SinusoidsSegmentalPositionEmbedder(self._hparams["position_embedder"]["hparams"])
        self.segment_id, self.offset_id = self.get_position_embedding()
        self.RNN_cell = tf.contrib.rnn.BasicRNNCell(num_units = shape_list(self._embedding)[-1])
        self.bi_lstm = Bi_LSTM(shape_list(self._embedding)[-1])
        self.pos_embed = tf.get_variable('pos_embeddings', [self._hparams["batch_size"], 50, shape_list(self._embedding)[-1]], \
                    initializer = tf.random_normal_initializer(mean=0, stddev=1), trainable = False) #trainable=False
    @staticmethod
    def default_hparams():
        """default hyperrams for transformer deocder.
            sampling_method: argmax or sample. To choose the function transforming the logits to the sampled id in the next position when inferencing.
        """
        return {
            'sampling_method': 'argmax',
            'initializer': None,
            'multiply_embedding_mode': 'sqrt_depth',
            'position_embedder': None,
            'share_embed_and_transform': True,
            'transform_with_bias': True,
            "use_embedding": True,
            "name":"decoder",
            "num_heads":8,
            "num_blocks":6,
            "zero_pad": False,
            "bos_pad": False,
            "max_seq_length":10,
            "maximum_decode_length":10,
            "beam_width":1,
            'alpha':0,
            "embedding_dropout":0.1,
            'attention_dropout':0.1,
            'residual_dropout':0.1,
            "sinusoid":True,
            'poswise_feedforward':None,
            'num_units':512,
            'eos_idx': 2,
            'bos_idx': 1,
        }

    def get_position_embedding(self):
        time_id = [x  + 1 for x in range(50)]#1~50
        print("time_id:", time_id)
        all_time = []
        for i in range(self._hparams["batch_size"]):
            all_time.append(time_id)
        all_time = np.array(all_time)#bs*50
        print("all time id shape:", all_time.shape)
        segment_id = tf.Variable(all_time, name = "segment_id", trainable = False)
        offset_id = tf.zeros((self._hparams["batch_size"], 50), dtype = tf.int64)
        return segment_id, offset_id



    def get_mask(self ,vocab_size, batch_size):
        def get_mask_(vocab_size, batch_size):
            mask_ = [0] * 8 + [1] * (vocab_size - 8)
            mask_1 = [1] * vocab_size
            mask_ = np.array(mask_).reshape((1,vocab_size)).astype(np.float32)
            mask_1 = np.array(mask_1).reshape((1,vocab_size)).astype(np.float32)
            mask = mask_
            mask_train = mask_
            mask_train = np.concatenate((mask_train,mask_1),axis = 0)
            m = []
            m_t = []
            for i in range(batch_size):
                m.append(mask)
                m_t.append(mask_train)
            m = np.array(m).astype(np.float32)
            m_t = np.array(m_t).astype(np.float32)
            return m, m
        return tf.py_func(get_mask_, [vocab_size, batch_size], [tf.float32,tf.float32])

    def prepare_tokens_to_embeds(self, tokens):
        """ a callable function to transform tokens into embeddings."""
        token_emb = tf.nn.embedding_lookup(self._embedding, tokens)
        return token_emb

    def FeedForward(self,template_input,fb_drop,res_drop,scope, is_training):
        with tf.variable_scope(scope):
            dense1 = tf.layers.dense(inputs = layer_normalize(template_input),
                units = self._hparams["hidden_dim"]*4, activation = tf.nn.relu, use_bias = True)
            drop1 = tf.layers.dropout(dense1, rate=fb_drop, training = is_training)
            sub_output = tf.layers.dense(inputs = drop1,
                units = self._hparams["hidden_dim"], use_bias = True)
            sub_output = tf.layers.dropout(sub_output, rate=res_drop, training = is_training)
        return sub_output

    def get_mask_info(self,template, mask_id_, xing_id):
        def get_mask_info_(template, mask_id_, xing_id):
            #print("template shape:", template.shape)
            batch_size = template.shape[0]
            length = template.shape[1]
            template = list(template)
            blank = []
            b_id = []
            a_id = []
            b_delt = []
            a_delt = []
            for i, temp in enumerate(template):
                #print("temp:",temp)
                b = []
                t =  list(temp)
                for ind, t in enumerate(t):
                    if t == mask_id_:
                        b.append(ind)
                blank.append(b)
            blank = np.array(blank).astype(np.int32)
            return blank#bs*12
        return tf.py_func(get_mask_info_, [template, mask_id_, xing_id], [tf.int32])


    def similarity_his_with_current(self, template_word_embeds, history_word_embeds):
        #template: if_self_position:bs*49*dim, else: bs*50*dim
        #history_word_embeds: bs*num*50*dim
        batch_size = shape_list(template_word_embeds)[0]
        #boa_ = tf.fill([batch_size,1], 1)#bs*1
        #boa_ = tf.nn.embedding_lookup(self._embedding, boa_)#bs*1*dim
        #template_word_embeds = tf.concat((boa_, template_word_embeds), axis = 1)#bs*50*dim
        #if shape_list(template_word_embeds)[1] != 50:
        #    print("dim error")
        alpha = []
        for d in range(self._hparams["history_num"]):
            history = history_word_embeds[:,d,:,:]#bs*50*dim
            #there can be other way
            if self._hparams["attn_way"] == "cos_distance":
                dis = distance_scale(template_word_embeds, history)#bs*1
            elif self._hparams["attn_way"] == "attn":
                dis = attn_scale(template_word_embeds, history)#bs*1
            elif self._hparams["attn_way"] == "RNN_attn":
                dis = RNN_attn_scale(template_word_embeds, history, self.RNN_cell)#bs*1
            if d == 0:
                alpha = dis#bs*1
            else:
                alpha = tf.concat((alpha, dis),axis = -1)
        alpha = tf.nn.softmax(alpha, axis = -1)#bs*3
        his = []
        for d in range(self._hparams["history_num"]):
            a = tf.reshape(alpha[:,d], (batch_size, 1, 1))#bs*1*1
            his_ = history_word_embeds[:,d,:,:] * a#bs*50*dim
            if d == 0:
                his = his_
            else:
                his += his_
        return alpha, his#bs*3, bs*50*dim

    def get_blank_embeds(self, pos_embeds, blank):
        #blank: bs*12
        #pos_embeds: bs*98*dim
        b_pos_embed = []
        for i in range(self._hparams["batch_size"]):
            b_p_e = []
            for j in range(self._hparams["blank"]):
                b = blank[i][j]
                if j == 0:
                    b_p_e = pos_embeds[i:i+1,b:b+1,:]#1*1*dim
                else:
                    t = pos_embeds[i:i+1,b:b+1,:]#1*1*dim
                    b_p_e = tf.concat((b_p_e,t), axis = 1)
            if i == 0:
                b_pos_embed = b_p_e
            else:
                b_pos_embed = tf.concat((b_pos_embed, b_p_e), axis = 0)
        return b_pos_embed#bs*12*dim

    def get_history_template(self, history_word_embeds):
        #history_word_embeds = bs*3*50*dim
        his = []
        for i in range(self._hparams["history_num"]):
            if i == 0:
                his = history_word_embeds[:,i,:,:]
            else:
                his = tf.concat((his, history_word_embeds[:,i,:,:]), axis = 1)
        return his#bs*(50*num)*dim

    def get_user_embedding(self, user_id, length):
        #user_id: bs*1
        user_all_id = []
        for i in range(length):
            if i == 0:
                user_all_id = user_id
            else:
                user_all_id = tf.concat((user_all_id, user_id), axis = 1)
        user_embedding = tf.nn.embedding_lookup(self._user_embedding, user_all_id)#bs*length*dim
        return user_embedding

    def get_day_embedding(self, day_id):
        #user_id: bs*1
        day_all_id = []
        for i in range(50):
            if i == 0:
                day_all_id = day_id
            else:
                day_all_id = tf.concat((day_all_id, day_id), axis = 1)
        day_embedding = tf.nn.embedding_lookup(self._day_embedding, day_all_id)#bs*50*dim
        return day_embedding

    def update_template(self, template_input, x, blank):
        #template_input: bs*50*dim
        #x: bs*blank_num*dim
        #blank: bs*blank_num
        with tf.name_scope("de_en_interaction") as scope:
            batch_size = self._hparams["batch_size"]
            blank_num = self._hparams["blank"]
            temp = []
            for i in range(batch_size):
                t = []
                for j in range(blank_num-1):
                    ind = blank[i,j]
                    ind_next = blank[i,j+1]
                    if j == 0:
                        t = tf.concat((template_input[i:i+1,0:ind,:], x[i:i+1,j:j+1,:], template_input[i:i+1,ind+1:ind_next,:]), axis = 1)
                    else:
                        t = tf.concat((t, x[i:i+1,j:j+1,:], template_input[i:i+1,ind+1:ind_next,:]), axis = 1)
                t = tf.concat((t, x[i:i+1,blank_num-2:blank_num-1,:], template_input[i:i+1,blank[i,-1]+1:,:]), axis = 1)#1*50*dim
                if i == 0:
                    temp = t
                else:
                    temp = tf.concat((temp, t), axis = 0)
            return temp


    def _build(self, template_input_pack, encoder_decoder_attention_bias, args, mask_id,xing_id, is_training, user_id, day_id):
        """
            this function is called on training generally.
            Args:
                targets: [bath_size, target_length], generally begins with [bos] token
                template_input: [batch_size, source_length, channels]
                segment_ids: [batch_size, source_length], which segment this word belongs to
                user_id = bs*1
                day_id: bs*1
            outputs:
                logits: [batch_size, target_length, vocab_size]
                preds: [batch_size, target_length]
        """
        with tf.name_scope("train_decoder") as scope:

            template = template_input_pack['templates']#bs*50
            template_length = shape_list(template)[1]#50
            batch_size = shape_list(template)[0]
            #get blank info
            blank = self.get_mask_info(template, mask_id, xing_id)
            blank = tf.reshape(blank, (batch_size, self._hparams["blank"]))#bs*blank_num

            template_word_embeds = tf.nn.embedding_lookup(self._embedding, template)#bs*50*dim
            channels = shape_list(template_word_embeds)[-1]
            #bs*98*dim
            template_pos_embeds = self.position_embedder._build(template_length, channels, self.segment_id, self.offset_id)
            #template_pos_embeds = self.pos_embed
            #prepare user embedding
            #user_embeds = self.get_user_embedding(user_id, 50)#bs*50*dim
            #day_embeds = self.get_day_embedding(day_id)#bs*50*dim
            #template_pos_embeds = template_pos_embeds + user_embeds
            #template_pos_embeds = template_pos_embeds + user_embeds + day_embeds

            if self._hparams["if_history"] == 1:
                #use history
                #history = batch_size*hisnum*50
                history = template_input_pack["history"]
                #print("template:", history)
                history_word_embeds = tf.nn.embedding_lookup(self._embedding, history)#history = batch_size*hisnum*50*dim

                history_inputs = history_word_embeds[:,1,:,:] +  template_pos_embeds  #second day
                #history_inputs = template_pos_embeds
                history_inputs = tf.reshape(history_inputs, (batch_size, 50, channels))

            else:
                history_inputs = None
            template_inputs = template_word_embeds + template_pos_embeds
            #template_inputs = template_pos_embeds
            blank_pos_embeds = self.get_blank_embeds(template_pos_embeds, blank)
            inputs = blank_pos_embeds#no boa
            #print("history inputs:", history_inputs)

            decoder_output, attn_map, his_attn_map = self._self_attention_stack(
                inputs,
                template_inputs,
                history_inputs,
                #decoder_self_attention_bias=decoder_self_attention_bias,
                attn_drop = self._hparams["attention_dropout"],
                fb_drop = self._hparams["fb_dropout"],
                res_drop = self._hparams["residual_dropout"],
                embed_drop = self._hparams["embedding_dropout"],
                is_training = is_training,
                blank = blank,
            )

            #add user info at output
            #self.decoder_output = self.decoder_output * user_embeds#bs*blank_num*dim
            #user_embeds = self.get_user_embedding(user_id, self._hparams["blank"])#bs*blank_num*dim
            #decoder_output = tf.concat((decoder_output, user_embeds), axis = -1)#bs*blank_num*2dim
            #decoder_output = tf.layers.dense(decoder_output, channels, use_bias=False, name='user_output')#bs*blank_num*dim

            #output layer
            logits = self.output_layer(decoder_output)
            if self._hparams["if_mask"] == 1:
                _, self.mask_train = self.get_mask(self._vocab_size, shape_list(template)[0])
                logits = logits * self.mask_train
            logits_prob = tf.nn.softmax(logits, axis = -1)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            preds = tf.reshape(preds, (batch_size, self._hparams["blank"], 1))#bs*num*1
            preds = tf.transpose(preds, (1,0,2))#num*bs*1

        return logits, preds, logits_prob, attn_map, his_attn_map

    def _self_attention_stack(self,
                              inputs,
                              template_input,
                              history_inputs,
                              decoder_self_attention_bias=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              attn_drop = 0,
                              res_drop = 0,
                              fb_drop = 0,
                              embed_drop = 0,
                              is_training = True,
                              blank = None):
        """
            stacked multihead attention module.
        """
        inputs = tf.layers.dropout(inputs, rate=self._hparams["embedding_dropout"], training = is_training)

        x = inputs
        his = []

        if self._hparams["structure"] == "equal_connect": #every encoder to deocder
            for i in range(self._hparams["num_blocks"]):
                #print("layer:",i)
                layer_name = 'layer_{}'.format(i)
                #print(layer_name,i,self._hparams["num_blocks"])
                layer_cache = cache[layer_name] if cache is not None else None
                with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
                        
                    #Multihead Attention for encoder
                    with tf.variable_scope("self_attention_encoder"):
                        selfatt_enc_output, attn_enc_map = multihead_attention(
                            queries=layer_normalize(template_input),
                            memory=None,
                            memory_attention_bias=None,
                            num_units=self._hparams["num_units"],
                            num_heads=self._hparams["num_heads"],
                            dropout_rate=attn_drop,
                            cache=layer_cache,
                            scope="multihead_attention",
                            is_training = is_training
                        )
                        template_input = template_input + tf.layers.dropout(selfatt_enc_output,rate=res_drop, training = is_training)
                    if self._hparams["if_history"] == 1:

                        with tf.variable_scope("self_attention_history"):
                            selfatt_history, attn_enc_map = multihead_attention(
                                queries=layer_normalize(history_inputs),
                                memory=None,
                                memory_attention_bias=None,
                                num_units=self._hparams["num_units"],
                                num_heads=self._hparams["num_heads"],
                                dropout_rate=attn_drop,
                                cache=layer_cache,
                                scope="multihead_attention",
                                is_training = is_training
                            )
                            history_inputs = history_inputs + tf.layers.dropout(selfatt_history,rate=res_drop, training = is_training)

                        with tf.variable_scope('history_attention'):
                            his_output, his_attn_map = multihead_attention(
                                queries=layer_normalize(template_input),
                                memory=history_inputs,
                                memory_attention_bias=encoder_decoder_attention_bias,
                                num_units=self._hparams["num_units"],
                                num_heads=self._hparams["num_heads"],
                                dropout_rate=attn_drop,
                                scope="multihead_attention",
                                is_training = is_training
                            )
                            if i == 0:
                                his = his_attn_map#bs*head*50*50
                            else:
                                his = tf.concat((his, his_attn_map), axis = 2)

                            template_input = template_input + tf.layers.dropout(his_output, rate=res_drop, training = is_training)
                            #template_input = his_output

                    #FeedForward Netword for encoder
                    template_input_FF = self.FeedForward(template_input,fb_drop,res_drop,"encoder_FF", is_training)

                    template_input = template_input + tf.layers.dropout(template_input_FF,rate=res_drop, training = is_training)

                    if 1:
                        #x shape: bs*blank_num*dim
                        with tf.variable_scope('encdec_attention'):
                            encdec_output, attn_map = multihead_attention(
                                queries=layer_normalize(x),
                                memory=template_input,
                                memory_attention_bias=encoder_decoder_attention_bias,
                                num_units=self._hparams["num_units"],
                                num_heads=self._hparams["num_heads"],
                                dropout_rate=attn_drop,
                                scope="multihead_attention",
                                is_training = is_training
                            )
                            x = x + tf.layers.dropout(encdec_output, rate=res_drop, training = is_training)
                            #x = encdec_output
                        '''
                        with tf.variable_scope("self_attention_decoder"):
                            selfatt_output, attn_map = multihead_attention(
                                queries=layer_normalize(x),
                                memory=None,
                                memory_attention_bias=decoder_self_attention_bias,
                                num_units=self._hparams["num_units"],
                                num_heads=self._hparams["num_heads"],
                                dropout_rate=attn_drop,
                                cache=layer_cache,
                                scope="multihead_attention",
                                is_training = is_training
                            )
                            x = x + tf.layers.dropout(selfatt_output,rate=res_drop,  training = is_training)
                        '''
                        #FeedForward Network
                        x_FF = self.FeedForward(x,fb_drop,res_drop,"decoder_FF", is_training)
                        x = x + tf.layers.dropout(x_FF,rate=res_drop, training = is_training)
                    #trans decoder output to encoder
                    #template_input = bs*50*dim
                    #x: bs*blank_num*dim
                    #blank: bs*blank_num
                    #template_input = self.update_template(template_input, x, blank)#bs*50*dim
                    #template_input = tf.reshape(template_input, (self._hparams["batch_size"], 50, self._hparams["num_units"]))#bs*50*dim

        return layer_normalize(x), attn_map, his# his: bs*head*(50*layer)*50

    def build_output_layer(self, num_units):
        if self._hparams["share_embed_and_transform"]:
            if self._hparams["transform_with_bias"]:
                with tf.variable_scope(self.variable_scope):
                    affine_bias = tf.get_variable('affine_bias',
                        [self._vocab_size])
            else:
                affine_bias = None
            def outputs_to_logits(outputs):
                shape = shape_list(outputs)
                outputs = tf.reshape(outputs, [-1, num_units])
                logits = tf.matmul(outputs, self._embedding, transpose_b=True)
                if affine_bias is not None:
                    logits += affine_bias
                logits = tf.reshape(logits, shape[:-1] + [self._vocab_size])
                return logits
            return outputs_to_logits
        else:
            layer = tf.layers.Dense(self._vocab_size, \
                use_bias=self._hparams["transform_with_bias"])
            layer.build([None, num_units])
            return layer

    
    @property
    def output_size(self):
        """
        The output of the _build function, (logits, preds)
        logits: [batch_size, length, vocab_size]
        preds: [batch_size, length]
        """
        return TransformerDecoderOutput(
            output_logits=tensor_shape.TensorShape([None, None, self._vocab_size]),
            sample_id=tensor_shape.TensorShape([None, None])
            )

    def output_dtype(self):
        """
        The output dtype of the _build function, (float32, int32)
        """
        return TransformerDecoderOutput(
            output_logits=dtypes.float32, sample_id=dtypes.int32)

