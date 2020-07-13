from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import data_util as du


def get_angles(pos, i, d_model):
    '''
    计算位置编码中，sin或cos括号中的值
    :param pos: [[1],[2],[3],....[position]] shape:[pos,1]
    :param i: 具体的维度 [[1,2,3,...,d_model]] shape:[1,d_model]
    :param d_model: 位置编码的总维度
    :return:
    '''

    # sin和cos括号里面的定义 shape:[1,d_model]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    # 矩阵相乘 shape:[pos,1]*[1,d_model]=[pos,d_model]
    return pos * angle_rates


def positional_encoding(position, d_model):
    '''
    获取位置编码的矩阵 shape=[position, d_model]
    :param position:输入句子的最大长度,type=int
    :param d_model: 位置编码的维度,type=int
    :return:

    e.g:
    pos_encoding = positional_encoding(50, 512)
    pos_encoding shape:[1,50,512]
    '''

    # np.arange(position)[:, np.newaxis]->shape:[position,1] e.g.[[1],[2],[3],....[position]]
    # np.arange(d_model)[np.newaxis, :]->shape:[1,d_model] e.g.[[1,2,3,...,d_model]]
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # 将sin 应用于数组中的偶数索引(2i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将cos 应用于数组中的奇数索引(2i+1)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # 扩充一个维度pos_encoding shape:[1,position,d_model]
    pos_encoding = angle_rads[np.newaxis, ...]

    # 类型转换
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    '''
    对输入的序列生成mask,原始序列中为0的数值，在mask矩阵中为1；不被mask的数值在mask句子中为0；
    用于对encoder的输入进行mask
    :param seq: 输入序列
    :return: 与输入的shape相同

    e.g:
    seq=[[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]

    return
    [[[[0., 0., 1., 1., 0.]]],
     [[[0., 0., 0., 1., 1.]]],
     [[[1., 1., 1., 0., 0.]]]]
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # (batch_size, 1, 1, seq_len) 【扩充维度的目的？？】
    return seq[:, tf.newaxis, tf.newaxis, :]  #


def create_look_ahead_mask(size):
    '''
    返回一个[size,size]的上三角为1，下三角和主对角线为0的矩阵
    用于在decoder阶段时的mask attention
    :param size: type=int
    :return:

    e.g: create_look_ahead_mask(3)
    [[0., 1., 1.],
     [0., 0., 1.],
     [0., 0., 0.]]
    '''
    # 创建（size,size）值为1矩阵；下三角元素全部保留，包括对角线；上三角设置为0
    # tf.linalg.band_part 是否要保留对角元素以下几行或以上几行的元素；
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # shape:[size,size]
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
  q, k, v 必须具有匹配的前置维度。
  q, k 必须有匹配的倒数第二个维度;dq=dk
  虽然 mask 根据其类型（填充或前瞻）有不同的形状，但是 mask 必须能进行广播转换以便求和。

  参数:
    q: 请求的形状 == (..., n, dq) #dq是向量的维度，n是输入序列的长度
    k: 主键的形状 == (..., n, dk)
    v: 数值的形状 == (..., n, dv)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。

  返回值:
    输出，注意力权重
  """

    # print('signal attention mask shape', mask.shape)

    # 所有的q和k进行矩阵相乘 (..., n, n)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  #

    # 缩放 matmul_qk ，获取k的维度，再除于k
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 【但是前提是mask的shape必须与n*n一致！】
    if mask is not None:
        # 被mask的数值为1，不mask的值为0；因此mask的位置会加上一个无穷大的负数
        # 在进行softmax的时候，这些mask为位置的注意力权重则为0

        # print('scaled_attention_logits.shape',scaled_attention_logits.shape) (128, 8, 5, 5)
        # mask shape (128, 1, 1, 5)
        scaled_attention_logits += (mask * -1e9)

    # 对每一列进行softmax操作 (..., n, n)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 将每一列的注意力与v进行相乘(..., n, d_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 自注意力头数目
        self.d_model = d_model  # k,q,v向量的维度

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # 多头注意力的向量通过对总向量进行分割得到

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        分拆最后一个维度到 (num_heads, depth).转置结果使得形状为 (batch_size, num_heads, seq_len, depth)

        将单头注意力的d_model分为num_heads份
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # num_heads*depth=d_model

        # (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 方法里面的q,k,v都是上一层的输入
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 将原始的单个头的长度，拆分为num_head个头；也可以采用另外的做法！（可能采用for循环了！）
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # 最终输出的长度与原始的输入序列的长度是一致的
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    '''
    点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数
    :param d_model: 第二层神经元数目，为了使得能够搭建多个encoder_layer，因此第二层的unit必须是d_model
    :param dff: 第一层的神经元数目
    :return:
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 这里不涉及到token embedding与位置编码的相加操作!!!
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # 需要数以x的维度

        # 不涉及到位置编码
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # 第一层使用的是look_ahead_mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # 第二层使用的是与输入一样的padding【主要是对输入进行mask?】
        # 【而且除了租后一个向量维度外，其余的维度out1需要一致；这不是输入序列长度要一致了嘛？？？】
        # 输出维度和头数目是一样的！！只要输出的头数目一样就可以！
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        # encoder的层数
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    # 不知道训练的时候，到底输入的是什么
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # 对输入的词向量乘上sqrt(d_model)?
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # 与位置向量进行相乘
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # 在decoder阶段，原来都是在对最终的encoder_output进行注意力对齐
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    '''
    Transformer 包括编码器，解码器和最后与词汇表大小相同的线性层；解码器的输出是线性层的输入，返回线性层的输出。
    '''

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
