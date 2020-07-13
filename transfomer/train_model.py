from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import data_util as du
import transformer_model as tm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问

# ---------------
# 超参数（hyperparameters）
# ---------------
num_layers = 1  # encoder和decoder的层数
d_model = 128  # 词向量的维度
dff = 128  # point_wise中第一层神经元数目
num_heads = 8  # 多头注意力的数目
input_vocab_size = 937  # 输入序列词汇数目
target_vocab_size = 983  # 输出序列词汇数目
dropout_rate = 0.1
EPOCHS = 50
batch_size = 128


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    WarmUp学习率
    '''

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    '''
    输入序列为0填充值的标签不添加到总的loss中
    :param real: 真实标签
    :param pred: 预测标签
    :return:
    '''
    # 交叉熵损失函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 若标签为0,mask=0；反之mask=1
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)

    # 若mask=0,则loss不加到最终优化的loss中
    loss_ *= mask  # 乘0

    return tf.reduce_mean(loss_)


def create_masks(inp, tar):
    '''

    :param inp: shape[]
    :param tar: shape[]
    :return:
    '''
    # print('inp.shape', inp.shape) [None,None]
    # print('tar.shape', tar.shape) [None,None]

    # 编码器填充遮挡
    enc_padding_mask = tm.create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = tm.create_padding_mask(inp)  # 与上面一个一样！！

    # 在解码器的第一个注意力模块使用。（Mask Multi-head Attention）
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = tm.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = tm.create_padding_mask(tar)

    # tf.maximum同位置那个数比较大就返回那个数【shape需要了解】
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def train(encoder_input, encoder_target):
    # Transformer模型
    transformer = tm.Transformer(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, target_vocab_size,
                                 pe_input=input_vocab_size,
                                 pe_target=target_vocab_size,
                                 rate=dropout_rate)

    # 设置学习率
    learning_rate = CustomSchedule(d_model)
    # WarmUp学习率+Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # 设置loss metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 设置模型是否存在
    checkpoint_path = "ckpt"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # 创建bathc数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((encoder_input, encoder_target)).shuffle(len(encoder_input))
    train_dataset = train_dataset.batch(128, drop_remainder=True)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> chinese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            # 获取decoder的输入和标签
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            # 创建mask
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = transformer(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving model for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def translate(sentence, input_word_index, target_index_word, plot_attention=False):
    '''
    将输入的英文字符串翻译为中文
    :param inp_sentence:
    :param input_word_index:
    :param target_index_word:
    :param plot_attention:
    :return:
    '''
    sentences_encoder = []

    # 进行标点符号过滤
    sentence = du.preprocess_sentence(sentence, False, False)

    print('English input:', sentence)

    # 进行编码
    for word in sentence.split(' '):
        sentences_encoder.append(input_word_index.get(word, 0))

    # 进行填充
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences([sentences_encoder],
                                                                  maxlen=5,
                                                                  padding="post",
                                                                  value=0)

    # WarmUp学习率+Adam
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # 加载模型
    transformer = tm.Transformer(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, target_vocab_size,
                                 pe_input=input_vocab_size,
                                 pe_target=target_vocab_size,
                                 rate=dropout_rate)
    # 设置模型是否存在
    checkpoint_path = "ckpt"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # 中文的开始标记。
    output = tf.expand_dims([target_word_index['<S>']], 0)

    dec_sentence = []

    for i in range(14):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # print(int(predicted_id[0][0]), target_index_word[int(predicted_id[0][0])])
        # dec_sentence += target_index_word[int(predicted_id[0][0])]
        dec_sentence.append(target_index_word[int(predicted_id[0][0])])
        # 如果 predicted_id 等于结束标记，就返回结果
        if target_index_word[int(predicted_id[0][0])] == "<E>":
            break

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return dec_sentence, attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    # sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 画出注意力权重
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels([i for i in sentence] + ['<E>'], fontdict=fontdict, rotation=90)

        ax.set_yticklabels([i for i in result], fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    encoder_input, encoder_target, input_word_index, input_index_word, target_word_index, target_index_word = du.get_encoder_data()

    # 训练模型
    train(encoder_input, encoder_target)

    sentence = 'I believe in you'
    dec_sentence, attention_weights = translate(sentence, input_word_index, target_index_word, plot_attention=False)
    print(dec_sentence)
    plot_attention_weights(attention_weights, sentence.split(' '), dec_sentence, 'decoder_layer1_block2')

    sentence = 'My cat looks sad'
    dec_sentence, attention_weights = translate(sentence, input_word_index, target_index_word, plot_attention=False)
    print(dec_sentence)
    plot_attention_weights(attention_weights, sentence.split(' '), dec_sentence, 'decoder_layer1_block2')

    sentence = 'She is a teacher'
    dec_sentence, attention_weights = translate(sentence, input_word_index, target_index_word, plot_attention=False)
    print(dec_sentence)
    plot_attention_weights(attention_weights, sentence.split(' '), dec_sentence, 'decoder_layer1_block2')
