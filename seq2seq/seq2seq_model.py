# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import data_util as du
import time

UNITS = 30  # 隐藏层神经元数目
EMBEDDING_DIM = 50  # 词向量长度
INPUT_VOCAB_SIZE = 784  # 输入词汇表大小
TARGET_VOCAB_SIZE = 814  # 输出词汇表大小
INPUT_MAX_LEN = 4  # 输入序列的最大长度
TRAGET_MAX_LEN = 12  # 目标序列的最大长度
EPOCHS = 20  # 训练的轮次
BATCH_SIZE = 250
ENCODER_PATH = 'model/encoder.h5'  # 训练好的encoder路径
DECODER_PATH = 'model/decoder.h5'


def lstm(units):
    return tf.keras.layers.LSTM(units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')


def get_encoder(vocab_size, embedding_dim, encoder_unit):
    enc_input = tf.keras.layers.Input((INPUT_MAX_LEN,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(enc_input)

    # x:[batch_sz, input_max_length, units] h_c:[batch_sz, units] h_t:[batch_sz, units]
    x, h_c, h_t = lstm(encoder_unit)(x)

    return tf.keras.models.Model(inputs=enc_input, outputs=[x, h_c, h_t], name='encoder')


def get_decoder(vocab_size, embedding_dim, decoder_unit):
    # encoder的输出
    enc_h_c = tf.keras.layers.Input((UNITS,))
    enc_h_t = tf.keras.layers.Input((UNITS,))

    dec_input = tf.keras.layers.Input((1,) )
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(dec_input)
    x, dec_h_c, dec_h_t = lstm(decoder_unit)(x, initial_state=[enc_h_c, enc_h_t])
    dec_outputs = tf.keras.layers.Dense(TARGET_VOCAB_SIZE)(x)
    dec_outputs = tf.squeeze(dec_outputs, axis=1)
    return tf.keras.models.Model(inputs=[dec_input, enc_h_c, enc_h_t], outputs=[dec_outputs, dec_h_c, dec_h_t],
                                 name='decoder')


def loss_function(real, pred):
    # 标签为填充的，则mask=0;反之为mask=1
    mask = 1 - np.equal(real, 0)
    # 如果标签为0，则loss不进行优化
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

    return tf.reduce_mean(loss_)


def train_model(encoder_input, encoder_target, model_exist):
    encoder = get_encoder(INPUT_VOCAB_SIZE, EMBEDDING_DIM, UNITS)
    decoder = get_decoder(TARGET_VOCAB_SIZE, EMBEDDING_DIM, UNITS)

    encoder.summary()
    decoder.summary()

    optimizer = tf.optimizers.RMSprop(learning_rate=0.001)

    # 加载现存的模型
    if model_exist:
        encoder.load_weights(ENCODER_PATH)
        decoder.load_weights(DECODER_PATH)

    # 创建bathc数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((encoder_input, encoder_target)).shuffle(len(encoder_input))
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    N_BATCH = 1
    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for (batch, (inp, tar)) in enumerate(train_dataset):
            loss = 0

            with tf.GradientTape() as tape:
                # 计算输入
                # print('inp.shape', inp.shape)
                _, encoder_h_c, encoder_h_t = encoder(inp)

                # 创建起始符号<S>作为decode的第一个输入
                dec_input = tf.expand_dims([target_word_index['<S>']] * BATCH_SIZE, 1)

                # 每次为decoder添加一个词
                for t in range(1, tar.shape[1]):
                    # 将词添加到decoder
                    prediction, encoder_h_c, encoder_h_t = decoder([dec_input, encoder_h_c, encoder_h_t])

                    # 将上一刻的标签，作为下一刻的输入
                    dec_input = tf.expand_dims(tar[:, t], 1)
                    # print('prediction.shape',prediction.shape)
                    # print('tar[:, t]', tar[:, t].shape)
                    # 计算loss
                    loss += loss_function(tar[:, t], prediction)

            # 获取平均batch loss
            batch_loss = loss / int(tar.shape[1])
            total_loss += batch_loss

            # 计算loss对encoder和decoder的梯度
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)

            # 参数更新
            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 10 == 0:
                print('Epoch:{0}, Batch:{1}, Batch loss:{2:.4f}'.format(epoch, batch, batch_loss))

            N_BATCH = batch

        print('Epoch{0}, Train loss:{1:.4f}'.format(epoch, total_loss / N_BATCH))
        print('Training time of epoch {0}:{1:.4f} sec\n'.format(epoch, time.time() - start))

    encoder.save(ENCODER_PATH)
    decoder.save(DECODER_PATH)


def translate(sentence, input_word_index, target_index_word):
    '''
     将输入的英文字符串翻译为中文
    :param sentence: 英文字符串
    :param input_word_index: input_word_index字典
    :param target_index_word: target_index_word字典
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
    sentences_encoder = tf.keras.preprocessing.sequence.pad_sequences([sentences_encoder],
                                                                      maxlen=INPUT_MAX_LEN,
                                                                      padding="post",
                                                                      value=0)

    # 加载模型
    encoder = get_encoder(INPUT_VOCAB_SIZE, EMBEDDING_DIM, UNITS)
    decoder = get_decoder(TARGET_VOCAB_SIZE, EMBEDDING_DIM, UNITS)
    encoder.load_weights(ENCODER_PATH)
    decoder.load_weights(DECODER_PATH)

    # 输入到encoder
    _, encoder_h_c, encoder_h_t = encoder(sentences_encoder)

    # 创建起始符号<S>作为decode的第一个输入
    dec_input = tf.expand_dims([target_word_index['<S>']], 1)
    translate_Chinese = ''

    # 进行翻译
    for i in range(TRAGET_MAX_LEN):
        prediction, encoder_h_c, encoder_h_t = decoder([dec_input, encoder_h_c, encoder_h_t])

        # 预测下一个词
        word_index = np.argmax(prediction[0])

        # 预测的词
        pred_word = target_index_word.get(word_index)

        # 预测到终止符
        if pred_word == '<E>':
            break

        translate_Chinese += pred_word

        dec_input = tf.expand_dims([word_index], 1)

    print('Translate Chinese:{}\n'.format(translate_Chinese))


if __name__ == '__main__':
    # 获取数据集
    encoder_input, encoder_target, input_word_index, input_index_word, target_word_index, target_index_word = du.get_encoder_data()
    train_model(encoder_input, encoder_target, model_exist=True)

    test_sentence = 'Join us.'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'Get out of bed!'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'Happy New Year!'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'Who helps her?'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'A bird can fly.'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'I\'m so excited.'
    translate(test_sentence, input_word_index, target_index_word)

    test_sentence = 'Let me come in'
    translate(test_sentence, input_word_index, target_index_word)
