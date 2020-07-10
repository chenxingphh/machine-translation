# coding-utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import re

# 数据集路径
DATA_PATH = r'data\cmn.txt'
# 训练的数据集数目
NUM_SAMPLE = 1000


def preprocess_sentence(sent, cmn=True, add=True):
    '''
    过滤字符串中的标点符号，并保证每个字符之间使用空格进行隔开

    :param sent: 一行字符串
    :param cmn: 是否为中文
    :param add: 是否添加起始和终止符
    :return:
    '''
    # 过滤换行
    sent = sent.replace('\n', '')

    # 过滤标点符号,只保留字母和数字
    sent = re.sub(r'([^\s\w]|_)+', "", sent)

    # 为每一个字之间添加空格
    if cmn:
        sent = ' '.join(sent)

    # 添加起始和终止符
    if add:
        sent = '<S> ' + sent + ' <E>'

    return sent


def get_max_sent(sents):
    '''
    判断数字编码后的最大句子长度
    :param sents: e.g:[[1,2,3],[8,7]]
    :return:
    '''
    return max([len(sent) for sent in sents])


def encoder_sentences(sentences):
    '''
    将给定的文本list映射为数字，并且字之间使用空格进行分割
    :param sentences: e.g:['<S> 你 好 <E>','<S> 嗨 <E>',]
    :return:
    '''
    # 对文本数字化
    token = keras.preprocessing.text.Tokenizer(num_words=2000,
                                               filters='',
                                               lower=False,
                                               split=' ',  # 按照空格进行分词
                                               char_level=False)  # 词粒度
    token.fit_on_texts(sentences)
    sentences_encoder = token.texts_to_sequences(sentences)  # 将每一个word映射为一个整数

    # 最大句子长度
    max_len = get_max_sent(sentences_encoder)

    # 进行填充
    sentences_encoder = tf.keras.preprocessing.sequence.pad_sequences(sentences_encoder, maxlen=max_len, padding="post",
                                                                      value=0)

    # 为词典添加<UNK>
    word_index = token.word_index
    index_word = {word_index[k]: k for k in word_index.keys()}

    # index_word = token.index_word
    word_index['<UNK>'] = 0
    index_word[0] = '<UNK>'

    return sentences_encoder, word_index, index_word, max_len


def get_encoder_data(data_path=DATA_PATH):
    '''
    对输入的文本进行数字编码，使用空格进行分割
    :param data_path:
    :return:
    '''
    input_texts = []
    target_texts = []

    # 读取文本信息
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[:min(len(lines), NUM_SAMPLE)]:
        input_text, target_text = line.split('\t')[:2]

        # 将源句子进行逆序
        input_texts.append(preprocess_sentence(input_text, cmn=False, add=False))

        # 为目标句子添加起始符号（\t）、终止符号(\n)
        target_texts.append(preprocess_sentence(target_text))

    # 对input文本数字化
    input_encoder, input_word_index, input_index_word, input_max_len = encoder_sentences(input_texts)

    # 对target文本数字化
    target_encoder, target_word_index, target_index_word, target_max_len = encoder_sentences(target_texts)

    MAX_INPUT_LEN = input_max_len
    MAX_TRAGET_LEN = target_max_len
    print('MAX_INPUT_LEN', MAX_INPUT_LEN)
    print('MAX_TRAGET_LEN', MAX_TRAGET_LEN)

    return input_encoder, target_encoder, input_word_index, input_index_word, target_word_index, target_index_word


# 将编码的数字转换为文本
def decoder_sentence(encoder_sent, index_word):
    decoder_sent = []

    for i in encoder_sent:
        decoder_sent.append(index_word.get(i, 0))

    return decoder_sent


if __name__ == '__main__':
    input_encoder, target_encoder, input_word_index, input_index_word, target_word_index, target_index_word = get_encoder_data()

    # 对文本信息进行解码
    print('encoder input:', input_encoder[10])
    print('input:', decoder_sentence(input_encoder[10], input_index_word))

    # 对文本信息进行解码
    print('encoder input:', target_encoder[10])
    print('input:', decoder_sentence(target_encoder[10], target_index_word))

    # input_index_word,input_word_index
    print('input vocabulary size', len(input_index_word.keys()), len(input_word_index.keys()))
    print('target vocabulary size', len(target_word_index.keys()), len(target_index_word.keys()))
