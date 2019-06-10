# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
from utils import calc_num_batches
import tokenization

# 许海明
def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.strip() for line in open(vocab_fpath, mode='r',encoding="utf-8").read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token




# 许海明
def load_data(fpath1, fpath2, maxlen1, maxlen2,vocab_fpath):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_fpath, do_lower_case=True)

    with open(fpath1,mode= 'r',encoding="utf-8") as f1, open(fpath2,mode= 'r',encoding="utf-8") as f2:
        for sent1, sent2 in zip(f1, f2):
            # 这里隐藏着 一对翻译对一个不满足，所有的都扔掉
            sent1 = sent1.strip()
            sent2 = sent2.strip()
            if sent1 != "" and sent2 != "":
                if len(tokenizer.tokenize(sent1)) + 1 > maxlen1:
                    continue  # 1: <s>
                if len(tokenizer.tokenize(sent2)) + 1 > maxlen2:
                    continue  # 1: </s>
                sents1.append(sent1)
                sents2.append(sent2)
            else:
                continue
    return sents1, sents2





# 返回的是一个 iter()对象
def generator_fn(sents1, sents2,vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents


    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_fpath, do_lower_case=True)
    for source, target in zip(sents1, sents2):
        sent1 = tokenizer.tokenize(source.strip())
        sent2 = tokenizer.tokenize(target.strip())
               # source 加上结束符号
        sent1.append("[SEP]")
        # target 加上开始和结束符号
        sent2.insert(0, "[CLS]")
        sent2.append("[SEP]")

        sent1_ids = tokenizer.convert_tokens_to_ids(sent1)
        sent2_ids = tokenizer.convert_tokens_to_ids(sent2)
        decoder_input, y = sent2_ids[:-1], sent2_ids[1:]
        x_seqlen, y_seqlen = len(sent1_ids), len(sent2_ids)

        yield (sent1_ids, x_seqlen, source.strip()), (decoder_input, y, y_seqlen,target.strip())

def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    # (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    '''
    If the dimension is unknown (e.g. tf.Dimension(None)), the component will be padded out to the maximum length of all elements in that dimension.
    '''
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

#许海明
def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''

    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2,vocab_fpath)
    # 这里返回的是一个 dataset 对象
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
