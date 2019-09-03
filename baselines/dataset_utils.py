#
"""
Various utilities specific to dataset processing.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import nltk

import pickle
import random
import tensorflow as tf
import numpy as np


class SpecialTokens(object):
	"""Special tokens, including :attr:`PAD`, :attr:`BOS`, :attr:`EOS`,
	:attr:`UNK`. These tokens will by default have token ids 0, 1, 3, 4,
	respectively.
	"""
	PAD = "<PAD>"
	BOS = "<BOS>"
	EOS = "<EOS>"
	UNK = "<UNK>"

class Batch:
	'''
	batch:'text':array([[bos,...,eos],[bos,...,eos]],dtype=object), Batch_size x max_seq_len+2
	'length': array([50, 50, 50], dtype=int32), 50 = max_seq_len+2
	'text_id': array([[1,...,2],[1,...,2]]), Batch_size x max_seq_len+2
	'''
	def __init__(self):
		self.text = []
		self.length = []
		self.text_id = []

def loadDataset(train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams):
	'''
	:return: word2id, id2word, traindata, testdata
	'''
	random.seed(666)
	train_dataset_path = train_dataset_hparams["dataset"]["files"]
	test_dataset_path = test_dataset_hparams["dataset"]["files"]
	valid_dataset_path = valid_dataset_hparams["dataset"]["files"]
	vocab_dataset_path = train_dataset_hparams["dataset"]["vocab_file"]
	print('Loading training dataset from {}'.format(train_dataset_path))
	traindata = []
	trainuser = []
	traintime = []
	with open(train_dataset_path) as f:
		for line in f:
			temp = line.strip().split('\t')
			trainuser.append(int(temp[0]))
			traintime.append(temp[1])
			traj = temp[2].split(' ')
			traindata.append(traj)
	print('Loading testing dataset from {}'.format(test_dataset_path))
	testdata = []
	testuser = []
	testtime = []
	with open(test_dataset_path) as f:
		for line in f:
			temp = line.strip().split('\t')
			testuser.append(int(temp[0]))
			testtime.append(temp[1])
			traj = temp[2].split(' ')
			testdata.append(traj)
	print('Loading valid dataset from {}'.format(valid_dataset_path))
	validdata = []
	validuser = []
	validtime = []
	with open(valid_dataset_path) as f:
		for line in f:
			temp = line.strip().split('\t')
			validuser.append(int(temp[0]))
			validtime.append(temp[1])
			traj = temp[2].split(' ')
			validdata.append(traj)
	print('Loading vocabulary dataset from {}'.format(vocab_dataset_path))
	word2id = {}
	id2word = {}
	with open(vocab_dataset_path) as f:
		word2id[SpecialTokens.PAD] = 0
		id2word[0] = SpecialTokens.PAD
		word2id[SpecialTokens.BOS] = 1
		id2word[1] = SpecialTokens.BOS
		word2id[SpecialTokens.EOS] = 2
		id2word[2] = SpecialTokens.EOS
		word2id[SpecialTokens.UNK] = 3
		id2word[3] = SpecialTokens.UNK
		for index, line in enumerate(f):
			word2id[line.strip()] = index+4
			id2word[index+4] = line.strip()
	if train_dataset_hparams['allow_smaller_final_batch'] == False:
		train_len = len(traindata)
		#print(int(train_len/train_dataset_hparams['batch_size'])*train_dataset_hparams['batch_size'])
		traindata = traindata[:int(train_len/train_dataset_hparams['batch_size'])*train_dataset_hparams['batch_size']]
		trainuser = trainuser[:int(train_len/train_dataset_hparams['batch_size'])*train_dataset_hparams['batch_size']]
		traintime = traintime[:int(train_len/train_dataset_hparams['batch_size'])*train_dataset_hparams['batch_size']]
		test_len = len(testdata)
		testdata = testdata[:int(test_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
		testuser = testuser[:int(test_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
		testtime = testtime[:int(test_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
		valid_len = len(validdata)
		validdata = validdata[:int(valid_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
		validuser = validuser[:int(valid_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
		validtime = validtime[:int(valid_len/test_dataset_hparams['batch_size'])*test_dataset_hparams['batch_size']]
	return word2id, id2word, traindata, testdata, trainuser, traintime, testuser, testtime, validdata, validuser, validtime

def createBatch(word2id, id2word, data, dataset_hparams):
	'''
	data: samples
	dataset_hparams: train_dataset_hparams["batch_size"],test_dataset_hparams["batch_size"]
	return: one batch
	'''
	random.seed(666)
	random.shuffle(data)
	batchSize = dataset_hparams["batch_size"]
	seq_len = dataset_hparams["dataset"]["max_seq_length"]
	for i in range(0, len(data), batchSize):
		samples = data[i:i+batchSize]
		one_batch = {'text':[],'length':[],'text_ids':[]}
		for sample in samples:
			text_seq = [SpecialTokens.BOS]
			text_seq.extend(sample)
			text_seq.append(SpecialTokens.EOS)
			text_ids = [word2id[word] for word in text_seq]
			one_batch['text'].append(text_seq)
			one_batch['length'].append(seq_len+2)
			one_batch['text_ids'].append(text_ids)
		one_batch['text'] = np.array(one_batch['text'],dtype=int)
		one_batch['length'] = np.array(one_batch['length'],dtype=int)
		one_batch['text_ids'] = np.array(one_batch['text_ids'])
		yield one_batch

def createData(word2id, id2word, data, dataset_hparams, user_id, time_id):
	'''
	data: samples
	dataset_hparams: train_dataset_hparams["batch_size"],test_dataset_hparams["batch_size"]
	return: one batch
	'''
	random.shuffle(data)
	seq_len = dataset_hparams["dataset"]["max_seq_length"]
	one_batch = {'text':[],'length':[],'text_ids':[], 'user_id':[], 'time_id':[]}
	print("size:", len(data))
	for i in range(0, len(data)):
		sample = data[i]
		text_seq = [SpecialTokens.BOS]
		text_seq.extend(sample)
		text_seq.append(SpecialTokens.EOS)
		text_ids = [word2id[word] for word in text_seq]
		one_batch['text'].append(text_seq)
		one_batch['length'].append(seq_len+2)
		one_batch['text_ids'].append(text_ids)
		one_batch['user_id'].append(user_id[i])
		one_batch['time_id'].append(time_id[i])
	one_batch['text'] = np.array(one_batch['text'],dtype=object)
	print("text shape = ", one_batch['text'].shape)
	one_batch['length'] = np.array(one_batch['length'],dtype=int)
	print("length shape = ", one_batch['length'].shape)
	one_batch['text_ids'] = np.array(one_batch['text_ids'],dtype=int)
	print("text_ids shape = ", one_batch['text_ids'].shape)
	one_batch['user_id'] = np.array(one_batch['user_id'],dtype=int).reshape((-1,1))
	print("user_id shape = ", one_batch['user_id'].shape)
	one_batch['time_id'] = np.array(one_batch['time_id'],dtype=str).reshape((-1,1))
	print("time_id shape = ", one_batch['time_id'].shape)
	return 	one_batch

"""
data_iterators.py
"""
class TrainTestDataIterator(object):
	def __init__(self, train=None, test=None, valid = None):
		self._datasets = {}
		self._train_name = 'train'
		self._test_name = 'test'
		self._valid_name = 'valid'
		if train is not None:
			self._datasets[self._train_name] = train
		if test is not None:
			self._datasets[self._test_name] = test   #dataser_dict = {'train:', trainDataset, 'test:', testDataset}
		if valid is not None:
			self._datasets[self._valid_name] = valid
		arb_dataset = self._datasets[next(iter(self._datasets))]
		self._iterator = tf.data.Iterator.from_structure(arb_dataset.output_types, arb_dataset.output_shapes)
		self._iterator_init_ops = {name: self._iterator.make_initializer(d) for name, d in self._datasets.items()}

	def switch_to_train_data(self, sess):
		sess.run(self._iterator_init_ops[self._train_name])

	def switch_to_test_data(self, sess):
		sess.run(self._iterator_init_ops[self._test_name])
		
	def switch_to_valid_data(self, sess):
		sess.run(self._iterator_init_ops[self._valid_name])

	def get_next(self):
		return self._iterator.get_next()

def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False,
                            zero_pad=True):
    """Cross entropy with label smoothing to limit over-confidence.
    Args:
        logits: Tensor of size [batch_size, ?, vocab_size]
        labels: Tensor of size [batch_size, ?]
        vocab_size: Tensor representing the size of the vocabulary.
        confidence: Used to determine on and off values for label smoothing.
            If `gaussian` is true, `confidence` is the variance to the gaussian
            distribution.
        gaussian: Uses a gaussian distribution for label smoothing
        zero_pad: use 0 as the probabitlity of the padding
            in the smoothed labels. By setting this, we replicate the
            numeric calculation of tensor2tensor, which doesn't set the
            <BOS> token in the vocabulary.
    Returns:
        the cross entropy loss.
    """
    with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
        # Low confidence is given to all non-true labels, uniformly.
        if zero_pad:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 2)
        else:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)

        if gaussian and confidence > 0.0:
            labels = tf.cast(labels, tf.float32)
            normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
            soft_targets = normal_dist.prob(
                tf.cast(tf.range(vocab_size), tf.float32)\
                    [:, None, None])
            # Reordering soft_targets from [vocab_size, batch_size, ?]
            # to match logits: [batch_size, ?, vocab_size]
            soft_targets = tf.transpose(soft_targets, perm=[1, 2, 0])
        else:
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence,
                dtype=logits.dtype)
        if zero_pad:
            soft_targets = tf.concat([tf.expand_dims(\
                tf.zeros_like(labels, dtype=tf.float32), 2),\
                soft_targets[:, :, 1:]], -1)

        if hasattr(tf.nn, 'softmax_cross_entropy_with_logits_v2'):
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        else:
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
        logits=logits, labels=soft_targets)

def parse_segment(lengths, masks):
    def _parse_segment(lengths, masks):
        """
        mask:        [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
        segment_ids: [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4],
                      [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4]] <- start from 0
        offsets:     [[0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0],
                      [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0]]
        :param masks:
        :return: segment_ids, offsets
        """
        segment_ids = np.full_like(masks, 0)
        offsets = np.full_like(masks, 0)
        batch_size = masks.shape[0]
        for i in range(batch_size):
            mask = masks[i]
            segment_ids[i][0] = 0
            for j in range(1, lengths[i]):
                if mask[j] == mask[j-1]:
                    segment_ids[i][j] = segment_ids[i][j-1]
                    offsets[i][j] = offsets[i][j-1] + 1
                else:
                    segment_ids[i][j] = segment_ids[i][j-1] + 1
                    offsets[i][j] = 0
        return segment_ids, offsets

    return tf.py_func(_parse_segment, [lengths, masks], [masks.dtype, masks.dtype])

def _pad_array_list(arrays, lens, pad_id):
    """
    :param ar: a list of 1-D array of different lengths, [batch_size, unfixed length]
    :return: a 2-D array, [batch_size, max_seq_len_in_original_list]
    """
    rst = []
    max_len = np.amax(lens)
    for idx, ar in enumerate(arrays):
        rst.append(np.pad(ar, (0, max_len - lens[idx]),
                          'constant', constant_values=pad_id))
    return np.array(rst), max_len

def _parse_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    """
    :param inputs:
    :param masks:
    :param start_positions: [batch_size, mask_num]
    :param end_positions:
    :param mask_id:
    :return:
    """
    inputs = inputs.tolist()
    masks = masks.tolist()
    l = len(inputs[0])
    rst, mask_rst, template_len = [], [], []
    for input, mask, start_pos_, end_pos_ in zip(inputs, masks, start_positions, end_positions):
        start_pos = [0]
        start_pos.extend(end_pos_.tolist())
        end_pos = start_pos_.tolist()
        end_pos.append(l)
        tmp_rst, tmp_mask = [], []
        for s, e in zip(start_pos, end_pos):
            tmp_rst.extend(input[s:e])
            tmp_rst.append(mask_id)
            tmp_mask.extend(mask[s:e])
            tmp_mask.append(1)
        tmp_rst.pop()  # delete the last mask_id
        tmp_mask.pop()
        rst.append(tmp_rst)
        mask_rst.append(tmp_mask)
        template_len.append(len(tmp_rst))
    rst, _ = _pad_array_list(rst, template_len, pad_id)
    mask_rst, _ = _pad_array_list(mask_rst, template_len, pad_id)
    return rst, mask_rst

def _prepare_squeezed_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    templates, template_masks = \
        tf.py_func(_parse_template,
                   [inputs, masks, start_positions, end_positions, mask_id, pad_id],
                   [tf.int64, tf.int64])
    batch_size = tf.shape(inputs)[0]
    templates = tf.reshape(templates, shape=tf.stack([batch_size, -1]))
    template_masks = tf.reshape(template_masks, shape=tf.stack([batch_size, -1]))
    return templates, template_masks


def generate_dynamic_mask(inputs, lengths, present_rate, mask_id, boa_id,
                          eoa_id, pad_id, partition_num,args):
    def _fill_mask(inputs, lengths, present_rate, eoa_id, pad_id, partition_num):
        """
        The input batch has the same mask pattern, randoms through max_seq_length in lengths.
        :param inputs:
        :param lengths:
        :param present_rate:
        :return: answers: a tensor of shape [batch_size, sum(unfixed_answer_len for each ans)]
        start_pos and end_pos marks out ranges for answers
        """
        def _fill_mask_py_func(inputs, lengths, present_rate, eoa_id, pad_id, partition_num, mask_id):
            # TODO(wanrong): bound check
            def _get_split_pos(masked_num):
                # split masked_num into partition_num segments
                if masked_num <= 1:
                    return [1] * (partition_num - 1)

                splitted = np.array_split(range(masked_num), partition_num)
                split_positions = [a.size for a in splitted]
                for i in range(1, partition_num):
                    split_positions[i] += split_positions[i - 1]
                return np.insert(split_positions, 0, 0, axis=0)

            batch_size = inputs.shape[0]
            lengths = (lengths+(args.history_num + 2)*2)/(args.history_num + 3)#length = 50
            present_rate = 1 - partition_num / (args.one_seq_length)
            #print(batch_size)
            #print("lengths:",lengths)
            #print("partition_num:", partition_num)
            #print(inputs)
            eoa_ = np.full(shape = (batch_size,1), fill_value = 2, dtype = np.int32)
            boa_ = np.full(shape = (batch_size,1), fill_value = 1, dtype = np.int32)
            history = []
            len_ = args.one_seq_length + 1
            seq_len = args.one_seq_length
            his_num = args.history_num
            for i in range(his_num):
                if i == 0:
                    h = np.concatenate((inputs[:,:len_],eoa_),axis = -1)
                    history = h #bs*50
                else:
                    h = np.concatenate((boa_,inputs[:,len_ + seq_len * (i-1):len_  + seq_len * i],eoa_), axis = -1)
                    #history.append(h)
                    history = np.concatenate((history,h),axis = -1)
            history = history.reshape((batch_size,seq_len+2,his_num))#bs*50*3
            history = history.transpose((0,2,1))#bs*3*50
            #print("history:",history.shape)
            #history = np.concatenate((inputs[:,:49],eoa_),axis = -1)
            #history1 = np.concatenate((boa_,inputs[:,49:49+48],eoa_), axis = -1)
            #history2 = np.concatenate((boa_,inputs[:,49+48:49+48*2],eoa_), axis = -1)

            linear = np.concatenate((boa_,inputs[:,len_ + seq_len * (his_num):len_  + seq_len * (his_num + 1)], eoa_),axis = -1)
            tmp = np.concatenate((boa_ ,inputs[:,-len_:]), axis = -1)#template 50
            inputs = np.concatenate((boa_,inputs[:,len_ + seq_len * (his_num-1):len_  + seq_len * his_num], eoa_),axis = -1)#his1+seq 50
            if args.if_linear == 1:
                inputs_ = inputs
                inputs = linear

            masked_nums = ((lengths - 2) * (1 - present_rate)).astype(np.int64)  # [batch_size]
            split_positions = \
                [_get_split_pos(masked_num) for masked_num in masked_nums]  # [batch_size, partition_num+1]
            #print len(split_positions)

            # calculate the length of each mask segment
            mask_lengths = np.zeros(shape=(batch_size, partition_num), dtype=np.int64)
            left_len = np.zeros(shape=(batch_size, partition_num + 1), dtype=np.int64)  # add a -1 at the end
            for bid, split_position in enumerate(split_positions):
                for idx, (prev, cur) in enumerate(zip(split_position[:-1], split_position[1:])):
                    mask_lengths[bid][idx] = cur - prev
                left_len[bid][-1] = 0  # leave <EOS> unmasked
                for idx, cur_len in reversed(list(enumerate(mask_lengths[bid]))):
                    left_len[bid][idx] = left_len[bid][idx+1] + cur_len + 1
            left_len = left_len[:, :-1]  # remove last column
            #print("mask lengths:",mask_lengths)
            #print "left_len:", left_len

            # splitting
            start_positions = np.zeros(shape=(batch_size, 1))
            end_positions = np.zeros(shape=(batch_size, 1))
            answers = np.zeros((batch_size, 0))
            partitions = np.array([])
            masks = np.full_like(inputs, 0)
            #print("masks:",masks.shape)
            after_pad_ans_lens = np.zeros(shape=partition_num)
            boa = np.full(shape=(batch_size, 1), fill_value=boa_id)
            #generate begin pos
            #print("***************************")
            s_random = []

            for b in range(batch_size):
                s = []
                m = tmp[b]
                #print("************************")
                #print("template: ",m)
                #print("inputs: ", inputs[b])
                #print("t:", m)
                for t in range(len(m)):
                    #print(m[t],mask_id)
                    if m[t] == mask_id:
                        #print("ok")
                        s.append(t)
                #print("blank:", s)
                if args.if_linear == 1:
                    for i in s:
                        inputs[b][i] = inputs_[b][i]#give the blank ground truth
                s_random.append(s)

            for i in range(1, partition_num + 1):
                idx = i - 1  # ignore padding 0 in start/end_positions
                # get start and end position for current mask
                cur_start_pos = np.zeros(shape=(batch_size, 1), dtype=np.int64)
                cur_end_pos = np.zeros(shape=(batch_size, 1), dtype=np.int64)
                cur_answers = []
                for bid in range(batch_size):
                    s = end_positions[bid][idx] + 1
                    e = lengths[bid] - left_len[bid][idx] + 1
                    #cur_start_pos[bid][0] = s + (e - s) / (partition_num + 1)
                    cur_start_pos[bid][0] = s_random[bid][idx]
                    cur_end_pos[bid][0] = cur_start_pos[bid][0] + mask_lengths[bid][idx]
                    cur_answers.append(
                        np.append(inputs[bid][cur_start_pos[bid][0]:cur_end_pos[bid][0]], eoa_id))
                    # update mask
                    for j in range(cur_start_pos[bid][0], cur_end_pos[bid][0]):
                        masks[bid][j] = 1  # set masked element to 1
                start_positions = np.concatenate((start_positions, cur_start_pos), axis=1)
                end_positions = np.concatenate((end_positions, cur_end_pos), axis=1)

                # pad cur_answers to same length
                cur_padded_ans, cur_max_len = _pad_array_list(cur_answers, mask_lengths[:, idx], pad_id)
                #print "cur_padded_ans:",cur_padded_ans
                cur_padded_ans = np.concatenate((boa, cur_padded_ans), axis=1)
                after_pad_ans_lens[idx] = cur_max_len
                answers = np.concatenate((answers, cur_padded_ans), axis=1)

                # generate current partition index
                #print cur_padded_ans[0]
                cur_idx = np.full_like(cur_padded_ans[0], idx)
                partitions = np.concatenate((partitions, cur_idx), axis=0)
            #print("masks:",masks)
            #print("start_position:",start_positions[:, 1:].astype(np.int64))
            #print("end_position:",end_positions[:, 1:].astype(np.int64))
            #print("answer:",answers.astype(np.int64))
            #print("answer:", answers.shape)


            return masks, start_positions[:, 1:].astype(np.int64),\
                   end_positions[:, 1:].astype(np.int64),\
                   answers.astype(np.int64), after_pad_ans_lens.astype(np.int64), \
                       mask_lengths.astype(np.int32), partitions.astype(np.int32), \
                       np.array(history).astype(np.int64), \
                       inputs.astype(np.int64), lengths.astype(np.int64)

        eoa_id = tf.Variable(eoa_id, dtype=tf.int64,trainable=False)
        present_rate = tf.Variable(present_rate, dtype=tf.float32,trainable=False)
        partition_num = tf.Variable(partition_num, dtype=tf.int64,trainable=False)
        return tf.py_func(_fill_mask_py_func,
                          [inputs, lengths, present_rate, eoa_id, pad_id, partition_num, mask_id],
                          [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int32, tf.int32,tf.int64, tf.int64, tf.int64])

    masks, start_positions, end_positions, answers, after_pad_ans_lens, true_ans_lens, partitions, history,inputs, lengths = \
        _fill_mask(inputs, lengths, present_rate, eoa_id, pad_id, partition_num)
    answers = tf.dynamic_partition(data=tf.transpose(answers, perm=[1, 0]),  # [sum(lens), batch_size]
                                   partitions=partitions,
                                   num_partitions=partition_num)
    answers = [tf.transpose(ans, perm=[1, 0]) for ans in answers]
    mask_id = tf.Variable(mask_id, dtype=tf.int64,trainable=False)
    pad_id = tf.Variable(pad_id, dtype=tf.int64,trainable=False)
    templates, template_masks = \
        _prepare_squeezed_template(inputs, masks, start_positions, end_positions, mask_id, pad_id)

    return masks, answers, after_pad_ans_lens, true_ans_lens, templates, template_masks, \
           start_positions, end_positions, history, inputs, lengths

def generate_prediction_offsets(inputs, max_length):
    batch_size = tf.shape(inputs)[0]
    max_length = tf.cast(max_length, dtype=tf.int32)
    _, offsets = parse_segment(tf.fill([batch_size], max_length),
                               tf.fill([batch_size, max_length], 0))
    return tf.cast(offsets, dtype=tf.int64)

def generate_prediction_segment_ids(inputs, segment_id, max_length):
    batch_size = tf.shape(inputs)[0]
    return tf.cast(tf.fill([batch_size, tf.cast(max_length, dtype=tf.int32)], segment_id), dtype=tf.int64)

def _get_start_end_pos(mask_by_word, mask_id):
    def _get_start_end_pos_py_func(mask_by_word, mask_id):
        #print("**********************************")
        #print("mask by word", mask_by_word)
        start_pos, end_pos = [[-2] for i in range(len(mask_by_word))], [[-2] for i in range(len(mask_by_word))]
        for idx, template in enumerate(mask_by_word):
            for i, word in enumerate(template):
                if word == mask_id:
                    #if end_pos[idx][-1] == i:
                    #    end_pos[idx].pop()
                    #else:
                    #    start_pos[idx].append(i)
                    start_pos[idx].append(i)
                    end_pos[idx].append(i+1)
        #print("start pos", start_pos)
        #print("end pos", end_pos)
        #s = np.array(start_pos)
        #e = np.array(end_pos)
        #print("start shape:",s.shape)
        #print("end shape:",e.shape)
        #print(s[:,1:].astype(np.int64))
        #print(e[:,1:].astype(np.int64))
        return np.array(start_pos)[:, 1:].astype(np.int64), np.array(end_pos)[:, 1:].astype(np.int64)

    mask_id = tf.Variable(mask_id, dtype=tf.int64,trainable=False)
    return tf.py_func(_get_start_end_pos_py_func,
                      [mask_by_word, mask_id],
                      [tf.int64, tf.int64])

def prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id):
    """
    mask_id = 7
    pad_id = 6
    inputs:        [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]] <- a tensor
    mask:          [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
    masked_inputs: [[3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1], [2, 1, 4, 3, 7, 7, 5, 4, 7, 7, 5]]
    templates:     [[3, 5, 4, 7, 1, 3, 3, 7, 1], [2, 1, 4, 3, 7, 5, 4, 7, 5]]
    segment_ids:   [[1, 1, 1, 2, 3, 3, 3, 4, 5], [1, 1, 1, 1, 2, 3, 3, 4, 5]]
    answers:       [[[4, 2], [5, 1]],
                    [[2, 5], [3, 1]]] <- used as decode outputs(targets) in training
    :param masked_inputs:
    :param mask_id:
    :return: masked_inputs, segment_ids, answers
    """
    inputs = data_batch['text_ids']
    lengths = data_batch['length']
    masks, answers, after_pad_ans_lens, true_ans_lens, templates, template_masks,\
        start_positions, end_positions, history,inputs, lengths = \
        generate_dynamic_mask(inputs, lengths, args.present_rate, mask_id, boa_id,
                              eoa_id, pad_id, args.blank_num,args)

    template_lengths = tf.fill(tf.shape(lengths), tf.shape(templates)[1])
    template_segment_ids, template_offsets = \
        parse_segment(template_lengths, template_masks)
    all_masked_out = tf.cast(tf.fill(tf.shape(inputs), mask_id), dtype=tf.int64)
    masked_inputs = tf.where(tf.equal(masks, tf.ones_like(inputs)),
                             all_masked_out, inputs)
    his_num = tf.Variable(args.history_num, dtype=tf.int64,trainable=False)
    template_pack = {
        'masks': masks,
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'templates': templates,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'template_lengths': template_lengths,
        'history':history,
        'history_num':his_num
    }

    answer_packs = []
    for idx, answer in enumerate(answers):
        mask_len = after_pad_ans_lens[idx] + 2  # has <eoa> and <boa>
        answer_segment_ids = generate_prediction_segment_ids(answer, idx * 2 + 1, mask_len)
        answer_offsets = generate_prediction_offsets(answer, mask_len)
        #answer1 = answer
        answer = tf.reshape(answer, shape=tf.stack([50, mask_len]))
        lengths = tf.reshape(true_ans_lens[:, idx], shape=tf.stack([-1]))
        #print("ans text_ids:",answer1)
        #print("ans seg id:",answer_segment_ids)
        #print("ans offsets:",answer_offsets)
        #print("ans len:", lengths)
        answer_packs.append({
            'text_ids': answer,
            'segment_ids': answer_segment_ids,
            'offsets': answer_offsets,
            'lengths': lengths
        })

    return template_pack, answer_packs

def _split_template(template, mask_start_positions, mask_end_positions):
    """
    template: [3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1]
    start_positions: [3, 8], starting positions of the masks
    end_positions: [5, 10], ending positions of the masks
    will be split into: [[3, 5, 4], [1, 3, 3], [1]]
    :param template: a list of numbers
    :return:
    """
    rst = []
    start_positions = [0] + mask_end_positions.tolist()
    end_positions = mask_start_positions.tolist() + [len(template)]
    for s, e in zip(start_positions, end_positions):
        rst.append(template[s: e])
    return rst

def _merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id):
    """
    template_segments: [[3, 5, 4], [1, 3, 3], [1]]
    fillings: [[4, 2], [2, 5]]
    rst: [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    :param template_segments:
    :param fillings:
    :return:
    """
    def _parse(id_list, eoa_id, pad_id, eos_id):
        rst = []
        for id in id_list:
            if id in [eoa_id, eos_id]:
                break
            elif id is not pad_id:
                rst.append(id)
        if len(rst) == 0:
            rst.append(7)
            #print("refix")
        return rst

    template_segment_num = len(template_segments)
    filling_segment_num = len(fillings)
    assert template_segment_num == filling_segment_num or \
           template_segment_num == filling_segment_num + 1

    rst = []
    for i in range(filling_segment_num):
        rst.extend(template_segments[i])
        rst.extend(_parse(fillings[i], eoa_id, pad_id, eos_id))
    if template_segment_num > filling_segment_num:
        rst.extend(template_segments[-1])
    return rst

def fill_template(template_pack, predictions, eoa_id, pad_id, eos_id):
    """
    :param template: [batch_size, max_seq_len]
    :param mask: [batch_size, max_seq_len]
    :param predictions: a list of tensors
    :return:
    """
    def _transpose(a):
        """
        :param a: mask_num * batch_size * undefined_len
        :return: batch_size * mask_num * undefined_len
        """
        rst = []
        for _ in a[0]:
            rst.append([])
        for ar in a:
            for idx, sent in enumerate(ar):
                rst[idx].append(sent)
        return rst

    start_positions = template_pack['start_positions']
    end_positions = template_pack['end_positions']
    templates = template_pack['text_ids']
    templates = templates.tolist()
    predictions = [prediction.tolist() for prediction in predictions]  # mask_num * batch_size * undefined_len
    predictions = _transpose(predictions)
    rst = []
    for template, start_pos, end_pos, fillings in zip(templates, start_positions, end_positions, predictions):
        template_segments = _split_template(template, start_pos, end_pos)
        rst.append(_merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id))
    return rst

def update_template_pack(template_pack, filling, mask_id, eoa_id, pad_id):
    def _fill_segment(masked_by_word_template, filling, start_pos, end_pos, eoa_id, pad_id):
        def _fill_segment_py_func(masked_by_word_templates, fillings, start_pos, end_pos, eoa_id, pad_id):
            masked_by_word_templates = masked_by_word_templates.tolist()
            fillings = fillings.tolist()
            start_pos = start_pos.tolist()
            end_pos = end_pos.tolist()
            rst, length = [], []
            for template, filling, s, e in zip(masked_by_word_templates, fillings, start_pos, end_pos):
                try:
                    #end_pos = filling.index(eoa_id)
                    #filling = filling[:end_pos]
                    filling_i = filling
                    if len(filling) == 0:
                        filling = [7]
                    else:
                        filling = [filling[0]]
                    #prevent generate <m> which will influence the operation below
                    if filling[0] <= 7:
                        filling[0] = 7
                    if filling[0] == 4:
                        print("error")
                        print(filling_i)

                except ValueError:
                    pass
                cur_rst = template[:s] + filling + template[e:]
                length.append(len(cur_rst))
                rst.append(cur_rst)
            rst, _ = _pad_array_list(rst, length, pad_id)
            return rst
        return tf.py_func(_fill_segment_py_func,
                          [masked_by_word_template, filling, start_pos, end_pos, eoa_id, pad_id],
                          tf.int64)

    eoa_id = tf.Variable(eoa_id, dtype=tf.int64, trainable = False)
    pad_id = tf.Variable(pad_id, dtype=tf.int64,trainable=False)
    #print("template",template_pack)
    #print("end_position:",template_pack['end_positions'])
    masked_inputs = _fill_segment(template_pack['text_ids'], filling,
                                  template_pack['start_positions'][:, 0],
                                  template_pack['end_positions'][:, 0], eoa_id, pad_id)
    masks = tf.where(tf.equal(masked_inputs, mask_id * tf.ones_like(masked_inputs)),
                     tf.ones_like(masked_inputs), tf.zeros_like(masked_inputs))
    start_positions, end_positions = _get_start_end_pos(masked_inputs, mask_id)
    templates, template_masks = \
        _prepare_squeezed_template(masked_inputs, masks, start_positions, end_positions, mask_id, pad_id)
    template_lengths = tf.fill(tf.shape(template_pack['template_lengths']), tf.shape(templates)[1])
    template_segment_ids, template_offsets = \
        parse_segment(template_lengths, template_masks)
    return_pack = {
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'templates': templates,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'masks': masks,
        'template_lengths': template_lengths,
        'history':template_pack['history'],#bs*his_num*50
        'history_num':template_pack['history_num'],
    }
    return return_pack
