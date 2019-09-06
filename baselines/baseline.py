from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import os
import setproctitle
import random
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
#setproctitle.setproctitle('TrajFill_seq48@qiyunhan')


import sys
import codecs
import numpy as np
import tensorflow as tf
from sklearn import metrics
import pickle
import json
import time
#import texar as tx
#from matplotlib import pyplot as plt
#plt.switch_backend('agg')

import self_attn_hyperparams_region
import dataset_utils
from dataset_utils import TrainTestDataIterator
import Embedder
from model_utils import RNN_Module

print("load module sucessful")
"map lnglat to grids and map grid to regionID, lookup "
grid2region = json.load(open('/data1/xiatong/trajen/region_extra/region_beijing_grid2region_REID_no_less10.json'))  #GRID2REGION lookup
region_distacne = np.load('/data1/xiatong/trajen/region_extra/region_distance_no_less10.npy') #Reigon distance lookup
print('Shape of region distance:', region_distacne.shape)
RN = len(region_distacne)
lng_ld = 116.1
lat_ld = 39.7
lng_max = 116.7
lat_max = 40.2
horizontal_number = 3000
vertical_number = 2500

def transcoor2grid(lng,lat):
	return str(int((lng-lng_ld)/0.0002)) + '-' + str(int((lat-lat_ld)/0.0002))

def transgrid2region(grid):
	count = 0
	while(1):
		if grid in grid2region:
			return grid2region[grid]
		else:
			grid_lng, grid_lat = int(grid.split('-')[0]),int(grid.split('-')[1])
			if count % 2 == 0:
				if grid_lat >= 1:
					grid_lat -= 1
			else: 
				if grid_lng >= 1:
					grid_lng -= 1
			grid = str(grid_lng) + '-' + str(grid_lat)
			count += 1

def get_acc(hyp, tplt, tgt,blank, prob5, prob3, prob, word2id):
	#prob: 12*vocab
	#print("*****************************")
	#print("tgt:",len(tgt), tgt)
	#print("gen:",len(hyp), hyp)
	#print("template:",len(tplt), tplt)
	#print(blank)
	vocab_size = len(word2id)
	recall_blank3 = []
	recall_blank5 = []
	mAP = []
	flag = 0
	if (len(hyp) != len(tgt)):
		flag += 1
		diff = len(tgt) - len(hyp)
		add = []
		for i in range(diff):
			add.append("*")
		hyp = hyp + add
	acc = 0
	count = 0
	for k in blank:
		if tgt[k] != "*":
			true_id = word2id[tgt[k]]
			#recall5
			pred_top_index = prob5[count,:]#k
			if true_id in pred_top_index:
				recall = 1.0
			else:
				recall = 0.0
			recall_blank5.append(recall)

			#recall3
			pred_top_index = prob3[count,:]#k
			if true_id in pred_top_index:
				recall = 1.0
			else:
				recall = 0.0
			recall_blank3.append(recall)

			#recall1
			if tgt[k] == hyp[k]:
				acc += 1

			#map
			combine = []
			pred_prob = prob[count,:]
			#print("************************************")
			#print("len pred_prob:", len(pred_prob))
			#print("sum pred_prob:", np.mean(pred_prob))

			truth_prob = [0] * len(pred_prob)
			truth_prob[true_id] = 1
			for t_p, p_p in zip(truth_prob, pred_prob):
				combine.append((t_p, p_p))
			combine = sorted(combine, key = lambda x:x[1], reverse = True)
			#print("*******************************")
			for idx, com in enumerate(combine):
				if com[0] == 1:
					#if idx < 10:
						#print(true_id, idx)
						#print(true_id, word2id(hyp[k], idx))
					mAP.append(1.0 / (idx + 1))
					break
		count += 1


	blank_acc = acc*1.0/count
	acc = 0
	count = 0
	for i in range(len(tgt[1:])):
		if tgt[i+1] != "*":
			count += 1
			if tgt[i+1] == hyp[i+1]:
				acc += 1
	all_acc = acc*1.0/count
	return blank_acc, all_acc, flag, recall_blank5, recall_blank3, mAP

def get_distance(hyp, tplt, tgt, blank):
	dis_blank = []
	#print("*************************")
	#print('hyp',hyp)
	#print('tplt',tplt)
	#print('tgt',tgt)
	#print('blank',blank)	
	for k in blank:
		if tgt[k] != "*" and hyp[k] not in ["*", "<BOA>" , "<EOA>", "<BOS>", "<m>", "<PAD>", "<EOS>", "<UNK>"]:
			try:
				d = region_distacne[int(tgt[k])][int(hyp[k])]
				dis_blank.append(d)
			except ValueError:#IndexError:
				print(k,blank,tgt,hyp)

	dis_all = []
	for k in range(len(tgt[1:])):
		if tgt[k+1] != "*" and hyp[k+1] not in ["*", "<BOA>", "<EOA>", "<BOS>", "<m>", "<PAD>", "<EOS>", "<UNK>"]:
			d = region_distacne[int(tgt[k+1])][int(hyp[k+1])]
			dis_all.append(d)
	return np.mean(dis_blank) if len(dis_blank) >0 else 100000, np.mean(dis_all) if len(dis_all) >0 else 80000

def get_rmse_loss(preds, tgt, id2word):
	#tgt blank + eos, shape = bs * 2
	#preds shape = bs * 2
	#select the item to calculate the RMSE

	def get_rmse_loss_(preds, tgt):
		#print("bbb")
		#print(preds.shape,preds)
		#print(tgt.shape,tgt)
		#def _id2word_map(id_arrays):
		#	return [' '.join([id2word[i] for i in sent]) for sent in id_arrays]
		#preds = preds.tolist()
		#tgt = tgt.tolist()
		#pred_, tgt_ = _id2word_map(preds[:,0]), _id2word_map(tgt[:,0])
		loss = []
		#print(len(pred_),pred_)
		for i in range(len(preds)):
			if preds[i,0] > 7 and tgt[i,0] > 7:
				pred_ = id2word[preds[i,0]]
				tgt_ = id2word[tgt[i,0]]
				loss.append(region_distacne[int(pred_)][int(tgt_)])
			else:
				continue
		#print(loss)
		if len(loss) == 0:
			loss.append(10000.0)
			#print("bad generate")
		return np.mean(loss).astype(np.float32)
	#print("aaa")
	print(preds)
	print(tgt)
	return tf.py_func(get_rmse_loss_, [preds, tgt], tf.float32)

def get_topk(logits_prob, k, batch_size):
	#inputs = bs*12*vocab
	t = tf.nn.top_k(logits_prob, k)
	t = t[1]#bs*12*k
	return t

def _main(_):
	hparams = self_attn_hyperparams_region.load_hyperparams()
	train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, \
	decoder_hparams, opt_hparams, opt_vars, loss_hparams, args = \
		hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
		hparams['test_dataset_hparams'], hparams['decoder_hparams'], \
		hparams['opt_hparams'], hparams['opt_vars'], \
		hparams['loss_hparams'], hparams['args']

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	setproctitle.setproctitle(args.process_name)
	# Dataset 
	word2id, id2word, traindata, testdata, trainuser, traintime, testuser, testtime, validdata, validuser, validtime = \
				dataset_utils.loadDataset(train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams)
	xing_id = word2id['*']
	mask_id = word2id['<m>']
	boa_id = word2id['<BOA>']
	eoa_id = word2id['<EOA>']
	eos_id = word2id['<EOS>']
	pad_id = word2id['<PAD>']
	unk_id = word2id['<UNK>']
	print("xing_id:",xing_id)
	print("mask id:",mask_id)
	print("boa_id",boa_id)
	print("eoa_id",eoa_id)
	print("eos_id",eos_id)
	print("pad_id",pad_id)
	print("unk_id",unk_id)

	print("begin train")
	train_pare = dataset_utils.createData(word2id, id2word, traindata, train_dataset_hparams, trainuser, traintime)
	print("end train")
	test_pare = dataset_utils.createData(word2id, id2word, testdata, test_dataset_hparams, testuser, testtime)
	print("end test")
	valid_pare = dataset_utils.createData(word2id, id2word, validdata, valid_dataset_hparams, validuser, validtime)

	traindataset = tf.data.Dataset.from_tensor_slices({"text": train_pare['text'],"length": train_pare['length'],"text_ids": train_pare['text_ids'], \
												"user_id": train_pare['user_id'], "time_id": train_pare['time_id']}).batch(train_dataset_hparams["batch_size"])
	testdataset = tf.data.Dataset.from_tensor_slices({"text": test_pare['text'],"length": test_pare['length'],"text_ids": test_pare['text_ids'], \
												"user_id": test_pare['user_id'], "time_id": test_pare['time_id']}).batch(test_dataset_hparams["batch_size"])
	validdataset = tf.data.Dataset.from_tensor_slices({"text": valid_pare['text'],"length": valid_pare['length'],"text_ids": valid_pare['text_ids'], \
												"user_id": valid_pare['user_id'], "time_id": valid_pare['time_id']}).batch(valid_dataset_hparams["batch_size"])
	iterator = TrainTestDataIterator(train=traindataset,test=testdataset, valid = validdataset)
	data_batch = iterator.get_next()


	template_pack, answer_packs = dataset_utils.prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id)
	
	dis_matrix = tf.Variable(region_distacne, name = 'distance', dtype = tf.float32,trainable = False)
	print("generating dis matrix successfully")
	# Model architecture
	embedder = Embedder.WordEmbedder(vocab_size=len(word2id),hparams=args.word_embedding_hparams)
	decoder = RNN_Module(embedding=embedder,hparams=decoder_hparams) #_dynamic
	cetp_loss = None
	cur_template_pack = template_pack
	inputs = []
	#logits: bs*12*vocab; preds: 12*bs*1
	is_training = tf.placeholder(tf.bool)
	logits, preds,  logits_prob = decoder._build(template_input_pack=template_pack,
								encoder_decoder_attention_bias=None,
								args=args, mask_id = mask_id,xing_id = xing_id, is_training = is_training)
	topk5 = get_topk(logits_prob, 5, args.batch_size)
	topk3 = get_topk(logits_prob, 3, args.batch_size)
	#attn = []
	count = tf.Variable(0, trainable=False)
	for hole in answer_packs:
		#print(hole['text_ids'])
		cur_loss = dataset_utils.smoothing_cross_entropy(
			logits[:, count:count + 1, :],#bs*12*vocab
			hole['text_ids'][:, 1:2],
			len(word2id),
			loss_hparams['label_confidence'])
		count = count + 1
		cetp_loss = cur_loss if cetp_loss is None \
			else tf.concat([cetp_loss, cur_loss], -1)

	cetp_loss = tf.reduce_mean(cetp_loss)

	global_step = tf.Variable(0, trainable=False)
	if args.learning_rate_strategy == 'static':
		learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
	elif args.learning_rate_strategy == 'dynamic':
		fstep = tf.to_float(global_step)
		learning_rate = opt_hparams['lr_constant'] \
						* args.hidden_dim ** -0.5 \
						* tf.minimum(fstep ** -0.5, fstep * opt_hparams['warmup_steps'] ** -1.5)
	else:
		raise ValueError('Unknown learning_rate_strategy: %s, expecting one of '
						 '[\'static\', \'dynamic\']' % args.learning_rate_strategy)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
									   beta1=opt_hparams['Adam_beta1'],
									   beta2=opt_hparams['Adam_beta2'],
									   epsilon=opt_hparams['Adam_epsilon'])
	#add regularization loss
	if args.reg == 1:
		tv = tf.trainable_variables()
		for item in tv:
			print(item)
		if args.reg_kind == 'l1':
			regularization_cost = tf.reduce_mean([ tf.contrib.layers.l1_regularizer(args.reg_lambda_l1)(v) for v in tv ])
		elif args.reg_kind == 'l2':
			print("l2 regulization")
			regularization_cost = tf.reduce_mean([ tf.contrib.layers.l2_regularizer(args.reg_lambda)(v) for v in tv ])
		elif args.reg_kind == 'l1_l2':
			regularization_cost = tf.reduce_mean([ tf.contrib.layers.l1_l2_regularizer(scale_l1 = args.reg_lambda_l1, scale_l2 = args.reg_lambda)(v) for v in tv ])
		cetp_loss_ = cetp_loss + regularization_cost
	else:
		cetp_loss_ = cetp_loss

	train_op = optimizer.minimize(cetp_loss_, global_step)
	

	def _train_epochs(session, cur_epoch, mode='train'):
		loss_lists, ppl_lists = [], []
		cnt = 0
		iterator.switch_to_train_data(session)
		while True:
			try:
				fetches = {
					'step': global_step,
					'lr': learning_rate,
					'loss': cetp_loss_,
					'embedder':embedder,
					#'data_batch':train_data_batch,
				}
				if mode == 'train':
					fetches['train_op'] = train_op

				feed = {is_training : True}
				if args.learning_rate_strategy == 'static':
					feed[learning_rate] = opt_vars['learning_rate']
				rtns = session.run(fetches, feed_dict=feed)
				step, loss = rtns['step'], rtns['loss']
				if rtns['step'] % 200 == 0:
					print("learning rate = ", rtns["lr"])

				embed = rtns['embedder']
				'''
				if cur_epoch % 2 == 0:
					embedd_path = args.analyse_data_save_dir + "embedding" + str(cur_epoch)  + ".npy"
					np.save(embedd_path,embed)
				'''
				loss_lists.append(loss)

				cnt += 1
				if mode is not 'train' and cnt >= 50:
					break
			except tf.errors.OutOfRangeError:
				if args.learning_rate_strategy == 'static':
					avg_loss = np.average(loss_lists)
					if avg_loss < opt_vars['best_train_loss']:
						opt_vars['best_train_loss'] = avg_loss
						opt_vars['epochs_not_improved'] = 0
					else:
						opt_vars['epochs_not_improved'] += 1
					if opt_vars['epochs_not_improved'] >= 8 and opt_vars['decay_time'] <= 3:
						opt_vars['learning_rate'] *= opt_vars['lr_decay_rate']
						print("[LR DECAY]: lr decay to %f at epoch %d" %
							  (opt_vars['learning_rate'], cur_epoch))
						opt_vars['decay_time'] += 1
				break
		#print("finish train")
		return loss_lists
	
	def _test_epoch(cur_sess, cur_epoch, mode='test'):
		#print("begin test")
		def _id2word_map(id_arrays):
			return [' '.join([id2word[i] for i in sent]) for sent in id_arrays]

		if mode == 'test':
			iterator.switch_to_test_data(cur_sess)
		elif mode == 'train':
			iterator.switch_to_train_data(cur_sess)
		else:
			iterator.switch_to_valid_data(cur_sess)

		templates_list, targets_list, hypothesis_list, user_list, week_list, prob5_list, prob3_list, prob_list = [], [], [], [], [], [], [], []
		cnt = 0
		loss_lists, ppl_lists = [], []
		s_random = []
		while True:
			try:
				
				fetches = {
					'data_batch': data_batch,
					'logits_5':topk5,
					'logits_3':topk3,
					'logits':logits_prob,
					'predictions': preds,
					'template': template_pack,
					'step': global_step,
					'loss': cetp_loss_,
				}

				feed = {is_training : False}
				rtns = cur_sess.run(fetches, feed_dict=feed)
				real_templates_, templates_, targets_t, predictions_ = \
					rtns['template']['templates'], rtns['template']['text_ids'], \
					rtns['data_batch']['text_ids'], rtns['predictions']
				loss = rtns['loss']

				u_id = rtns['data_batch']['user_id']#bs*1
				week_time = rtns['data_batch']['time_id']
				prob5 = rtns['logits_5']#bs*12*vocab
				prob3 = rtns['logits_3']#bs*12*vocab
				prob = rtns['logits']#bs*12*vocab

				batch_size = len(targets_t)
				targets_ = []
				len_ = args.one_seq_length + 1
				seq_len = args.one_seq_length
				his_num = args.history_num

				for i in range(batch_size):
					t = [1]
					t.extend(targets_t[i][len_ + seq_len * (his_num-1):len_  + seq_len * his_num])
					t.extend([2])
					targets_.append(t)

				for b in range(batch_size):
					s = []
					m = real_templates_[b]
					for t in range(len(m)):
						#print(m[t],mask_id)
						if m[t] == mask_id:
							#print("ok")
							s.append(t)
					#print("blank:", s)
					s_random.append(s)

				loss_lists.append(loss)

				filled_templates = \
					dataset_utils.fill_template(template_pack=rtns['template'],
										   predictions=rtns['predictions'],
										   eoa_id=eoa_id, pad_id=pad_id, eos_id=eos_id)
				
				templates, targets, generateds = _id2word_map(real_templates_.tolist()), \
												 _id2word_map(targets_), \
												 _id2word_map(filled_templates)

				for template, target, generated, u, w, p5, p3, p in zip(templates, targets, generateds, u_id, week_time, prob5, prob3, prob):
					template = template.split('<EOS>')[0].strip().split()
					target = target.split('<EOS>')[0].strip().split()
					got = generated.split('<EOS>')[0].strip().split()
					templates_list.append(template)
					targets_list.append(target)
					hypothesis_list.append(got)
					user_list.append(u)
					week_list.append(w)
					prob5_list.append(p5)#12*k
					prob3_list.append(p3)#12*k
					prob_list.append(p)

				cnt += 1
				if mode is not 'test' and cnt >= 150:
					break
			except tf.errors.OutOfRangeError:
				break

		avg_loss = np.mean(loss_lists)

		#calculate acc
		all_accur = []
		all_dis = []
		blank_accur = []
		blank_dis = []
		bad_num = 0
		flag_ = 0
		recall5 = []
		recall3 = []
		MAP = []
		#count = 0
		#print("bat:",len(hypothesis_list))
		for hyp, tplt, tgt, p5, p3, p in zip(hypothesis_list, templates_list, targets_list, prob5_list, prob3_list, prob_list):
			#print(flag_)
			blank = s_random[flag_]
			#start = time.time()
			blank_acc_, all_acc_, flag, recall5_blank, recall3_blank, mAP = get_acc(hyp, tplt, tgt,blank, p5, p3, p, word2id)
			#print(count, time.time() - start)
			#count += 1
			recall5.extend(recall5_blank)
			recall3.extend(recall3_blank)
			MAP.extend(mAP)
			#f1_score.extend(f1_blank)
			blank_dis_, all_dis_ = get_distance(hyp, tplt, tgt,blank)
			bad_num += flag
			blank_accur.append(blank_acc_)
			blank_dis.append(blank_dis_)
			all_accur.append(all_acc_)
			all_dis.append(all_dis_)
			flag_ += 1

		#print("bad_num:",bad_num)
		#print("before_W:{} before_b:{} after_w:{} after_b{}".format(b_w, b_b, a_w, a_b))
		print('epoch:{} {}_loss:{}  {}_all_acc:{} {}_blank_acc:{} {}_all_dis:{} {}_blank_dis:{}'.
			format(cur_epoch, mode, avg_loss ,mode, np.mean(all_accur),mode, np.mean(blank_accur), mode, np.mean(all_dis), mode, np.mean(blank_dis)))
		print('epoch:{} {}_recall@3:{} {}_recall@5: {} {}_MAP:{}'.format(cur_epoch, mode, np.mean(recall3), mode, np.mean(recall5), mode, np.mean(MAP)))
		if mode == "test":
			if args.save_eval_output:
				result_filename = \
					args.log_dir + 'epoch{}.beam{}.{}.allacc{:.3f}.reacc{:.3f}.all_dis:{:.3f}.blank_dis:{:.3f}' \
						.format(cur_epoch, args.beam_width, mode,np.mean(all_accur),np.mean(blank_accur), np.mean(all_dis),  np.mean(blank_dis))
				with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
					for tmplt, tgt, hyp, u , w in zip(templates_list, targets_list, hypothesis_list, user_list, week_list):
						resultfile.write("- template: " + ' '.join(tmplt) + '\n')
						resultfile.write("- expected: " + ' '.join(tgt) + '\n')
						resultfile.write('- got:      ' + ' '.join(hyp) + '\n')
						resultfile.write('- user:     ' + ' ' + str(u[0]) + '\n')
						resultfile.write('- time_id:  ' + ' ' + str(w[0]) + '\n\n')
		return np.mean(all_accur), np.mean(blank_accur)
		
	eval_saver = tf.train.Saver(max_to_keep=5)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print("global var init")
		sess.run(tf.local_variables_initializer())
		print("local var init")
		sess.run(tf.tables_initializer())
		print("table init")
		
		writer = tf.summary.FileWriter("TensorBoard_test",sess.graph)
		writer.close()
		max_acc = -1
		count = 0
		if args.running_mode == 'train_and_evaluate':
			for epoch in range(args.max_train_epoch):

				if epoch % args.bleu_interval == 0 or epoch == args.max_train_epoch - 1 and epoch != 0:
					
					#all_acc_train, blank_acc_train = _test_epoch(sess, epoch, mode='train')
					all_acc_valid, blank_acc_valid = _test_epoch(sess, epoch, mode = 'valid')
					#all_acc_test, blank_acc_test = _test_epoch(sess, epoch, mode = 'test')
					if blank_acc_valid > max_acc:
						eval_saver.save(sess, args.log_dir + 'my-model-latest.ckpt')
						max_acc = blank_acc_valid
						count = 0
					if blank_acc_valid < max_acc:
						count += 1
					if count == 2:
						opt_vars['learning_rate'] = opt_vars['learning_rate'] * opt_vars['lr_decay_rate']
						if opt_vars['learning_rate'] < 1e-6:
							#opt_vars['learning_rate'] = 5e-5
							break
						count = 0
				losses = _train_epochs(sess, epoch)
				print("train loss:", np.mean(losses))
				
				sys.stdout.flush()
			print("beginning loading model...")
			eval_saver.restore(sess, args.log_dir + 'my-model-latest.ckpt')
			print("loading model sucessfully...")
			print("final testing")
			all_acc, blank_acc = _test_epoch(sess, 0)
		else:
			print("beginning loading model...")
			eval_saver.restore(sess, args.log_dir + 'my-model-latest.ckpt')
			print("loading model sucessfully...")
			all_acc, blank_acc = _test_epoch(sess, 0)
			print('all acc:{} blank_acc:{} '.format(all_acc, blank_acc))

if __name__ == '__main__':
	tf.app.run(main=_main)
