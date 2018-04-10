# -*- coding:utf-8 -*- 
from __future__ import division
import numpy as np 
import os
import cPickle as Pickle
import pynlpir 
from math import log
from numpy import linalg as LA




iter_r = 0.685
iter_a = 0.9
tmax = 1


def F(rhoM, proDict): # 目标函数

	# print rhoM

	res = 0
	for pm in  proDict:
		print proDict[pm]
		P = np.trace(np.dot(proDict[pm][1], rhoM))
		# print pm
		# print rhoM
		# print proDict[pm][0]
		# print P
		# print proDict[pm][1]
		# print '\n'
		res += proDict[pm][0] * log(P)

	# print res
	# exit()
	return res

def Grad_F(rhoM, proDict): # 目标函数的导数

	# dim = proList[0].shape[0]
	dim = 2
	res = np.zeros((dim, dim))
	for pm in  proDict:
		P = np.trace(np.dot(proDict[pm][1], rhoM))
		# print proDict[pm][0]
		# print proDict[pm][0]
		# print P
		# print proDict[pm][1]
		res += (proDict[pm][0] * proDict[pm][1] / P)
	# print res
	# print '\n'

	# exit()
	return res


def a_q(t, rhoM, proDict): # q(t)

	FRF = np.dot(np.dot(Grad_F(rhoM, proDict), rhoM), Grad_F(rhoM, proDict))
	# print 'FRF is'
	# print FRF
	res = (1 + 2 * t + t * t *np.trace(FRF))
	# print 't is'
	# print t
	# print 'q(t) is'
	# print res
	# exit()
	return res


def Maxt_rho(rhoM, proDict): #

	FRF = np.dot(np.dot(Grad_F(rhoM, proDict), rhoM), Grad_F(rhoM, proDict))
	# print 'FRF is'
	# print FRF
	# print 'trace is'
	# print np.trace(FRF)
	# print res
	# exit()
	res = (FRF / np.trace(FRF))

	return res


def Maxt_D(rhoM, proDict): #公式（18）

	res = Maxt_rho(rhoM, proDict) - rhoM


	# for pm in  proDict:
	# 	print proDict[pm][0]
	# 	print proDict[pm][1]
	
	# print 'rho is'
	# print rhoM
	# print res
	# print Maxt_rho(rhoM, proDict)
	# exit()
	return res


def Mean_rho(rhoM, proDict): #公式（15）

	FR = np.dot(Grad_F(rhoM, proDict), rhoM)
	RF = np.dot(rhoM, Grad_F(rhoM, proDict))
	res = (FR +RF) / 2
	# print 'rho is'
	# print rhoM
	# print 'FR is'
	# print FR
	# print 'RF is'
	# print RF
	# print res
	# exit()
	return res

def Mean_D(rhoM, proDict): #公式（17）
	res = Mean_rho(rhoM, proDict) - rhoM
	# print 'rho is'
	# print rhoM
	# print res
	# exit()
	return res


def D(t, rhoM, proDict): # 公式（19）
	FRF = np.dot(np.dot(Grad_F(rhoM, proDict), rhoM), Grad_F(rhoM, proDict))
	temp1 = (2 * Mean_D(rhoM, proDict) / a_q(t, rhoM, proDict))
	temp2 = (t * np.trace(FRF) * Maxt_D(rhoM, proDict) / a_q(t, rhoM, proDict))
	res = temp2 + temp1

	# for pm in  proDict:
	# 	print proDict[pm][0]
	# 	print proDict[pm][1]

	# print 't is'
	# print t
	# print 'q(t) is'
	# print a_q(t, rhoM, proDict)
	# print 'FRF is'
	# print FRF
	# print 'max_D is'
	# ttt = Maxt_D(rhoM, proDict)
	# print ttt
	# print "temp1 is"
	# print temp1
	# print 'temp2 is'
	# print temp2
	# print 'res is'
	# print res
	# exit()
	return res


def set_t(t):
	return max(1, t)


def judgement(rhoM, proDict): #算法终止条件
	FRF = np.dot(np.dot(Grad_F(rhoM, proDict), rhoM), Grad_F(rhoM, proDict))
	FR = np.dot(Grad_F(rhoM, proDict), rhoM)
	if LA.norm(rhoM- FRF)< 0.00000001:
		if LA.norm(rhoM -FR)< 0.00000001:
			return False
		else:
			return True
	else:
		return True




def judge_t(t, d, rhoM, proDict):
	# print 'please see here:'
	temp1 = F(rhoM + t * d, proDict)
	temp2 = F(rhoM, proDict)
	temp3 = iter_r * t * np.trace(np.dot(Grad_F(rhoM, proDict), d))
	temp = temp1 - temp2 - temp3

	# exit()
	
	if temp <= 0:
		return True
	else:
		return False


def bound_t(t):
	res = 1
	for i in range(1000):
		res *= 0.9

	return res

	

def build_dict(s, s_dict, voc_dict):
	for word in s:
		word_pro = s_dict.get(word, None)
		if word_pro is None:
			vec = voc_dict[word]
			mat = np.outer(vec, vec) / np.inner(vec, vec)
			s_dict[word] = []
			s_dict[word] = [1.0, mat]
			# print s_dict[word][1]
			# s_dict[word][0] = 1.0
			# s_dict[word][1] = mat
		else:
			s_dict[word][0] += 1.0

	for word in s_dict:
		s_dict[word][0] /= len(s)





def Judge(t, rhoM, proDict): #公式（10）
	# print rhoM
	F(rhoM, proDict)

	while judgement(rhoM, proDict):
		t = set_t(t)
		flag = False

		iter_D = D(t, rhoM, proDict)
		print iter_D
		# print np.trace(np.dot(Grad_F(rhoM, proDict), iter_D))
		while judge_t(t, iter_D, rhoM, proDict):
			# print np.trace(rhoM)
			ttt = np.trace(np.dot(Grad_F(rhoM, proDict), iter_D))
			# print np.trace(np.dot(Grad_F(rhoM, proDict), iter_D))
			if ttt <= 0:
				exit()
			t *= iter_a
			# print t
			# if t < bound_t(t):
			# 	flag = True
			# 	break
			iter_D = D(t, rhoM, proDict)
		if flag:
			break
		rhoM += t * iter_D




def scoring(qrhoM, arhoM): #计算相似度得分
	q_eigvals, q_eigvec = np.linalg.eig(qrhoM)
	a_eigvals, a_eigvec = np.linalg.eig(arhoM)
	res = 0
	

	for q_val in q_eigvals:
		# print q_val
		for q_vec, a_val, a_vec in zip(q_eigvec, a_eigvals, a_eigvec):
			# print a_val
			# if(a_val <= 0 or res < 0):
			# 	res = -10000
			# 	break
			# else:
			res += (q_val * log(a_val * np.inner(q_vec, a_vec) * np.inner(q_vec, a_vec)))
				# res = 1


	return res


# def iteration():
# 	# train_fname = 'train//test.txt'
# 	# train_fname = 'train//nlpcc-iccpol-2016.dbqa.training-data'
# 	train_fname = 'train//test'


# 	pynlpir.open()
	

# 	simf = open('sim.txt','a')
# 	rhomf = open('rhoM', 'a')
# 	voc_name = 'voc//ndim_50.vocab'

# 	voc_dict = Pickle.load(open(voc_name))
# 	print 'beginning'
# 	line_count = 0

# 	with open(train_fname) as f:
# 		lines = f.readlines()
# 		print 'lines number', len(lines)
# 		for line in lines:
# 			line_count += 1
# 			if line_count % 1000 == 0:
# 				print line_count, 'have been proced'
# 			print line_count, 'have been proced'
# 			line = line.split('	')
# 			q_list_pre = pynlpir.segment(line[0], pos_tagging = False)
# 			a_list_pre = pynlpir.segment(line[1], pos_tagging = False)
# 			a_dict = {}
# 			q_dict = {}
# 			build_dict(q_list_pre, q_dict, voc_dict)
# 			build_dict(a_list_pre, a_dict, voc_dict)

# 			dim_init = 50
# 			# dim_init = len(q_dict[q_dict.keys()[0]])
# 			randomnum = np.random.random(dim_init)
# 			diagmat = np.diag(randomnum)
# 			arhoM = diagmat / np.trace(diagmat)
# 			# arhoM=np.eye(dim_init,dim_init)/ dim_init

# 			qrhoM = diagmat / np.trace(diagmat)
# 			# qrhoM=np.eye(dim_init,dim_init)/ dim_init

# 			t = tmax
# 			Judge(t, qrhoM, q_dict)

# 			t = tmax
# 			Judge(t, arhoM, a_dict)
# 			score = scoring(qrhoM, arhoM)

# 			simf.write(str(score))
# 			simf.write('\n')


# 		simf.close()
# 		rhomf.close()
# 		pynlpir.close()


if __name__ == '__main__':
	# buid_voc()
	# iteration()
	word1 = np.array([1, 0,0])
	word2 = np.array([0, 1,0])
	word3 = np.array([0, 0,1])


	Word1 = np.outer(word1, word1) / np.inner(word1, word1)
	Word2 = np.outer(word2, word2) / np.inner(word2, word2)
	word3 = np.outer(word3, word3) / np.inner(word3, word3)

	test_dict = {'we': [3/5, Word1], 'are': [2/5, Word2],"chinese":[0.1,word3]} #输入格式

	# dim_init = 50
	dim_init = word1.shape[0]
	# # dim_init = len(q_dict[q_dict.keys()[0]])
	# randomnum = np.random.random([dim_init])
	# randomnum = np.random.random(dim_init)
	diagmat = np.diag([1/3, 2/3,0]) 
	arhoM = diagmat / np.trace(diagmat) # 初始密度矩阵
	# arhoM=np.eye(dim_init,dim_init)/ dim_init

	qrhoM = diagmat / np.trace(diagmat)
	qrhoM=np.eye(dim_init,dim_init)/ dim_init
	# print qrhoM[0][0]
	# print '\n'

	t = tmax
	Judge(t, qrhoM, test_dict)