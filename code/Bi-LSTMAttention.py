
# coding = utf-8
# Copyright 2019.7.01. Jiangchenglin done 
# In[1]:

import gzip
import jieba
import re
import json
import tarfile
import configparser
import pickle
import os
import csv
import time
import datetime
import random
import json
import math
import warnings
from collections import Counter
from math import sqrt
from tensorflow.python.platform import gfile
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

tf.app.flags.DEFINE_boolean("inference", False, "Set to True for inference")
FLAGS = tf.app.flags.FLAGS
warnings.filterwarnings("ignore")
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
_DIGIT_RE2 = re.compile(r"\d")
# In[2]:

# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 135
    learningRate = 0.001
    
class ModelConfig(object):
    embeddingSize = 50
    
    hiddenSizes = [256, 128]  # LSTM结构的神经元个数
    
    dropoutKeepProb = 0.8
    l2RegLambda = 0.05
    
class Config(object):
    sequenceLength = 35  # 取了所有序列长度的均值
    batchSize = 32
    
    dataSource = "../data/preProcess/ecm_test_data_with_label.csv"
    
    stopWordSource = "../data/english"
    
    numClasses = 6
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()

    
# 实例化配置参数对象
config = Config()


# In[3]:

# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource  
        
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        
        self._stopWordDict = {}
        
        self.trainReviews = []
        self.trainLabels = []
        
        self.evalReviews = []
        self.evalLabels = []
        
        self.wordEmbedding =None
        
        self._wordToIndex = {}
        self._indexToWord = {}
        
    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """
        
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength
        
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)
            
        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["_UNK"]

        return reviewVec

    def sentence_to_token_ids(self, sentence, vocabulary, sequenceLength, tokenizer=None, normalize_digits=True):

        # reviewVec = np.zeros((sequenceLength))
        # sequenceLen = sequenceLength
        #
        # # 判断当前的序列是否小于定义的固定序列长度
        # if len(sentence) < sequenceLength:
        #     sequenceLen = len(sentence)
        # print('1: ', sentence)
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.basic_tokenizer(sentence)
        if not normalize_digits:
            seqlist = [vocabulary.get(w, vocabulary["_UNK"]) for w in words]
        else:

            # Normalize digits by 0 before looking words up in the vocabulary.
            seqlist =  [vocabulary.get(_DIGIT_RE2.sub(r"0", w), vocabulary["_UNK"]) for w in words]
        sequenceLen = sequenceLength
        # print('2: ', seqlist)

        # 判断当前的序列是否小于定义的固定序列长度
        if len(seqlist) < sequenceLength:
            sequenceLen = len(seqlist)
        for i in range(config.sequenceLength-sequenceLen):
            seqlist.append(int(0))


        # print('3: ' , seqlist)
        return seqlist

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        
        reviews = []
        labels = []
        
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self.sentence_to_token_ids(x[i], self._wordToIndex, self._sequenceLength)
            reviews.append(reviewVec)
            
            labels.append(y[i])
            
        trainIndex = int(len(x) * rate)
        
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    # def basic_tokenizer(self, sentence):
    #     """Very basic tokenizer: split the sentence into a list of tokens."""
    #     words = []
    #     senten = ' '.join(sentence)#空格很重要，记得因为原数据里面整体不是字符串
    #     # print('====it is =====')
    #     # print(senten)
    #     for space_separated_fragment in senten.strip().split():
    #         words.extend(_WORD_SPLIT.split(space_separated_fragment))
    #     # print(words[0])
    #     # print(type(words[0]))
    #     # print([w for w in words if w])
    #     return [w for w in words if w]

    #加入自己处理数据的代码即可

   

    def load_word_vector(self, fname):
        dic = {}
        with open(fname) as f:
            data = f.readlines()
            for line in data:
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                dic[word] = vector
        return dic

    def load_vocab(self, fname):
        vocab = []
        with open(fname) as f:
            data = f.readlines()
            for d in data:
                vocab.append(d[:-1])
        return vocab

    def random_init(self, dim):
        return 2 * math.sqrt(3) * (np.random.rand(dim) - 0.5) / math.sqrt(dim)

    def refine_wordvec(self, rvector, vocab, dim=50):
        wordvec = []
        count = 0
        found = 0
        for word in vocab:
            count += 1
            if word in rvector:
                found += 1
                aa = np.array(list(map(float, rvector[word].split())))
                # print('aa.shape:', aa.shape)
                wordvec.append(aa)
            else:
                bb = np.array(self.random_init(dim))
                wordvec.append(bb)
                # print('bb.shape:', bb.shape)
        # print('Total words: %d, Found words: %d, Overlap: %f' % (count, found, float(found)/count))
        return np.array(wordvec)

    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        
        # allWords = [word for review in reviews for word in review]
        #
        # # 去掉停用词
        # subWords = [word for word in allWords if word not in self.stopWordDict]
        #
        # wordCount = Counter(subWords)  # 统计词频
        # sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        #
        # # 去除低频词
        # words = [item[0] for item in sortWordCount if item[1] >= 5]
        init_dicpath = '../data/wordJson/word2id'
        vector_path = '../data/wordJson/vector.txt'
        self.create_vocabulary(init_dicpath, reviews, 40000)
        
        # vocab, wordEmbedding = self._getWordEmbedding(words)
        # self.wordEmbedding = wordEmbedding
        #
        # self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        # self._indexToWord = dict(zip(list(range(len(vocab))), vocab))
        self._wordToIndex, self._indexToWord = self.initialize_vocabulary(init_dicpath)


        self.loadVec(init_dicpath, vector_path)


        
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)
        
        with open("../data/wordJson/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)
            
    # def _getWordEmbedding(self, words):
    #     """
    #     按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    #     """
    #
    #     wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)
    #     vocab = []
    #     wordEmbedding = []
    #
    #     # 添加 "pad" 和 "UNK",
    #     vocab.append("pad")
    #     vocab.append("UNK")
    #     wordEmbedding.append(np.zeros(self._embeddingSize))
    #     wordEmbedding.append(np.zeros(self._embeddingSize))
    #
    #     for word in words:
    #         try:
    #             vector = wordVec.wv[word]
    #             vocab.append(word)
    #             wordEmbedding.append(vector)
    #         except:
    #             print(word + "不存在于词向量中")
    #
    #     return vocab, np.array(wordEmbedding)
    
    # def _readStopWord(self, stopWordPath):
    #     """
    #     读取停用词
    #     """
    #
    #     with open(stopWordPath, "r") as f:
    #         stopWords = f.read()
    #         stopWordList = stopWords.splitlines()
    #         # 将停用词用列表的形式生成，之后查找停用词时会比较快
    #         self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
    def loadVec(self, init_dicpath, vector_path):
        print('loading word vector...')
        raw_vector = self.load_word_vector(vector_path)
        print('loading vocabulary...')
        vocab_post = self.load_vocab(init_dicpath)
        print('refine word vector...')
        wordEmbedding = self.refine_wordvec(raw_vector, vocab_post)
        self.wordEmbedding = wordEmbedding

    def labels2onehot(self, labels, class_num=None, class_labels=None):
        """
        生成句子的情感标记。调用时class_num与class_labels必选其一。
        :param labels: list; 数据的标记列表
        :param class_num: int; 类别总数
        :param class_labels: list; 类别标记，如[0, 1]、['a', 'b']
        :return: numpy array.
        """
        tik = []
        if class_num is None and class_labels is None:
            raise Exception("Parameter eithor class_num or class_labels must be given!  -- by lic")
        if class_labels is not None:
            class_num = len(class_labels)

        def label2onehot(label_):
            if class_labels is None:
                label_index = label_
            else:
                label_index = class_labels.index(label_)

            print(int(label_index))
            print(type(int(label_index)))
            onehot_label = [0] * class_num
            onehot_label[int(label_index)] = 1
            print('======-=====')
            print(onehot_label)
            return onehot_label
        for label_ in labels:
            tik.append(label2onehot(label_))

        return tik
            
    def dataGen(self):
        """
        初始化训练集和验证集
        """
        
        # 初始化停用词
        # self._readStopWord(self._stopWordSource)
        
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        
        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)
        labels = self.labels2onehot(labels, class_num = config.numClasses)
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
        
if not FLAGS.inference:
    data = Dataset(config)
    data.dataGen()
else:
    init_dicpath = '../data/wordJson/word2id'
    vector_path = '../data/wordJson/vector.txt'
    data = Dataset(config)
    data.loadVec(init_dicpath, vector_path)

# In[4]:

# print("train data shape: {}".format(data.trainReviews.shape))
# print("train label shape: {}".format(data.trainLabels.shape))
# print("eval data shape: {}".format(data.evalReviews.shape))


# In[5]:

# 输出batch数据集

def nextBatch(x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
    
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        
        numBatches = len(x) // batchSize

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = np.array(x[start: end], dtype="int64")
            batchY = np.array(y[start: end], dtype="float32")
            
            yield batchX, batchY
def getinferBatch(sentence):
    init_dicpath= '../data/wordJson/word2id'
    word2id, _ = data.initialize_vocabulary(init_dicpath)
    if sentence == '':
        return None

    # First step: Divide the sentence in token
    # tokens = nltk.word_tokenize(sentence) # English
    tokens = [w for w in jieba.cut(sentence)]  # Chinese
    # print (tokens)
    if len(tokens) > config.sequenceLength:
        return None

    newseq = data.sentence_to_token_ids(tokens, word2id, config.sequenceLength)
    # Second step: Convert the token in word ids
    # wordIds = []
    # for token in tokens:
    #     wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences
    #
    # # Third step: creating the batch (add padding, reverse)
    # batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output
    batch = [newseq]

    return batch


# In[6]:

# 构建模型
class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, inference, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, config.numClasses], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            
        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize),
                                                                 output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize),
                                                                 output_keep_prob=self.dropoutKeepProb)


                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))
        
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)
                
        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)
        
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]
            print('输出的维度是： ')
            print(H)

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = config.model.hiddenSizes[-1]

        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
           
        if not inference:
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            self.predictions = tf.argmax(self.logits, -1, name="predictions")
            # 计算二元交叉熵损失
            with tf.name_scope("loss"):
            
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

        else:
            print('进入输出模式：')
            print(output)
            self.logits = tf.nn.xw_plus_b([output], outputW, outputB, name="logits")
            self.predictions = tf.argmax(self.logits, -1, name="predictions")


    
    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        print('attention')
        print(output)
        print(type(output))
        
        return output
    def set_accuracy(self):
        """
        准确率
        """
        with tf.name_scope("accuracy_scope"):
            correct_pred = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.inputY, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[7]:

# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)
    
    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)


# In[ ]:

# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
inference = FLAGS.inference
# 定义计算图
with tf.Graph().as_default():

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  

    sess = tf.Session(config=session_conf)
    
    # 定义会话
    with sess.as_default():
        #构建模型过程
        lstm = BiLSTMAttention(inference, config, wordEmbedding)
        
        globalStep = tf.Variable(0, name="globalStep", trainable=False)

        if not inference:
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(lstm.loss)
            # 将梯度应用到变量下，生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

            lstm.set_accuracy()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        #加载模型
        model_path = '../model/my-model'
        # 用summary绘制tensorBoard
        if not FLAGS.inference and not gfile.Exists(model_path):

            print("Created model with fresh parameters.")
            gradSummaries = []
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(outDir))

            lossSummary = tf.summary.scalar("loss", lstm.loss)
            summaryOp = tf.summary.merge_all()

            trainSummaryDir = os.path.join(outDir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

            # 初始化所有变量


            # 保存模型的一种方式，保存为pb文件
            #         builder = tf.saved_model.builder.SavedModelBuilder("../model/Bi-LSTM/savedModel")
            sess.run(tf.global_variables_initializer())
        else:
            ckpt = tf.train.get_checkpoint_state('../model')
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            feed_dict = {
              lstm.inputX: batchX,
              lstm.inputY: batchY,
              lstm.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, accr, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, lstm.loss, lstm.accuracy, lstm.logits, lstm.predictions],
                feed_dict)
            # print("========原始计算数值：")
            # print("Step: ", step)
            # print("predictions: ", predictions)
            # print(" binaryPreds: ", binaryPreds)
            timeStr = datetime.datetime.now().isoformat()
            # acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}".format(timeStr, step, loss, accr))
            trainSummaryWriter.add_summary(summary, step)

        def inferStep(batchX):
            print('进入infer')
            feed_dict = {
                lstm.inputX: batchX,
                lstm.dropoutKeepProb: 1.0
            }
            predictions, binaryPreds = sess.run(
                [lstm.logits, lstm.predictions],
                feed_dict)
            return predictions, binaryPreds
        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
              lstm.inputX: batchX,
              lstm.inputY: batchY,
              lstm.dropoutKeepProb: 1.0
            }
            summary, step, loss, accr, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, lstm.loss,lstm.accuracy, lstm.logits, lstm.predictions],
                feed_dict)
            
            # acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            
            evalSummaryWriter.add_summary(summary, step)
            
            return loss, accr

        
        if not inference:
            for i in range(config.training.epoches):
                # 训练模型
                print("start training model")
                for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                    # print("============= 数据样式： ")
                    # print("=》 句子： ", batchTrain[0])
                    # print("=》 标签： ", batchTrain[1])
                    trainStep(batchTrain[0], batchTrain[1])

                    currentStep = tf.train.global_step(sess, globalStep)
                    if currentStep % config.training.evaluateEvery == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []

                        for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                            loss, acc = devStep(batchEval[0], batchEval[1])
                            losses.append(loss)
                            accs.append(acc)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}, step: {}, loss: {}, acc: {}".format(time_str, currentStep, mean(losses),
                                                                       mean(accs)))

                    if currentStep % config.training.checkpointEvery == 0:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, "../model/my-model", global_step=currentStep)
                        print("Saved model checkpoint to {}\n".format(path))
        else:
            l = '加载分词工具'
            print('\n')
            getinferBatch(l)
            print('\n')
            print('分词工具加载完毕')
            while True:
                # id = user_input.split('$')[0]


                sentence = input('请输入你想要预测情感的语句： ')
                if sentence == '' or sentence == 'exit':
                    break

                questionSeq = []  # Will be contain the question as seen by the encoder
                # batch = sentence2enco(sentence, word2id, model.en_de_seq_len)
                # output =  model.step(sess, batch.encoderSeqs, batch.decoderSeqs, batch.targetSeqs,
                #
                #                                     batch.weights, goToken)
                batchX = getinferBatch(sentence)
                # print('batch是', batchX)
                _, type = inferStep(batchX)
                # answer = self.singlePredict(question, questionSeq)
                if not type:
                    print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                    continue  # Back to the beginning, try again

                # finalAwer = ''.join(type)
                emotionType = ['null(other)', 'like', 'Sadness', 'Disgust', 'Anger', 'Happiness']
                print('{}{}'.format('可能的情感是 ：', emotionType[type[0]]))

                # if self.args.verbose:
                #     print(self.textData.batchSeq2str(questionSeq, clean=True, reverse=True))
                #     print(self.textData.sequence2str(answer))

                print()
#         inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.inputX),
#                   "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutKeepProb)}

#         outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(lstm.binaryPreds)}

#         prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
#                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#         legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
#         builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
#                                             signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

#         builder.save()

