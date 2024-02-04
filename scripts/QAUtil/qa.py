from utils.my_logger import logger
from utils.preprocess import word_embedding

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize


import numpy as np
import copy


class QA:
    count = 0

    def __init__(self, q, a, c, t=''):
        self.id = QA.count  # identify the line number of the qa
        QA.count += 1
        self.Q = q
        self.A = a
        # self.C = context_preprocess(c)
        self.C = c
        self.T = t
        self.C_Embedding = None
        self.PA = None
        # self.Q2S = q2s


class QAList:
    def __init__(self):
        self.qa_list = []
        self.ctx_list = []
        QA.count = 0

    def append_(self, qa):
        self.qa_list.append(qa)
        self.ctx_list.append(qa.C)

    def remove_qa_of_this_context(self, context):
        for qa in self.qa_list:
            if context == qa.C:
                self.qa_list[qa.id].C = ''
                self.qa_list[qa.id].C_Embedding = np.zeros(shape=(384,), dtype='float32')
                logger.debug(f"line {qa.id+1} in training-set has the same context with current context:{context} ")

    def set_all_context_embedding(self):
        contexts_all = []
        for qa in self.qa_list:
            contexts_all.append(qa.C)
        context_all_embedding = word_embedding(contexts_all)
        for id in range(len(self.qa_list)):
            self.qa_list[id].C_Embedding = context_all_embedding[id]

    def get_all_context_embedding(self):
        context_embedding_list = []
        for qa in self.qa_list:
            context_embedding_list.append((qa.C, qa.C_Embedding))
        return np.array(context_embedding_list)  # return np.array()

    def update_croups(self, context):
        new_croups = copy.deepcopy(self.ctx_list)
        new_croups.append(context)
        sents = new_croups
        sents=[word_tokenize(sent) for sent in sents] #对每个句子进行分词
        # print(sents)  #输出分词后的结果
        corpus=TextCollection(sents)  #构建语料库
        return corpus

