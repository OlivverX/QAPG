import nltk
import stanza
import copy
import pandas as pd
import numpy as np
import spacy
import re

from textrank4zh import TextRank4Keyword, TextRank4Sentence
from utils.my_logger import logger
from nltk.stem import WordNetLemmatizer

from calc_sim.answer_sim import is_same_answer, pbert_keyword_similarity

wnl = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
nlp2 = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def get_wrongword(question):
    doc = nlp(question)
    hasROOT = False
    for token in doc:
        if token.dep_ == 'ROOT':
            if token.text in ['is','are','was','were']:
                local_be = token
                hasROOT = True
                continue
            else:
                break
        if hasROOT and token.tag_ in ['NN','NNS'] and token.head == local_be:
            wrong_word = token.text
            return wrong_word
    return '1234'

def get_structure_score(sentence, keyword, des):
    new_s = sentence.replace(keyword,des,1)
    doc1 = nlp(sentence)
    doc2 = nlp(new_s)
    
    word_phrase = keyword.split()
    phrase_length = len(word_phrase)+1 if  "\'s" in keyword else len(word_phrase)
    des_length = len(des.split())
    found = False

    for i in range(len(doc1) - phrase_length + 1):
        if doc1[i:i+phrase_length].text == ' '.join(word_phrase):

#             print(f"Word phrase found at position {i}")
            begin_idx=i
            found = True
            break
    if not found:
        print('NOTFOUND:',doc1,word_phrase)
        return None
    sen_dic = {}
    for idx,i in enumerate(doc1):
        if idx in range(begin_idx,begin_idx+phrase_length):
            continue
        sen_dic[i.text] = [i.dep_, i.head.text]
#     print(sen_dic)
    
    des_dic = {}
    for idx,i in enumerate(doc2):
        if idx in range(begin_idx,begin_idx+des_length):
            continue
        des_dic[i.text] = [i.dep_, i.head.text]
#     print(des_dic)
    
    wrong_place = 0
    for key,value in sen_dic.items():
        if key not in des_dic:
            print('wrong!!!',f'{doc1}\n{doc2}\n{sen_dic}\n{des_dic}')
            continue
        if des_dic[key][0]!=value[0] or des_dic[key][1]!=value[1]:
            wrong_place+=1
    final_score = 1-wrong_place/len(sen_dic)
    # print(f'''length in dic is {len(sen_dic)} vs {len(des_dic)}, wrong_place num is {wrong_place}, socre = {final_score}''')
    return round(final_score,2)

# def get_correct_structure()


def get_textrank(context):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=context, lower=True, window=2) # 一个句子中相邻的window个单词，两两之间认为有边，注意如果window小于2，会自动设为2
    output = []
    for item in tr4w.get_keywords(num=10, word_min_len=1):  # 取长度大于等于1的，重要性最高的五个关键词
        if item.word != None:
            output.append(wnl.lemmatize(item.word,'n'))
    return output
    

stop_words = ['you','these','what','who','where','which','whom','one','how','why','it']
stop_words+=['he','me','his','she','her','its','your','my','their']
def extract_question_keywords(question, context, is_bool, havecontext=True):    
    keywords_list = []
    doc = nlp(question)
    for np in doc.noun_chunks:
        keywords_list.append(np.text)
        for word in np.text.split():
            if word.lower() in stop_words:
                keywords_list.pop()
                print('poped! ==>',np.text)
                break
        

    # if is_bool: wrong_code = '---'
    # else : wrong_code = get_wrongword(question)
    # for words in keywords_list:
    #     if ("'s"  == output[-1][-2:]) or (words in output[-1]):
    #         s = output.pop()
            # print('wrongword:',s)

    # print('textrank:', get_textrank(context))
    textrank_word = get_textrank(context)
    output_unique = pbert_keyword_similarity(keywords_list, textrank_word)
    

        # tmp = [w for w in unique_word.split()]
        # # print('lemmatized keywords:',tmp)
        # for tmp_word in tmp:
        #     if is_same_answer(textrank_word,tmp_word):
        #         output_final.append(unique_word)
        #         break
    print(keywords_list,'==>',output_unique)
    return output_unique

    
    #             max_tf_idf = score
    #         score_dic[output_unique[i]] = max_tf_idf

    # question_keywords_in_order = dict(sorted(score_dic.items(), key=lambda x: x[1], reverse=True))

    # question_keywords_in_order_list = []
    # for key,value in question_keywords_in_order.items():
    #     print(value, tf_idf_threshold)
    #     if value > tf_idf_threshold:
    #         question_keywords_in_order_list.append(key)

    # question_keywords_in_order_list = [key for key in question_keywords_in_order.keys()]
    # print(question_keywords_in_order_list)
    

def extract_description_by_tree(description_all):
    
    def traverse_tree(tree):
        if tree.label()=='NP':
            keywords_list.append(tree.leaves())
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                traverse_tree(subtree)
                
    keywords_list = []
    doc = nlp2(description_all)  
    for sentence in doc.sentences:
        tree_nltk = nltk.Tree.fromstring(str(sentence.constituency.children))
        traverse_tree(tree_nltk)
        
#         print(keywords_list)
    output = []
    for words in keywords_list:
        word_str = ''
        for word in words:
            word_str += word+' '
        output.append(word_str.strip())

    if len(output)<3:
        return 'this logic has failed'
    return output[2]

