import transformers
import torch
import numpy as np
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from utils.my_logger import logger


from utils.extract import extract_description_by_tree
# from utils.my_logger import logger

sbert_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')  # ../3rd_models/multi-qa-MiniLM-L6-cos-v1

device = torch.device('cuda:0')
sbert_model = sbert_model.to(device)

generator = transformers.pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', device='cuda:0')

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# generator.to(device)

# def generate_description(final_word):
#     prompt = 'It is acknowledged that ' + str(final_word) + ' is the'
#     tmp = generator(prompt, do_sample=True, min_length=50, temperature = 0.7)
#     description_all = tmp[0]['generated_text']
#     return description_all.replace('‘',"'")

stop_words = [',']


def generate_description(final_word, per_logic_num, context):
    prompt = 'It is acknowledged that '+ str(final_word) + ' is the'
    tmp = generator(prompt, do_sample=True, max_new_tokens=100, num_return_sequences=per_logic_num, temperature = 0.7, pad_token_id=generator.tokenizer.eos_token_id)
    torch.cuda.empty_cache()
    # w_embedding2 = sbert_model.encode(context)


    des_final_list = []
    for des in tmp:
        des_tmp = des['generated_text'].strip()
        # des_tmp = des['generated_text'].strip()
        description_word = extract_description_by_tree(des_tmp)
        # w_embedding1 = sbert_model.encode(description_word)
        # sim = util.dot_score(w_embedding1, w_embedding2).tolist()[0][0]
        # if sim>sim_threshod:
        
        stop_flag = False
        for stop_word in stop_words:
            if stop_word in description_word:
                stop_flag = True
                logger.info(f"->Failed Logic: {description_word}")
                break
        if not stop_flag:
            des_final = description_word.replace('‘',"'")
            if len(des_final.split())>3:
                des_final_list.append(des_final)
            # print(des_final)
        # logging.info(description_word + '\tscore:' + str(round(sim,3)))
    return des_final_list
    if des_final_list:
        w_embedding1 = sbert_model.encode(context)
        w_embedding2 = sbert_model.encode(des_final_list)
        # print(w_embedding1.shape, w_embedding2.shape)
        sim = np.array(util.dot_score(w_embedding1, w_embedding2).tolist()[0])
    # print(sim)
    # print(np.argsort(-np.array(sim)))

        des_final_list = [des_final_list[i] for i in np.argsort(-sim)]
    return des_final_list if des_final_list else []# 解决大模型输出中文字符影响分词的问题

def process_logi(c, final_word, description_word):
    c_ = description_word.title() +' is '+ final_word +'. '+ c
    return c_

def process_logi_in_sentence(sentences, most_similar_sentence_id, final_word, description_word):
    
    c_ = 'It is acknowledged that ' + description_word +' is '+ final_word +'. '
    sentences[most_similar_sentence_id] = sentences[most_similar_sentence_id].replace(final_word, description_word, 1)
    # sentences[most_similar_sentence_id] = c_
    return c_ + ' '.join(sentences)

# def process_logi2questiont(q, final_word, description_word):
#     c_ = 'It is acknowledged that ' + description_word +' is '+ final_word +'. '+ q
#     return c_