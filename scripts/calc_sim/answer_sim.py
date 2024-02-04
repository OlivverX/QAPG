from sentence_transformers import SentenceTransformer
import torch
from torch import nn
import numpy as np

model_path = 'whaleloops/phrase-bert'
phrase_bert_model = SentenceTransformer(model_path)
device = torch.device('cuda:0')
phrase_bert_model.to(device)

cos_sim = nn.CosineSimilarity(dim=0)


def same_boolq_answer(ans1, ans2):
    if 'no' not in [ans1, ans2] and 'yes' not in [ans1, ans2]:
        return False
    if 'no' in [ans1, ans2]:
        return ans1 == ans2
    elif 'yes' in [ans1, ans2]:
        if 'no' in [ans1, ans2]:
            return False
        elif ans1 == ans2:
            return True
        else:
            sim = pbert_ans_similarity(ans1, ans2)
            if sim >= 0.76:
                return True
            else:
                return False


def pbert_ans_similarity(ans1, ans2):
    phrase_embs = phrase_bert_model.encode([ans1, ans2])
    [p1, p2] = phrase_embs
    similarity = float(cos_sim(torch.tensor(p1), torch.tensor(p2)))
    # print(f'The cosine similarity between phrase {ans1} and {ans2} is: {similarity}')
    return similarity

def pbert_keyword_similarity(key_words, textrank_words):
    textrank_words_embs = np.mean(phrase_bert_model.encode(textrank_words),axis=0)
    # print(textrank_words_embs)
    key_word_embs = phrase_bert_model.encode(key_words)
    # print(key_word_embs)
    similarity_list = []
    for word in key_word_embs:
        similarity_list.append(float(cos_sim(torch.tensor(word), torch.tensor(textrank_words_embs))))
    return [key_words[i] for i in np.argsort(similarity_list)[::-1]]



def is_same_answer(ans1, ans2, is_bool=False):
    ans1 = ans1.strip().lower()
    ans2 = ans2.strip().lower()
    if is_bool:
        return same_boolq_answer(ans1, ans2)
    similarity = pbert_ans_similarity(ans1, ans2)
    if similarity >= 0.76:
        return True
    else:
        return False


if __name__ == '__main__':
    res = is_same_answer("Jack", "jack wisen")
    print(res)
    res = is_same_answer("yes", "No", is_bool=True)
    print(res)
    res = is_same_answer("No answer", "no", is_bool=True)
    print(res)
    res = pbert_keyword_similarity(['sheep','china'],['chinese','canada'])
    print('--->',res)




