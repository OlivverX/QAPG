import time
import os
import copy
import sys
import gc

import numpy as np
import torch

from nltk.tokenize import sent_tokenize


from configparser import ConfigParser
from utils.my_logger import logger
from utils.read_data import load_qa
from utils.model_predict import run_predict
from generate.template import *
from calc_sim.sent_sim import calculate_sim_origin_target, calculate_sim_origin_sentence
from utils.preprocess import get_sentences_from_contexts, context_preprocess
from calc_sim.answer_sim import is_same_answer

from utils.gen_logic import *
from utils.extract import *


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    np.random.seed(323)
    begin_time = time.time()
    logger.info(f"Start to run the script {__file__}")
    project_name = sys.argv[1].strip()
    config_file = ''
    if project_name == 'boolq':
        config_file = 'config_boolq.ini'
    elif project_name == 'squad2':
        config_file = 'config-squad2.ini'
    elif project_name == 'narrative':
        config_file = 'config-narrative.ini'
    else:
        assert False, "No such project"

    # config_file = 'config-squad2.ini'
    is_bool = False
    if project_name == 'boolq':
        is_boolq = True


    config = ConfigParser()
    config.read(f"config/{config_file}")
    params = config['PARAMETERS']
    # device = torch.device('cuda:0')


    training_set_file = params['TRAINING_FILE_PATH']
    test_set_file = params['TEST_FILE_PATH']
    qa_model_path = params['QA_MODEL_PATH']

    per_logic_num = int(params['PER_LOGIC_NUM'].strip())
    total_logic_num = int(params['TOTAL_LOGIC_NUM'].strip())
    tf_idf_threshold = float(params['TFIDF_THRESHOLD'].strip())
    sim_threshold = float(params['SIM_THRESHOLD'].strip())
    top_n_logic = int(params['TOP_N_LOGIC'].strip())

    attack_mods = params['ATTACK_MOD'].strip().split(',')
    # extra_sent_num2context = int(params['EXTRA_NUM2C'].strip())
    # max_attack_num = int(params['MAX_ATTACK_NUM'].strip())
    # combined_context_num = int(params['CONTEXT_NUM'].strip())
    origin_answer_flag = params['ORIGIN_ANSWER']
    # ------------------------------------------------------------------------------------------------------------------
    # project_name = test_set_file.split("data_")[-1].split('_')[0]
    logger.info(f"project name: {project_name}")
    # if project_name != "boolq" and "TI" in attack_mods:
    #     attack_mods.remove("TI")
    # is_boolq = False
    # if project_name == 'boolq':
    #     is_boolq = True

    res_dir = f'../results/{project_name}/res-dev'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    file_comprehensive_res = os.path.join(res_dir, f"{project_name}_test_cases_all.tsv")
    file_comprehensive_res_bug = os.path.join(res_dir, f"{project_name}_violation_all.tsv")
    # TODO: neglect <no answer> when cal the num of bugs

    # ----------------------------load training set and development set ------------------------------------------------
    qa_list_obj_test = load_qa(test_set_file, end_id=10000000)
    logger.info("origin question number: " + str(len(qa_list_obj_test.qa_list)))
    # qa_list_obj_training = load_qa(training_set_file, end_id=50000, remove_redundancy_context=True)
    # 这一步载入去除了重复的语料库，我之前做的时候好像没注意，需要重新观察tf-idf得分
    

    # with open(f"../datasets/compress/{project_name}.comp.tsv", "r", encoding="utf-8") as f:
    #     compress_question_list = f.readlines()

    seed_case_count = 0
    WQ_seed_count = 0
    WC_seed_count = 0
    IQC_seed_count = 0

    # # add statement
    # statement_file_path = f'/home/ASE21-QAAskeR-main-new/replication/declarative_sentence/{project_name}_ds.npy'
    # statement_total = np.load(statement_file_path, allow_pickle=True)



    bias = 0
    for obj in qa_list_obj_test.qa_list:
        gc.collect()
        logger.info(f"\n\nThe qa id[test-set] is: {obj.id}")

        if int(obj.id)<=1073:
            continue


        if origin_answer_flag == 'SUT':
            ori_ans = run_predict(obj.Q + '\n ' + '('+obj.T+') ' + obj.C)[0]
        elif origin_answer_flag == 'GT':
            ori_ans = obj.A
        else:
            raise Exception(print("wrong key of origin_answer_flag!"))

        logger.info(f"Q is: {obj.Q}")
        tmp_attack_mod = copy.deepcopy(attack_mods)
        this_attack = np.random.choice(tmp_attack_mod)
        

        # if ori_ans == 'no answer>':
        #     continue

        # ----------calculate the similarity between question and each context------------------------------------------
        # -------ignore the qa[in train-set] that have same context with current context.-------------------------------
        # qa_list_obj_training_no_this = copy.deepcopy(qa_list_obj_training)
        # qa_list_obj_training_no_this.remove_qa_of_this_context(obj.C)
        # # 这一步原来是删除训练集中重复语料，我们则添加进语料库以便于计算tf-idf值

        # qa_list_obj_training_this = copy.deepcopy(qa_list_obj_training)
        # new_croups = qa_list_obj_training_this.update_croups(obj.C)

        # question_keywords = extract_question_keywords(obj.Q, obj.C ,is_bool, havecontext=True)
        # logger.info(f"question_keywords list: {question_keywords}")
        
        
        logi_count = 0
        attack_success_flag = False

        
        # continue
        # print(int(statement_total[bias]['index']),obj.id)


        if 'WQ' == this_attack:
            question_keywords = extract_question_keywords(obj.Q, obj.C ,is_bool, havecontext=True)
            logger.info(f"question_keywords list: {question_keywords}")

            
            #TODO：后期用所有question搜索代替[:2]，使用tatal_logic_num控制数量
            for keyword in question_keywords[:3]:
                des_all = []
                # 这里返回的是和c最相关的一个句子
                # TODO：后期加FP过滤在这个地方加,如果FP比例不够高，考虑这个地方不使用并行的替换，而是用两次全范围替换引入迭代的部分
                des = generate_description(keyword, per_logic_num, obj.C)
                if des: 
                    for tmp in des:
                        des_all.append([keyword,tmp])

                if len(des_all)==0:
                    logger.info(f"==> No satisfied Logic!!! <==")
                    continue
                
                structure_score_list = []
                for i in range(len(des_all)):
                    keyword_tmp,des_tmp = des_all[i][0],des_all[i][1]
                    structure_score = get_structure_score(obj.Q,keyword_tmp,des_tmp)
                    if structure_score==None:
                        break
                    else:
                        structure_score_list.append(structure_score)
                if len(structure_score_list)>0:   
                    des_chosen = [des_all[np.argmax(structure_score_list)]]
                    break                    
                 
            if not structure_score_list:
                logger.info("no structure score here")
                continue
            
            logger.info("***************************************************************")
            logger.info(f"The attack_mod is: {this_attack}")
            logger.info(f"The keywords are: {keyword}")
            logger.info(f"The description_lists length is: {len(des_all)}") 
            logger.info(f"The des_chosen_lists are: {des_chosen}") 
            # logger.info(f"The sim scores are: {sim}") 
            logger.info("***************************************************************")

            ctx_add = ''
            new_q = obj.Q
            for i in range(len(des_chosen)):
                keyword,des = des_chosen[i][0],des_chosen[i][1]
                print('keyword,des:',keyword,des)
                new_q = new_q.replace(keyword, des, 1)
                if not ctx_add:
                    ctx_add += des.capitalize() +' is '+ keyword
                else:
                    ctx_add += des +' is '+ keyword

                if i==len(des_chosen)-2:
                    ctx_add += ' and '
                elif i==len(des_chosen)-1:
                    ctx_add += '. '
                else:
                    ctx_add += ', '
            print('ctx_add:',ctx_add)

            new_q=ctx_add+new_q
            new_c=obj.C
            new_ans = run_predict(new_q  + '\n '+ new_c)[0]

            str_output = f"{new_q} \\n {new_c}\t{ori_ans}\t{new_ans}\n"
            with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                f.write(str_output)

            if not is_same_answer(new_ans, ori_ans, is_bool=is_bool):
                attack_success_flag = True
                logger.info(f'.....................A Successful {this_attack} Attack...................................\n')
                logger.info(f"Original Question: {obj.Q}")
                logger.info(f"New Question: {new_q}")
                # logger.info(f"Logic score: {logic_score}")
                # logger.info(f"Origin context: {obj.C}")
                logger.info(f"New context: {new_c}")
                logger.info(f"Ground truth:{obj.A}, New answer:{new_ans}")
                # logger.info(f"The : {logi_count}")
                logger.info('...................................................................................\n')
                str_output_bug = f"{obj.id}\t{obj.Q}\t{new_q} \\n {new_c}\t{ori_ans}->{new_ans}\t{this_attack}\t{max(structure_score_list)}\n"
                with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                    f.write(str_output_bug)


            if len(des_chosen)!=0:
                WQ_seed_count+=1
                seed_case_count+=1
                

        if 'WC' == this_attack:
            sentences = sent_tokenize(obj.C.replace('\n',''))
            w_embedding1 = sbert_model.encode(obj.Q)
            w_embedding2 = sbert_model.encode(sentences)
            most_similar_sentence_id = np.argsort(-np.array(util.dot_score(w_embedding1, w_embedding2).tolist()[0]))[0]

            # TODO：这里先测量和Q的相似度，后期可以测试和ans的效果
            context_keywords = extract_question_keywords(sentences[most_similar_sentence_id], obj.Q ,is_bool, havecontext=True)
            logger.info(f"context_keywords list: {context_keywords}")
            
            for keyword in context_keywords[:3]:
                des_all = []
                # 这里返回的是和c最相关的一个句子
                des = generate_description(keyword, per_logic_num, obj.C)
                if des: 
                    for tmp in des:
                        des_all.append([keyword,tmp])

                if len(des_all)==0:
                    logger.info(f"==> No satisfied Logic!!! <==")
                    continue
                
                structure_score_list = []
                for i in range(len(des_all)):
                    keyword_tmp,des_tmp = des_all[i][0],des_all[i][1]
                    structure_score = get_structure_score(sentences[most_similar_sentence_id],keyword_tmp,des_tmp)
                    if structure_score==None:
                        break
                    else:
                        structure_score_list.append(structure_score)
                if len(structure_score_list)>0:   
                    des_chosen = [des_all[np.argmax(structure_score_list)]]
                    break     

            if not structure_score_list:
                logger.info("no structure score here")
                continue
            
            logger.info("***************************************************************")
            logger.info(f"The attack_mod is: {this_attack}")
            logger.info(f"The keywords are: {keyword}")
            logger.info(f"The description_lists length is: {len(des_all)}") 
            logger.info(f"The description_lists are: {des_chosen}") 
            # logger.info(f"The sim scores are: {sim}") 
            logger.info("***************************************************************")

            ctx_add = ''
            new_q, new_c = obj.Q, obj.C

            for i in range(len(des_chosen)):
                keyword,des = des_chosen[i][0],des_chosen[i][1]
                print('keyword,des:',keyword,des)
                sentences[most_similar_sentence_id] = sentences[most_similar_sentence_id].replace(keyword, des, 1)


                if not ctx_add:
                    ctx_add += des.capitalize() +' is '+ keyword
                else:
                    ctx_add += des +' is '+ keyword

                if i==len(des_chosen)-2:
                    ctx_add += ' and '
                elif i==len(des_chosen)-1:
                    ctx_add += '. '
                else:
                    ctx_add += ', '
            print('ctx_add:',ctx_add)
            new_c = ctx_add + ' '.join(sentences)

            new_ans = run_predict(new_q  + '\n '+ new_c)[0]

            str_output = f"{new_q} \\n {new_c}\t{ori_ans}\t{new_ans}\n"
            with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                f.write(str_output)

            if not is_same_answer(new_ans, ori_ans, is_bool=is_bool):
                attack_success_flag = True
                logger.info(f'.....................A Successful {this_attack} Attack...................................\n')
                logger.info(f"Original Question: {obj.Q}")
                logger.info(f"New Question: {new_q}")
                # logger.info(f"Logic score: {logic_score}")
                # logger.info(f"Origin context: {obj.C}")
                logger.info(f"New context: {new_c}")
                logger.info(f"Ground truth:{obj.A}, New answer:{new_ans}")
                # logger.info(f"The : {logi_count}")
                logger.info('...................................................................................\n')
                str_output_bug = f"{obj.id}\t{obj.Q}\t{new_q} \\n {new_c}\t{ori_ans}->{new_ans}\t{this_attack}\t{max(structure_score_list)}\n"
                with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                    f.write(str_output_bug)


            if len(des_chosen)!=0:
                WC_seed_count+=1
                seed_case_count+=1



        if 'IQC' == this_attack:
            question_keywords = extract_question_keywords(obj.Q, obj.C ,is_bool, havecontext=True)
            logger.info(f"question_keywords list: {question_keywords}")

            
            #TODO：后期用所有question搜索代替[:2]，使用tatal_logic_num控制数量
            for keyword in question_keywords[:3]:
                print('keywprd==>',keyword)
                des_all_q = []
                # 这里返回的是和c最相关的一个句子
                # TODO：后期加FP过滤在这个地方加,如果FP比例不够高，考虑这个地方不使用并行的替换，而是用两次全范围替换引入迭代的部分
                des = generate_description(keyword, per_logic_num, obj.C)
                if des: 
                    for tmp in des:
                        des_all_q.append([keyword,tmp])
                
                print('des_all_q==>',des_all_q)
                if len(des_all_q)==0:
                    logger.info(f"==> No satisfied Logic!!! <==")
                    continue
                
                structure_score_list_q = []
                for i in range(len(des_all_q)):
                    keyword_tmp,des_tmp = des_all_q[i][0],des_all_q[i][1]
                    structure_score = get_structure_score(obj.Q,keyword_tmp,des_tmp)
                    if structure_score==None:
                        break
                    else:
                        structure_score_list_q.append(structure_score)
                print('structure_score_list_q==>',structure_score_list_q)   
                if len(structure_score_list_q)>0:   
                    des_chosen_q = [des_all_q[np.argmax(structure_score_list_q)]]
                    break
                                 
            if not structure_score_list_q:
                logger.info("no structure score here")
                continue


            ctx_add = ''
            new_q, new_c = obj.Q, obj.C

            for i in range(len(des_chosen_q)):
                keyword,des = des_chosen_q[i][0],des_chosen_q[i][1]
                print('keyword,des:',keyword,des)
                new_q = new_q.replace(keyword, des, 1)
                if not ctx_add:
                    ctx_add += des.capitalize() +' is '+ keyword
                else:
                    ctx_add += des +' is '+ keyword

                if i==len(des_chosen_q)-2:
                    ctx_add += ' and '
                elif i==len(des_chosen_q)-1:
                    ctx_add += '. '
                else:
                    ctx_add += ', '
            print('ctx_add:',ctx_add)

            new_q=ctx_add+new_q

            sentences = sent_tokenize(obj.C.replace('\n',''))
            w_embedding1 = sbert_model.encode(obj.Q)
            w_embedding2 = sbert_model.encode(sentences)
            most_similar_sentence_id = np.argsort(-np.array(util.dot_score(w_embedding1, w_embedding2).tolist()[0]))[0]

            # TODO：这里先测量和Q的相似度，后期可以测试和ans的效果
            context_keywords = extract_question_keywords(sentences[most_similar_sentence_id], obj.Q ,is_bool, havecontext=True)
            logger.info(f"context_keywords list: {context_keywords}")
            
            for keyword in context_keywords[:3]:
                des_all_c = []
                # 这里返回的是和c最相关的一个句子
                des = generate_description(keyword, per_logic_num, obj.C)
                if des: 
                    for tmp in des:
                        des_all_c.append([keyword,tmp])

                if len(des_all_c)==0:
                    logger.info(f"==> No satisfied Logic!!! <==")
                    continue
                
                structure_score_list_c = []
                for i in range(len(des_all_c)):
                    keyword_tmp,des_tmp = des_all_c[i][0],des_all_c[i][1]
                    structure_score = get_structure_score(sentences[most_similar_sentence_id],keyword_tmp,des_tmp)
                    if structure_score==None:
                        break
                    else:
                        structure_score_list_c.append(structure_score)
                if len(structure_score_list_c)>0:   
                    des_chosen_c = [des_all_c[np.argmax(structure_score_list_c)]]
                    break

            if not structure_score_list_c:
                logger.info("no structure score here")
                continue            
            logger.info("***************************************************************")
            logger.info(f"The attack_mod is: {this_attack}")
            logger.info(f"The keywords are: {keyword}")
            logger.info(f"The description_lists length is: q-{len(des_all_q)},c-{len(des_all_c)}") 
            logger.info(f"The question description_lists are: {des_chosen_q}") 
            logger.info(f"The context description_lists are: {des_chosen_c}") 
            # logger.info(f"The sim scores are: {sim}") 
            logger.info("***************************************************************")

            ctx_add = ''
            for i in range(len(des_chosen_c)):
                keyword,des = des_chosen_c[i][0],des_chosen_c[i][1]
                print('keyword,des:',keyword,des)
                sentences[most_similar_sentence_id] = sentences[most_similar_sentence_id].replace(keyword, des, 1)


                if not ctx_add:
                    ctx_add += des.capitalize() +' is '+ keyword
                else:
                    ctx_add += des +' is '+ keyword

                if i==len(des_chosen_c)-2:
                    ctx_add += ' and '
                elif i==len(des_chosen_c)-1:
                    ctx_add += '. '
                else:
                    ctx_add += ', '
            print('ctx_add:',ctx_add)
            new_c = ctx_add + ' '.join(sentences)

            new_ans = run_predict(new_q  + '\n '+ new_c)[0]

            str_output = f"{new_q} \\n {new_c}\t{ori_ans}\t{new_ans}\n"
            with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                f.write(str_output)

            if not is_same_answer(new_ans, ori_ans, is_bool=is_bool):
                attack_success_flag = True
                logger.info(f'.....................A Successful {this_attack} Attack...................................\n')
                logger.info(f"Original Question: {obj.Q}")
                logger.info(f"New Question: {new_q}")
                # logger.info(f"Logic score: {logic_score}")
                # logger.info(f"Origin context: {obj.C}")
                logger.info(f"New context: {new_c}")
                logger.info(f"Ground truth:{obj.A}, New answer:{new_ans}")
                # logger.info(f"The : {logi_count}")
                logger.info('...................................................................................\n')
                str_output_bug = f"{obj.id}\t{obj.Q}\t{new_q} \\n {new_c}\t{ori_ans}->{new_ans}\t{this_attack}\t{max(structure_score_list_q)}\t{max(structure_score_list_c)}\n"
                with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                    f.write(str_output_bug)


            if len(des_chosen_q) or len(des_chosen_c)!=0:
                IQC_seed_count+=1
                seed_case_count+=1

    

        logger.info(f"The logic num is: {logi_count} attack_flag is: {attack_success_flag}") 
        logger.info(f"The seed case has already been: {seed_case_count}") 
        logger.info(f"WQ num: {WQ_seed_count}, WC num: {WC_seed_count}, IQC num: {IQC_seed_count}") 

    end_time = time.time()
    logger.info(f"Finish question_gen for question_{obj.id},during time is {round(end_time - begin_time, 2)}s")



    #         description_word = extract_description_by_tree(description_all)
            # if description_word in final_word --> FP


              
            
            
            # if a != 'no answer>':
            #     question_num += 1
            # if a != a_:
            #     violation_num += 1
            # violation_rate = round(violation_num/question_num,2)
            
            # print(index,': question_num:',question_num, 'violation_num:',violation_num,'violation_rate:',violation_rate,'\n')
            
            # logging.info(str(index) + ':\n' + q + '-->'+ str(question_keywords) + '-->'+ final_word + '\n' + description_all + '-->' + description_word + '\n' + a + '-->' + a_ + '\nquestion_num: ' + str(question_num) + ' violation_num: ' + str(violation_num) +' violation_rate: ' + str(violation_rate)+'\n')


