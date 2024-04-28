MC_30 = "base_data/MC-30.txt"
RG_65 = "base_data/RG-65.txt"
wordsim_240 = "base_data/wordsim_240.txt"
wordsim_297 = "base_data/wordsim_297.txt"
wordsim_353 = "base_data/wordsim_353.txt"
#.......................................保存词对对应的最近上下文路径...............................................#
MC_30_sense_best_path = r"input_data/best_sense/MC_30_sense.txt"
RG_65_sense_best_path = r"input_data/best_sense/RG_65_sense.txt"
wordsim_240_sense_best_path = r"input_data/best_sense/wordsim_240_sense.txt"
wordsim_297_sense_best_path = r"input_data/best_sense/wordsim_297_sense.txt"
wordsim_353_sense_best_path = r"input_data/best_sense/wordsim_353_sense.txt"
#.....................上下文.....................#
context_MC_30_sense_best_path = r"input_data/best_sense/context_MC_30_sense.txt"
context_RG_65_sense_best_path = r"input_data/best_sense/context_RG_65_sense.txt"
context_wordsim_240_sense_best_path = r"input_data/best_sense/context_wordsim_240_sense.txt"
context_wordsim_297_sense_best_path = r"input_data/best_sense/context_wordsim_297_sense.txt"
context_wordsim_353_sense_best_path = r"input_data/best_sense/context_wordsim_353_sense.txt"
# 。。。。。。。。。。。。词典数据。。。。。。。。。。。。。#
entity_path = "input_data/best_sense/entity.txt"
merge_path = r"input_data/best_sense/merge.txt"
#............................图片数据.............................#
pic_path_new2 = r"input_data/all_image"
#............................基于resnet50提取的图片向量.............................#
context_resnet50_using_padding_new_parm = r"input_data/picture_vec/context_resnet50_using_padding_new_"
# 。。。。。。。。。。。..................................。。。基于词典的词向量。。。.............。。。。。。。。。。#
roberta_vec_path1 = r"input_data/best_sense/roberta_word2vec1.npy"  # 添加短语到模型token后，输入句子，获取该个短语token的向量
roberta_vec_path2 = r"input_data/best_sense/roberta_word2vec2.npy"  # 添加短语到模型token后，输入句子，获取cls的向量，作为句子向量
roberta_vec_path3 = r"input_data/best_sense/roberta_word2vec3.npy"  # 计算短语中各个字的向量，输入句子，取均值，作为该个短语的词向量
roberta_vec_path4 = r"input_data/best_sense/roberta_word2vec4.npy"  # 不添加短语到模型token，输入句子，获取cls的向量，作为句子向量

# 使用simbert获取实体向量
simbert_base_chinese_word2vec_path1 = r"input_data/best_sense/simbert_base_chinese_word2vec1.npy"  # 计算短语中各个字的向量，输入句子，取均值，作为该个短语的词向量
simbert_base_chinese_word2vec_path2 = r"input_data/best_sense/simbert_base_chinese_word2vec2.npy"  # 不添加短语到模型token，输入句子，获取cls的向量，作为句子向量

# 使用simbert获取实体向量(使用遍历的方式，此时每个数据集对应着两个word sense数据集)
simbert_MC30_word_sense1_ave = r"input_data/simbert_use_new_method_data/MC_30_word_sense1_ave.npy"
simbert_MC30_word_sense1_cls = r"input_data/simbert_use_new_method_data/MC_30_word_sense1_cls.npy"
simbert_MC30_word_sense2_ave = r"input_data/simbert_use_new_method_data/MC_30_word_sense2_ave.npy"
simbert_MC30_word_sense2_cls = r"input_data/simbert_use_new_method_data/MC_30_word_sense2_cls.npy"

# 最优的词+词义样本
simbert_MC30_word_sense_ave = r"input_data/simbert_use_new_method_data/MC_30_word_sense_ave.npy"
simbert_MC30_word_sense_cls = r"input_data/simbert_use_new_method_data/MC_30_word_sense_cls.npy"

simbert_RG65_word_sense1_ave = r"input_data/simbert_use_new_method_data/RG65_word_sense1_ave.npy"
simbert_RG65_word_sense1_cls = r"input_data/simbert_use_new_method_data/RG65_word_sense1_cls.npy"
simbert_RG65_word_sense2_ave = r"input_data/simbert_use_new_method_data/RG65_word_sense2_ave.npy"
simbert_RG65_word_sense2_cls = r"input_data/simbert_use_new_method_data/RG65_word_sense2_cls.npy"

# 最优的词+词义样本
simbert_RG65_word_sense_ave = r"input_data/simbert_use_new_method_data/RG65_word_sense_ave.npy"
simbert_RG65_word_sense_cls = r"input_data/simbert_use_new_method_data/RG65_word_sense_cls.npy"

simbert_wordsim240_word_sense1_ave = r"input_data/simbert_use_new_method_data/wordsim240_word_sense1_ave.npy"
simbert_wordsim240_word_sense1_cls = r"input_data/simbert_use_new_method_data/wordsim240_word_sense1_cls.npy"
simbert_wordsim240_word_sense2_ave = r"input_data/simbert_use_new_method_data/wordsim240_word_sense2_ave.npy"
simbert_wordsim240_word_sense2_cls = r"input_data/simbert_use_new_method_data/wordsim240_word_sense2_cls.npy"

# 最优的词+词义样本
simbert_wordsim240_word_sense_ave = r"input_data/simbert_use_new_method_data/wordsim240_word_sense_ave.npy"
simbert_wordsim240_word_sense_cls = r"input_data/simbert_use_new_method_data/wordsim240_word_sense_cls.npy"

simbert_wordsim297_word_sense1_ave = r"input_data/simbert_use_new_method_data/wordsim297_word_sense1_ave.npy"
simbert_wordsim297_word_sense1_cls = r"input_data/simbert_use_new_method_data/wordsim297_word_sense1_cls.npy"
simbert_wordsim297_word_sense2_ave = r"input_data/simbert_use_new_method_data/wordsim297_word_sense2_ave.npy"
simbert_wordsim297_word_sense2_cls = r"input_data/simbert_use_new_method_data/wordsim297_word_sense2_cls.npy"

# 最优的词+词义样本
simbert_wordsim297_word_sense_ave = r"input_data/simbert_use_new_method_data/wordsim297_word_sense_ave.npy"
simbert_wordsim297_word_sense_cls = r"input_data/simbert_use_new_method_data/wordsim297_word_sense_cls.npy"

# 最优的词+词义样本
simbert_wordsim353_word_sense_ave = r"input_data/simbert_use_new_method_data/wordsim353_word_sense_ave.npy"
simbert_wordsim353_word_sense_cls = r"input_data/simbert_use_new_method_data/wordsim353_word_sense_cls.npy"

# 没有ws353数据集的
simbert_word_sense_ave = r"input_data/simbert_use_new_method_data/word_sense_ave.npy"
simbert_word_sense_cls = r"input_data/simbert_use_new_method_data/word_sense_cls.npy"
# 加了ws353数据集的
simbert_word_sense_ave2 = r"input_data/simbert_use_new_method_data/word_sense_ave2.npy"
simbert_word_sense_cls2 = r"input_data/simbert_use_new_method_data/word_sense_cls2.npy"

# 最优的词+词义样本（调参）
simbert_MC30_word_sense_ave_pram = r"input_data/simbert_use_new_method_data/MC_30_word_sense_ave_pram"
simbert_RG65_word_sense_ave_pram = r"input_data/simbert_use_new_method_data/RG65_word_sense_ave_pram"
simbert_wordsim240_word_sense_ave_pram = r"input_data/simbert_use_new_method_data/wordsim240_word_sense_ave_pram"
simbert_wordsim297_word_sense_ave_pram = r"input_data/simbert_use_new_method_data/wordsim297_word_sense_ave_pram"
simbert_wordsim353_word_sense_ave_pram = r"input_data/simbert_use_new_method_data/wordsim353_word_sense_ave_pram"
simbert_word_sense_ave3 = r"input_data/simbert_use_new_method_data/word_sense_ave3"

# 。。。。。。。。。。。。。。基于上下文的词向量。。。。。。。。。。。。。。。。。。。。#
roberta_vec_path5 = r"input_data/best_sense/roberta_word2vec1_content.npy"
save_roberta_wwm_extra_large_vec_path1 = r"input_data/best_sense/roberta_wwm_extra_large_word2vec1_content.npy" # 不添加短语到模型token, 计算短语中各个字的向量，输入句子，取均值，作为该个短语的词向量
save_roberta_wwm_extra_large_vec_path2 = r"input_data/best_sense/roberta_wwm_extra_large_word2vec2_content.npy" # 不添加短语到模型token，输入句子，获取cls的向量，作为句子向量
save_simbert_chinese_base_cev_path1 = r"input_data/best_sense/simbert_chinese_base_word2vec1_content.npy"  # ave
save_simbert_chinese_base_cev_path2 = r"input_data/best_sense/simbert_chinese_base_word2vec2_content.npy"  # cls


context_robert_MC30_word_sense_ave = r"input_data/robert_use_new_method_data/MC_30_word_sense_ave.npy"
context_robert_MC30_word_sense_cls = r"input_data/robert_use_new_method_data/MC_30_word_sense_cls.npy"

context_robert_RG65_word_sense_ave = r"input_data/robert_use_new_method_data/RG65_word_sense_ave.npy"
context_robert_RG65_word_sense_cls = r"input_data/robert_use_new_method_data/RG65_word_sense_cls.npy"

context_robert_wordsim240_word_sense_ave = r"input_data/robert_use_new_method_data/wordsim240_word_sense_ave.npy"
context_robert_wordsim240_word_sense_cls = r"input_data/robert_use_new_method_data/wordsim240_word_sense_cls.npy"

context_robert_wordsim297_word_sense_ave = r"input_data/robert_use_new_method_data/wordsim297_word_sense_ave.npy"
context_robert_wordsim297_word_sense_cls = r"input_data/robert_use_new_method_data/wordsim297_word_sense_cls.npy"

context_robert_wordsim353_word_sense_ave = r"input_data/robert_use_new_method_data/wordsim353_word_sense_ave.npy"
context_robert_wordsim353_word_sense_cls = r"input_data/robert_use_new_method_data/wordsim353_word_sense_cls.npy"

# 没有ws353数据集的
save_robert_chinese_base_cev_ave_path = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_ave_content.npy"  # ave
save_robert_chinese_base_cev_cls_path = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_cls_content.npy"  # cls
# 加了ws353数据集的
save_robert_chinese_base_cev_ave_path2 = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_ave_content2.npy"  # ave
save_robert_chinese_base_cev_cls_path2 = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_cls_content2.npy"  # cls

save_robert_chinese_base_cev_ave_cme_path2 = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_ave_cme_content2.npy"  # ave
save_robert_chinese_base_cev_ave_cmm_path2 = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_ave_cmm_content2.npy"  # ave

save_robert_chinese_base_cev_ave_path3 = r"input_data/robert_use_new_method_data/robert_chinese_base_word2vec_ave_content3"
# 最优的词+词义样本（调参）
context_robert_MC30_word_sense_ave_pram = r"input_data/robert_use_new_method_data/MC_30_word_sense_ave_parm"
context_robert_RG65_word_sense_ave_pram = r"input_data/robert_use_new_method_data/RG65_word_sense_ave_parm"
context_robert_wordsim240_word_sense_ave_pram = r"input_data/robert_use_new_method_data/wordsim240_word_sense_ave_parm"
context_robert_wordsim297_word_sense_ave_pram = r"input_data/robert_use_new_method_data/wordsim297_word_sense_ave_parm"
context_robert_wordsim353_word_sense_ave_pram = r"input_data/robert_use_new_method_data/wordsim353_word_sense_ave_parm"

# 。。。。。。。。。。。。。。fusion vector。。。。。。。。。。。。。。。。。。。。#
save_fusion_path1 = r"../vec_fusion/fusion_vec.npy"
save_fusion_path2 = r"../vec_fusion/fusion_vec_bilstm_last_hidden_state.npy"

#............roberta_wwm_extra_large.........................#
roberta_wwm_extra_large_config_path = r"../deberta_sim/roberta_wwm_extra_large/config .json"
roberta_wwm_extra_large_model_path = r"../deberta_sim/roberta_wwm_extra_large/pytorch_model.bin"
roberta_wwm_extra_large_vcab_path = r"../deberta_sim/roberta_wwm_extra_large/vocab.txt"

#............................................simbert_chinese_bese.........................................#
simbert_config_path = r"../deberta_sim/simbert_model/config.json"
simbert_vocab_path = r"../deberta_sim/simbert_model/vocab.txt"
simbert_model_path = r"../deberta_sim/simbert_model/pytorch_model.bin"

#  ...............tokenizer args...................
max_len = 100

# ..................................................read_data..............................................#
batch_size = 1

# ................................................lstm_model args..........................................#
embedding_dim = 768