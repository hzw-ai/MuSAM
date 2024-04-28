# -*- coding: utf-8 -*-

import codecs
import math
import os
from operator import itemgetter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config

ss = StandardScaler()


class WhiteningDataCompute:
    """
    降维数据和计算spearman
    """

    def __init__(self, words_data_path, sentence_data_path, dataset_path, dim, suffixStr, standard_scaler=False):
        self.dim = dim
        # ------------数据加载----------
        dataset = codecs.open(dataset_path, encoding="utf-8").readlines()
        self.dataset = [s.strip().split()[:2] for s in dataset]
        # 获得原始数据集的相似度
        sim_value = [s.strip().split()[-1] for s in dataset]
        # 加载多模态向量矩阵，并各个模态分别使用各自的降维方式降维至同一维度
        # 处理词典词向量------------------------------
        self.word_vec_data = np.load(config.simbert_word_sense_ave3+suffixStr, allow_pickle=True).item()
        self.wordsense = codecs.open(words_data_path, encoding="utf-8")
        # 词+词义的词对
        self.wordsense = [s.strip().split(sep="\t") for s in self.wordsense]
        # 完成降维后的所有词+词义向量矩阵
        self.word_fit_vec_data = np.concatenate(list(self.word_vec_data.values()))
        if standard_scaler:
            self.word_fit_vec_data = ss.fit_transform(self.word_fit_vec_data)
        self.word_fit_vec_data = self.use_whitening(self.word_fit_vec_data, dim)
        # 处理上下文词向量-----------------------------
        self.sentence_vec = np.load(config.save_robert_chinese_base_cev_ave_path3+'-8-20-.npy', allow_pickle=True).item()
        self.sentence_sense = codecs.open(sentence_data_path+'-8-20-.npy', encoding="utf-8")
        # 词+句子的词对
        self.sentence_sense = [s.strip().split(sep="\t") for s in self.sentence_sense]
        sentence_vecs = np.concatenate(list(self.sentence_vec.values()))
        # 完成降维后的所有词+句子向量矩阵
        if standard_scaler:
            sentence_vecs = ss.fit_transform(sentence_vecs)
        self.sentence_fit_vec = self.pca_fit_transform(sentence_vecs, dim)
        # 处理图片向量--------------------------------
        all_base_dataset_dict = {}
        all_base_dataset_dict[config.wordsim_240] = 'WS240'
        all_base_dataset_dict[config.wordsim_297] = 'WS297'
        all_base_dataset_dict[config.MC_30] = 'MC30'
        all_base_dataset_dict[config.RG_65] = 'RG65'
        all_base_dataset_dict[config.wordsim_353] = 'WS353'

        self.base_data = all_base_dataset_dict[dataset_path]
        all_pic_key = {}
        all_pic_vec = []
        for abdd_key in all_base_dataset_dict.keys():
            # 加载图片向量三维数组，一维为词下标，二维为词对应的5张图片下标，三维为该图片的向量
            self.picture_vec = np.load(
                config.context_resnet50_using_padding_new_parm + all_base_dataset_dict[abdd_key] + '.npy',
                allow_pickle=True).item()
            picture_vecs = np.concatenate(list(self.picture_vec.values()))
            all_pic_vec.append(picture_vecs)
            all_pic_key[abdd_key] = list(self.picture_vec.keys())
        picture_vecs = np.concatenate(all_pic_vec, axis=0)
        
        self.picture_vec_wh = self.use_whitening(picture_vecs, dim)
        picture_vec_dict = {}

        flag = 0
        for abdd_key in all_base_dataset_dict.keys():
            # 将key与降维后的value对应
            for key in all_pic_key[abdd_key]:
                picture_vec_dict[key] = self.picture_vec_wh[flag]
                flag = flag + 1
        self.picture_vec_dict = picture_vec_dict
        
        # ------------加载原始数据集排序与相似度信息-----------
        self.manual_dict = {}
        for i in range(len(self.dataset)):
            self.manual_dict[(self.dataset[i][0], self.dataset[i][1])] = sim_value[i]

    def use_whitening(self, data, dim):
        W, mu = self.compute_kernel_bias(data, dim)
        res = self.transform_and_normalize(data, kernel=W, bias=mu)
        return res

    def pca_fit_transform(self, vecs, n_components):
        x_demean = (vecs - np.mean(vecs, axis=0))
        sigma = (x_demean.T @ x_demean) / x_demean.shape[0]
        u, s, v = np.linalg.svd(sigma)
        u_reduced = u[:, :n_components]
        z = np.dot(x_demean, u_reduced)
        # 恢复到原有维度
        # x_recover = np.dot(z, u_reduced.T)
        return z

    def compute_spearman(self,
                         method,
                         wordsense_deco=False,
                         sentence_deco=True,
                         tupain_deco=True):

        auto_dict = {}
        word1s = []
        word2s = []
        words = list(self.word_vec_data.keys())
        vec_entity_list = list(self.sentence_vec.keys())
        pic_list = os.listdir(config.pic_path_new2)
        for i, pairwise in enumerate(self.dataset):
            # 获取词对的两个词的所有向量--------------
            # 词典向量
            current_words = self.wordsense[i]
            pairwise_index1 = words.index(current_words[0])
            pairwise_index2 = words.index(current_words[1])
            wordsense_vec_1 = self.word_fit_vec_data[pairwise_index1, :]
            wordsense_vec_2 = self.word_fit_vec_data[pairwise_index2, :]
            # 上下文向量
            current_sentences = self.sentence_sense[i]
            pairwise_index1 = vec_entity_list.index(current_sentences[0])
            pairwise_index2 = vec_entity_list.index(current_sentences[1])
            sentence_sense_vec_1 = self.sentence_fit_vec[pairwise_index1, :]
            sentence_sense_vec_2 = self.sentence_fit_vec[pairwise_index2, :]
            # 图片向量
            picture_vec_1 = self.picture_vec_dict[self.base_data + '-' + pairwise[0] + '-' + pairwise[1] + '-0']
            picture_vec_2 = self.picture_vec_dict[self.base_data + '-' + pairwise[0] + '-' + pairwise[1] + '-1']

            # 根据传入的融合方式融合向量
            if method == 0:
                # 融合（上下文、词典、图片）
               word1 = np.concatenate(
                   (at_one(sentence_sense_vec_1), at_one(wordsense_vec_1), at_one(picture_vec_1)))
               word2 = np.concatenate(
                   (at_one(sentence_sense_vec_2), at_one(wordsense_vec_2), at_one(picture_vec_2)))
                # 词典+图片
#                word1 = np.concatenate((at_one(wordsense_vec_1), at_one(picture_vec_1)))
#                word2 = np.concatenate((at_one(wordsense_vec_2), at_one(picture_vec_2)))
                # 上下文+图片
#                word1 = np.concatenate((at_one(sentence_sense_vec_1), at_one(picture_vec_1)))
#                word2 = np.concatenate((at_one(sentence_sense_vec_2), at_one(picture_vec_2)))
                # 上下文+词典
                # word1 = np.concatenate((at_one(sentence_sense_vec_1), at_one(wordsense_vec_1)))
                # word2 = np.concatenate((at_one(sentence_sense_vec_2), at_one(wordsense_vec_2)))
                # 上下文
#                word1 = sentence_sense_vec_1
#                word2 = sentence_sense_vec_2
                # 词典
#                word1 = wordsense_vec_1
#                word2 = wordsense_vec_2
                # 图片
#                word1 = picture_vec_1
#                word2 = picture_vec_2

            sim = cosine_similarity(word1.reshape(1, -1), word2.reshape(1, -1))
            auto_dict[(self.dataset[i][0], self.dataset[i][1])] = sim
        spear_score = self.spearmans_rho(self.assign_ranks(self.manual_dict), self.assign_ranks(auto_dict))
        return spear_score * 100

    def compute_kernel_bias(self, vecs, n_components):
        """
        计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        # vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W_inv = np.dot(u, np.diag(s ** 0.5))
        W = np.linalg.inv(W_inv.T)
        return W[:, :n_components], -mu

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """
        应用变换，然后标准化
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def assign_ranks(self, item_dict):  # 对一个字典进行排序
        ranked_dict = {}
        sorted_list = [(key, val) for (key, val) in
                       sorted(item_dict.items(), key=itemgetter(1), reverse=True)]  # 逆序排序,并且生成一个list
        for i, (key, val) in enumerate(sorted_list):
            same_val_indices = []
            for j, (key2, val2) in enumerate(sorted_list):
                if val2 == val:
                    same_val_indices.append(j + 1)
            if len(same_val_indices) == 1:
                ranked_dict[key] = i + 1
            else:
                ranked_dict[key] = 1. * sum(same_val_indices) / len(same_val_indices)  # 如果相似度相同，则排序相等，且序号取均值
        return ranked_dict

    def spearmans_rho(self, ranked_dict1, ranked_dict2):  # 输入两个dict
        assert len(ranked_dict1) == len(ranked_dict2)
        if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
            return 0.
        x_avg = 1. * sum([val for val in ranked_dict1.values()]) / len(ranked_dict1)  # 计算均值
        y_avg = 1. * sum([val for val in ranked_dict2.values()]) / len(ranked_dict2)  # 计算均值
        num, d_x, d_y = (0., 0., 0.)
        for key in ranked_dict1.keys():
            xi = ranked_dict1[key]
            yi = ranked_dict2[key]
            num += (xi - x_avg) * (yi - y_avg)  # 计算对应离差的乘积
            d_x += (xi - x_avg) ** 2  # 计算方差
            d_y += (yi - y_avg) ** 2  # 计算方差
        return num / (math.sqrt(d_x * d_y))


def task(dim, method):

#    num_attention_heads = [2, 4, 8, 16, 32, 64] # 上下文
#    num_attention_heads = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] # 字典
#    num_hidden_layers = [4, 8, 12, 16, 20, 24, 28, 32]
    
    num_attention_heads = [6]
    num_hidden_layers = [20]
    
    for nah in num_attention_heads:
        for nhl in num_hidden_layers:
            suffixStr = '-' + str(nah) + '-' + str(nhl) + '-.npy'
            # 词典调参
            class_whitening = WhiteningDataCompute(config.simbert_wordsim240_word_sense_ave_pram + suffixStr,
                                                   config.context_robert_wordsim240_word_sense_ave_pram,
                                                   config.wordsim_240, dim, suffixStr, standard_scaler=False)
            res1 = class_whitening.compute_spearman(method)
            class_whitening = WhiteningDataCompute(config.simbert_wordsim297_word_sense_ave_pram + suffixStr,
                                                   config.context_robert_wordsim297_word_sense_ave_pram,
                                                   config.wordsim_297, dim, suffixStr, standard_scaler=False)
            res2 = class_whitening.compute_spearman(method)
            class_whitening = WhiteningDataCompute(config.simbert_MC30_word_sense_ave_pram + suffixStr,
                                                   config.context_robert_MC30_word_sense_ave_pram,
                                                   config.MC_30, dim, suffixStr, standard_scaler=False)
            res3 = class_whitening.compute_spearman(method)
            class_whitening = WhiteningDataCompute(config.simbert_RG65_word_sense_ave_pram + suffixStr,
                                                   config.context_robert_RG65_word_sense_ave_pram,
                                                   config.RG_65, dim, suffixStr, standard_scaler=False)
            res4 = class_whitening.compute_spearman(method)
            class_whitening = WhiteningDataCompute(config.simbert_wordsim353_word_sense_ave_pram + suffixStr,
                                                   config.context_robert_wordsim353_word_sense_ave_pram,
                                                   config.wordsim_353, dim, suffixStr, standard_scaler=False)
            res5 = class_whitening.compute_spearman(method)
            print(dim, ":", [res1, res2, res3, res4,res5])

            # 上下文调参
#            class_whitening = WhiteningDataCompute(config.simbert_wordsim240_word_sense_ave,
#                                                   config.context_robert_wordsim240_word_sense_ave_pram+suffixStr,
#                                                   config.wordsim_240, dim, suffixStr, standard_scaler=False)
#            res1 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_wordsim297_word_sense_ave,
#                                                   config.context_robert_wordsim297_word_sense_ave_pram+suffixStr,
#                                                   config.wordsim_297, dim, suffixStr, standard_scaler=False)
#            res2 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_MC30_word_sense_ave,
#                                                   config.context_robert_MC30_word_sense_ave_pram+suffixStr,
#                                                   config.MC_30, dim, suffixStr, standard_scaler=False)
#            res3 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_RG65_word_sense_ave,
#                                                   config.context_robert_RG65_word_sense_ave_pram+suffixStr,
#                                                   config.RG_65, dim, suffixStr, standard_scaler=False)
#            res4 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_wordsim353_word_sense_ave,
#                                                   config.context_robert_wordsim353_word_sense_ave_pram+suffixStr,
#                                                   config.wordsim_353, dim, suffixStr, standard_scaler=False)
#            res5 = class_whitening.compute_spearman(method)
#            print(dim, ":", [res1, res2, res3, res4, res5])



#            class_whitening = WhiteningDataCompute(config.simbert_wordsim240_word_sense_ave_pram + suffixStr,
#                                                   config.context_robert_wordsim240_word_sense_ave_pram + suffixStr,
#                                                   config.wordsim_240, dim, suffixStr, standard_scaler=False)
#            res1 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_wordsim297_word_sense_ave_pram + suffixStr,
#                                                   config.context_robert_wordsim297_word_sense_ave_pram + suffixStr,
#                                                   config.wordsim_297, dim, suffixStr, standard_scaler=False)
#            res2 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_MC30_word_sense_ave_pram + suffixStr,
#                                                   config.context_robert_MC30_word_sense_ave_pram + suffixStr,
#                                                   config.MC_30, dim, suffixStr, standard_scaler=False)
#            res3 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_RG65_word_sense_ave_pram + suffixStr,
#                                                   config.context_robert_RG65_word_sense_ave_pram + suffixStr,
#                                                   config.RG_65, dim, suffixStr, standard_scaler=False)
#            res4 = class_whitening.compute_spearman(method)
#            class_whitening = WhiteningDataCompute(config.simbert_wordsim353_word_sense_ave_pram + suffixStr,
#                                                   config.context_robert_wordsim353_word_sense_ave_pram + suffixStr,
#                                                   config.wordsim_353, dim, suffixStr, standard_scaler=False)
#            res5 = class_whitening.compute_spearman(method)
#            print(dim, ":", [res1, res2, res3, res4, res5])



# 向量归一化
def at_one(x):
    vec_x = x / np.linalg.norm(x)
    # vec_x = torch.nn.functional.normalize(torch.from_numpy(x).unsqueeze(0), p=2, dim=1).squeeze(0)
    return vec_x


if __name__ == '__main__':
    print('融合（图片、词典、上下文）')
    task(120, 0)
