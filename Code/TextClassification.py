'''
KNN heavily borrowed from: https://github.com/weiyujian/knn-classification
'''
import jieba
import _pickle as cPickle
from nltk.util import ngrams as ingrams
from gensim import corpora, models, similarities

import my_setting.utils as utils


class KNN():
    def __init__(self):
        print()

    def get_seg_list(self, query):
        seg_list = jieba.cut(query)
        return seg_list

    def prepare_data(self, file_name):
        fw = open(utils.cfg.get('PROCESSED_DATA', 'knn_train_path'), "w")
        with open(file_name) as f:
            for line in f:
                tmp_list = line.strip().split('\t')
                if len(tmp_list) != 2: continue
                query = tmp_list[1]
                label = tmp_list[0]
                query_seg_list = self.get_seg_list(query)
                query_seg = ' '.join(query_seg_list).encode('utf-8')
                label_seg_list = self.get_seg_list(label)
                label_seg = ' '.join(label_seg_list).encode('utf-8')
                fw.write(str(query) + "\t" + str(query_seg) + "\t" + str(label) + "\t" + str(label_seg) + "\n")
        fw.close()

    def get_ngram(self, text, max_len, stop_set):
        word_list = []
        for inx in range(1, max_len + 1, 1):
            gram_list = self.generate_ngram(text, stop_set, inx)
            word_list += [word for word in gram_list if word not in stop_set]
        word_list += self.generate_skipgram(text, stop_set)
        return " ".join(word_list)

    def generate_ngram(self, text, stop_set, ngram_len):
        words = text.split(" ")
        ngram_list = []
        for wlist in ingrams(words, ngram_len, pad_right=True):
            if wlist[0] is None: continue
            skip = False
            w_inx = 0
            while w_inx < ngram_len:
                if wlist[w_inx] is None or wlist[w_inx] in stop_set:
                    skip = True
                    break
                w_inx += 1
            if skip: continue
            ngram_list.append("_".join(wlist))
        return ngram_list

    def generate_skipgram(self, text, stop_set, ngram_len=16):
        words = text.split(" ")
        pair_set = set()
        for inx in range(len(words)):
            if words[inx] in stop_set: continue
            for iny in range(1, ngram_len, 1):
                if inx + iny >= len(words): break
                if words[inx + iny] in stop_set: continue
                pair_set.add(words[inx] + "|" + words[inx + iny])
        return list(pair_set)

    def read_label(self, file_name):
        f = open(file_name)
        label_map = {}
        seg_map = {}
        data_list = []
        check = set([])
        for line in f:
            tmp_list = line.strip().split("\t")
            if len(tmp_list) != 4: continue
            query_raw = tmp_list[0]
            query_seg = tmp_list[1]
            label_raw = tmp_list[2]
            label_seg = tmp_list[3]
            seg_map[query_raw] = query_seg
            seg_map[label_raw] = label_seg
            label_map[query_raw] = label_raw
            label_map[label_raw] = label_raw
            if label_raw not in check:
                check.add(label_raw)
                data_list.append((label_raw, label_seg))
            if query_raw not in check:
                check.add(query_raw)
                data_list.append((query_raw, query_seg))
        f.close()
        return label_map, seg_map, data_list

    def read_test(self, file_name):
        f = open(file_name)
        label_map = {}
        seg_map = {}
        data_list = []
        for line in f:
            tmp_list = line.strip().split("\t")
            if len(tmp_list) != 4: continue
            query_raw = tmp_list[0]
            query_seg = tmp_list[1]
            label_raw = tmp_list[2]
            label_seg = tmp_list[3]
            seg_map[query_raw] = query_seg
            label_map[query_raw] = label_raw
            data_list.append((query_raw, query_seg))
        f.close()
        return label_map, seg_map, data_list

    def generate_model(self, data_list, stop_set=set([])):
        texts = []
        for inx in range(len(data_list)):
            sent, seg = data_list[inx]
            text = self.get_ngram(seg, 2, stop_set)  # bi_gram和skip_gram增强上下文信息
            texts.append(text.split(" "))
        dictionary = corpora.Dictionary(texts)  # 获取texts中所有词，作为词表，每个词会分配一个id，形成词袋模型
        # Dictionary.keys()：word-id  Dictionary.values()：word
        ret_list = []
        for text in texts:
            freq_text = dictionary.doc2bow(text)  # 把所有语料转化为bag of words，计算text中每个词对应的id以及出现的频率，返回稀疏矩阵（id,freq）
            ret_list.append(freq_text)
        tfidf = models.TfidfModel(ret_list)  # 使用tf-idf模型得出该数据集的tf-idf 模型
        vect_list = tfidf[ret_list]  # 此处已经计算得出所有样本的tf-idf值，即得到了每个样本的(id,tf-idf)
        index = similarities.SparseMatrixSimilarity(vect_list, num_features=len(dictionary.keys()),
                                                    num_best=10)  # 把所有样本做成索引，相似度查询，返回topn最相近的文档
        return tfidf, index, dictionary

    def get_top(self, index, tfidf, dictionary, train_list, seg, stop_set=set([])):
        text = self.get_ngram(seg, 2, stop_set)
        freq_text = dictionary.doc2bow(text.split(" "))  # 把测试语料转为词袋，得到[(id,freq),(id,freq)]
        vect = tfidf[freq_text]  # 直接使用之前得出的tf-idf 模型即可得出该条测试语料的tf-idf 值,得到[(id,tf-idf),(id,tf-idf)]
        sims = index[vect]  # 利用索引计算每一条训练样本和该测试样本之间的相似度，返回top n最相近的训练样本
        top_set = {}
        for inx, score in sims:
            if score <= 0: continue
            sim_text = train_list[inx][0]
            if sim_text == seg: continue
            top_set[sim_text] = score
        return top_set

    def eval_top(self, train_label_map, top_set):
        label_score = {}
        for query, score in top_set.items():
            pred = train_label_map.get(query, "")
            if pred == "": continue
            if pred not in label_score:
                label_score[pred] = 0
            label_score[pred] += score
        max_score = -1.0
        max_label = ""
        for label in label_score:
            if label_score[label] > max_score:
                max_score = label_score[label]
                max_label = label
        return max_label, max_score

    def train(self, train_file, model_version, stop_set):
        train_label_map, _, train_list = self.read_label(train_file)
        tfidf, index, dictionary = self.generate_model(train_list, stop_set)
        cPickle.dump([tfidf, index, dictionary, train_label_map, train_list], open(model_version, "wb"))
        return

    def predict_batch(self, test_file, model_version, stop_set):
        fw = open(test_file + ".out", "w")
        test_label_map, _, test_list = self.read_test(test_file)
        tfidf, index, dictionary, train_label_map, train_list = cPickle.load(open(model_version, "rb"))
        total_cnt = len(test_list)
        acc_cnt = 0
        for test_raw, test_seg in test_list:
            top_set = self.get_top(index, tfidf, dictionary, train_list, test_seg, stop_set)
            max_label, max_score = self.eval_top(train_label_map, top_set)
            fw.write(test_raw + "\t" + test_label_map[test_raw] + "\t" + max_label + "\t" + str(max_score) + "\n")
            if max_label == test_label_map[test_raw]:
                acc_cnt += 1
        acc = 1.0 * acc_cnt / total_cnt
        print("acc_cnt:", acc_cnt)
        print("total_cnt:", total_cnt)
        print("acc:", acc)
        fw.write("acc:" + "\t" + str(acc) + "\t" + "total_cnt:" + "\t" + str(total_cnt) + "\n")
        fw.close()
        return


if __name__ == '__main__':
    stop_set = set([])
    worker = KNN()

    worker.prepare_data(utils.cfg.get('PROCESSED_DATA', 'knn_hashtag_weibo_content_path'))
    worker.train(utils.cfg.get('PROCESSED_DATA', 'knn_train_path'), utils.cfg.get('COMMON_MODEL', 'knn_model_path'),
                 stop_set)
    worker.prepare_data(utils.cfg.get('PROCESSED_DATA', 'knn_test_path'),
                        utils.cfg.get('COMMON_MODEL', 'knn_model_path'),
                        stop_set)
