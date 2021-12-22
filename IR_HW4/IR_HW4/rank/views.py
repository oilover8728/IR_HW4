from django.shortcuts import render, redirect

import os
import re

# /rank folder location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import copy

# Count Sentence / words 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

back_space = "@$^*(<+=\{\\/'\"[-'_|"
front_space = ":~!#%^*)>+=\}\\,./?;'\"\]-_|"

# nltk.download('wordnet')

from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np   


""" 讀取檔案 """
def read_data(filename):
    data = pd.read_csv(filename)
    return data

""" 計算餘弦相似度 """
def cosine_similarity(vec1, vec2):
    """
    :param vec1: 向量 a 
    :param vec2: 向量 b
    :return: sim
    """
    vec1 = np.mat(vec1)
    vec2 = np.mat(vec2)
    num = float(vec1 * vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom
    # 將sim 從[-1:1] 變成 [0:1] 
    sim = 0.5 + 0.5 * cos
    return sim

def most_similar(w2v_model,user_input,avg_vec):
    vec_1=0
    i=0
    for word in word_tokenize(user_input):
        if(w2v_model.wv.__contains__(word)):
            i+=1
            vec_1+=w2v_model.wv[word]
    if(i!=0):
        vec_1 = vec_1/i

    similarity=[]
    for vec_2 in avg_vec:
        similarity.append(cosine_similarity(vec_1,vec_2))
    top10=[]
    top10_simi=[]
    for i in range(10):
        max_number = max(similarity)
        top10_simi.append(max_number)
        index = similarity.index(max_number)
        top10.append(index)
        similarity[index]=-1
    return top10,top10_simi
    
def default(request):
    sentences_list=[]
    sentences_dict={}
    sentences_words_list=[]
    nltk.download('punkt')
    PATH = BASE_DIR + '/rank/Hepatitis/'
    articles=[]
    result = []
    table = read_data(BASE_DIR + '/rank/arti_tfidf.csv')
    sent_table = read_data(BASE_DIR + '/rank/sent2_tfidf.csv')
    model = Word2Vec.load(BASE_DIR + "/rank/word2vec.model")

    index=1
    for i in range(1,11):
        file_path = PATH + str(i) + ".csv"
        train = read_data(file_path)
    #     train = train.dropna(subset=["abstract"])
        for title,context in zip(train['title'],train['abstract']):
            article = []

            article.append(index)
            article.append(title)
            article.append(context)
            article.append(round(table["newsum"][index-1],2))

            articles.append(article)
            for sentence in sent_tokenize(context):
                sentences_dict[sentence] = index
                sentences_list.append(sentence)
                sentence_split_temp=[]
                for word in word_tokenize(sentence):
                    sentence_split_temp.append(word)
                sentences_words_list.append(sentence_split_temp)

            index+=1

    sentence_average_vec=[]
    for i,sentence in enumerate(sentences_words_list):
        sentence_vec = 0
        for word in sentence:
            if(word in table[i:i+1]):
                sentence_vec+=np.multiply(model.wv[word],float(sent_table[i:i+1][word]))
            else:
                sentence_vec+=model.wv[word]
        sentence_average_vec.append(sentence_vec/len(sentence))


    if 'search_token' in request.GET:
        target = request.GET['search_token']

        top10,top10_simi = (most_similar(model,target,sentence_average_vec))
        print(top10)
        for num,score in zip(top10,top10_simi):
            sent = sentences_list[num]
            # 因為dict是存真的index但顯示是+1過後的，所以要-1
            sent2arti = sentences_dict[sent]-1
            articles_temp = copy.deepcopy(articles)
            
            mark_article=''
            for sentence in sent_tokenize(articles_temp[sent2arti][2]):
                if sent == sentence:
                    mark_article += '<span style="background:yellow; color:black;">' + sentence + '</span> '
                else:
                    mark_article += sentence + ' '
            articles_temp[sent2arti].append(round(score,3))
            articles_temp[sent2arti][2] = mark_article 

            result.append(articles_temp[sent2arti])
    else:
        result = articles
        
    return render(request, 'html/default.html', locals())

def default_no_tfidf(request):
    sentences_list=[]
    sentences_dict={}
    sentences_words_list=[]
    nltk.download('punkt')
    PATH = BASE_DIR + '/rank/Hepatitis/'
    articles=[]
    result = []
    table = read_data(BASE_DIR + '/rank/arti_tfidf.csv')
    sent_table = read_data(BASE_DIR + '/rank/sent2_tfidf.csv')
    model = Word2Vec.load(BASE_DIR + "/rank/word2vec.model")

    index=1
    for i in range(1,11):
        file_path = PATH + str(i) + ".csv"
        train = read_data(file_path)
    #     train = train.dropna(subset=["abstract"])
        for title,context in zip(train['title'],train['abstract']):
            article = []

            article.append(index)
            article.append(title)
            article.append(context)
            article.append(round(table["newsum"][index-1],2))

            articles.append(article)
            for sentence in sent_tokenize(context):
                sentences_dict[sentence] = index
                sentences_list.append(sentence)
                sentence_split_temp=[]
                for word in word_tokenize(sentence):
                    sentence_split_temp.append(word)
                sentences_words_list.append(sentence_split_temp)

            index+=1

    sentence_average_vec=[]
    for i,sentence in enumerate(sentences_words_list):
        sentence_vec = 0
        for word in sentence:
            sentence_vec+=model.wv[word]
        sentence_average_vec.append(sentence_vec/len(sentence))


    if 'search_token' in request.GET:
        target = request.GET['search_token']

        top10,top10_simi = (most_similar(model,target,sentence_average_vec))
        print(top10)
        for num,score in zip(top10,top10_simi):
            sent = sentences_list[num]
            # 因為dict是存真的index但顯示是+1過後的，所以要-1
            sent2arti = sentences_dict[sent]-1
            articles_temp = copy.deepcopy(articles)
            
            mark_article=''
            for sentence in sent_tokenize(articles_temp[sent2arti][2]):
                if sent == sentence:
                    mark_article += '<span style="background:yellow; color:black;">' + sentence + '</span> '
                else:
                    mark_article += sentence + ' '
            articles_temp[sent2arti].append(round(score,3))
            articles_temp[sent2arti][2] = mark_article 

            result.append(articles_temp[sent2arti])
    else:
        result = articles
        
    return render(request, 'html/default_no_tfidf.html', locals())