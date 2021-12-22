# IR_HW4

## Introduction:  
1. Use 100 articles about liver.  
2. Implement 2 different TF/IDF to rank documents. Each document has a score.  
3. You can type string to search the documents about the string.  
4. By clicking title, you can enter the content.  
5. After entering content, type string can search sentences containing the string.  
6. One calculates target string score, another calculates sentence average score.  

## System environment :   
。Architecture : Django   
。Language : Python4.0 , HTML , JavaScript    
。Dataset : PubMed about liver 100 articles  

## 觀察到的事  
1. 正常來說文章越長代表分數越高，因為更容易有重要的文字  
2. Gensim是先將你想要計算的文字tfidf利用模型預先計算好再根據你丟進來的文章去算tfidf分數，  
	沒出現的就不計分，且會有對STOPWORD的懲罰，所以分數會略低  

## Calculate TF/IDF in 4863 words in 100 articles :

![image](https://drive.google.com/uc?export=view&id=14Xkce52-kgrNgtw_un1CanMUJnqAhq92)    


## Use tfidf combine word embedding to sentence ranking
先對每個單詞計算其word embedding後  
在對所有句子算tfidf  
把每個單詞的embedding乘上對應的tfidf值來算出新的embedding  
利用新的embedding來計算similarity取出前10名印出  

## 實際畫面  
#### default page (use tfidf embedding):  
![image](https://drive.google.com/uc?export=view&id=1AcGrlpl_u12vs9GAa-lPEGH43di5EvL6)    
#### search:  
![image](https://drive.google.com/uc?export=view&id=17ewaR6qGUY2pfIitM5BNZHm3JAIZxcW1)    

rank/default_no_tfidf can see original embedding  

