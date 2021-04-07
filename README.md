# mofan_nlp
莫烦老师的课程地址：https://mofanpy.com/tutorials/machine-learning/nlp/
## 第一章 搜索

### 1.1 搜索引擎
#### 搜索过滤
层次地进行过滤，从大量文档一层层筛选过滤，第一层用时间快精度低的方法，后面逐渐用精度高时间长的方法。

#### 倒排索引
把所有材料都建立关键词与文章的对应关系。找到某个关键词索引的文章，再用tfidf进行排序。


### 1.2 Tf-Idf
#### Tf
词频： 

term frequency: how frequent a word appears in a doc

 tf = 文档d中词w出现的总数, [n_vocab, n_doc]
 
#### Idf
逆文本频率指数

idf = log(文档数/所有文档中的词w数）[n_vocab, 1]

两者乘积就是tf-idf

TODO：代码解析

