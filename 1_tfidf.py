import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt


docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup"
]

# idf = log(文档数/所有文档中的词w数）[n_vocab, 1]
idf_methods = {
    "log": lambda x: 1+ np.log(len(docs)/(x+1)),
    "prob": lambda x: np.maximum(0, np.log((len(docs)-x)/(x+1))),
    "len_norm": lambda x: x/(np.sum(np.square(x)))
}
# tf = 文档d中词w出现的总数, [n_vocab, n_doc]
tf_methods = {
    "log": lambda x:np.log(1+x),
    "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
    'boolean': lambda x: np.minimum(x, 1),
    'log_avg': lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True)))
}

def preprocessing():
    # 分词
    docs_words = [d.replace(',', "").split(" ") for d in docs]
    # print(docs_words)
    # 构建词典
    vocabulary = set(itertools.chain(*docs_words))
    # print(vocab)
    w2i = {w:i for i, w in enumerate(vocabulary)}
    i2v = {i:w for w, i in w2i.items()}
    #print(w2i)
    #print(i2v)
    vocab = {"words":vocabulary, "w2i":w2i, "i2w":i2v}

    return docs_words, vocab


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x


def get_tf(docs_words, vocab, method='log'):
    # term frequency: how frequent a word appears in a doc
    # tf = 文档d中词w出现的总数, [n_vocab, n_doc]
    # 创建指定shape的数组
    _tf = np.zeros((len(vocab['w2i']), len(docs_words)), dtype=np.float64)
    # 每个文档扫一遍，文档内出现的每个词进行赋值
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for word in counter.keys():
            _tf[vocab['w2i'][word], i] = counter[word]/counter.most_common(1)[0][1]
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    tf = weighted_tf(_tf)
    return tf



def get_idf(docs_words, vocab, method='log'):
    # idf = log(文档数/所有文档中的词w数）[n_vocab, 1]
    # inverse document frequency: low idf for a word appears in more docs, mean less importan
    # 虽然是idf，也是一个词一个值
    df = np.zeros((len(vocab['i2w']), 1))
    for i in range(len(vocab['i2w'])):
        d_count = 0
        for d in docs_words:
            d_count += 1 if vocab['i2w'][i] in d else 0
        df[i, 0] = d_count
    #get() 参数1是要查找的key，第二个是没找到的默认返回值
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df) # [n_vocab, 1]


def cosine_similarity(q, _tf_idf):
    unit_q = q/np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf/np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    print(unit_ds.shape)
    similarity = unit_ds.T.dot(unit_q).ravel() #ravel 多维转换成1维
    return similarity


def docs_score(q, vocab, idf, tf_idf, len_norm=False):
    q_words = q.replace(',', "").split(" ")

    # add unknown words
    unknown_w = 0
    for w in set(q_words):
        if w not in vocab['w2i']:
            vocab['w2i'][w] = len(vocab['w2i'])
            vocab['i2w'][len(vocab['i2w'])] = w
            unknown_w += 1

    # 如果有未知，扩张ifidf矩阵
    if unknown_w > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_w, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_w, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf

    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)  # [n_vocab, 1]
    for w in counter.keys():
        q_tf[vocab['w2i'][w], 0] = counter[w]

    q_vec = q_tf * _idf  # [n_vocab, 1]
    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores

def show_tfidf(tfidf, vocb, filename):
    # [n_vocab, n_doc]
    #print(tfidf.shape[0])
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocb, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0]+1), fontsize=6)
    plt.tight_layout()
    #plt.savefig("./visual/results/%s.png" % filename, format="png", dpi=500)
    plt.show()


def get_keywords(vocab, tfidf, n=2):
    for c in range(3):
        col = tfidf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{}, keywords {}".format(c, n, [vocab['i2w'][i] for i in idx]))


if __name__ == "__main__":
    docs_words, vocab = preprocessing()
    tf = get_tf(docs_words, vocab)# [n_vocab, n_doc]
    idf = get_idf(docs_words, vocab)# [n_vocab, 1]
    tf_idf = tf * idf # [n_vocab, n_doc]
    print("tf shape(vecb in each docs): ", tf.shape)
    print("\ntf samples:\n", tf[:2])
    print("\nidf shape(vecb in all docs): ", idf.shape)
    print("\nidf samples:\n", idf[:2])
    print("\ntf_idf shape: ", tf_idf.shape)
    print("\ntf_idf sample:\n", tf_idf[:2])

    show_tfidf(tf_idf.T, [vocab['i2w'][i] for i in range(len(vocab['i2w']))], "tfidf_matrix")

    #test
    q = "I get a coffee cup"
    scores = docs_score(q, vocab, idf, tf_idf)
    #print(scores.shape)
    # argsort 从小到大
    d_ids = scores.argsort()[-3:][::-1]
    print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))
    get_keywords(vocab, tf_idf, 2)