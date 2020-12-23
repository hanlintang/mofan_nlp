from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


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
    "It is coffee time, bring your cup",
]


def show_tfidf(tfidf, vocb, filename):
    # [n_vocab, n_doc]
    #print(tfidf.shape[0])
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocb, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0]+1), fontsize=6)
    plt.tight_layout()
    #plt.savefig("./visual/results/%s.png" % filename, format="png", dpi=500)
    plt.show()


# 用sklearn的sparse matrix，只记录有数值的地方，防止过度占用内存
if __name__ == "__main__":
    # tfidfvectorizer是tfidf transformer和countvectorizer（文档版词袋模型的向量）
    #先转成向量再算tfidf
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(docs)
    # from sklearn.feature_extraction.text import TfidfTransformer
    #transformer = TfidfTransformer()
    #tf_idf1 = transformer.fit_transform(docs)
    print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
    print("w2i: ", vectorizer.vocabulary_)

    query = "I get a coffee cup"
    # 此处不能再用fit_transform，fit只能在训练集当中用
    qtf_idf = vectorizer.transform([query])
    res = cosine_similarity(tf_idf, qtf_idf)
    res = res.ravel().argsort()[-3:]
    print("\ntop 3 docs for '{}':\n{}".format(query, [docs[i] for i in res[::-1]]))
    #print(tf_idf)

    i2w = {i:w for w, i in vectorizer.vocabulary_.items()}
    # sklearn默认是一个sparse matrix, 要转成dense matrix
    show_tfidf(tf_idf.todense(), [i2w[i] for i in range(len(i2w))], "tfidf_sklearn_matrix")
