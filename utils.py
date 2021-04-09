import numpy as np
import itertools
import datetime


PAD_ID = 0


class Dataset:
    def __init__(self, x, y, v2i, i2v):
        self.x, self.y = x, y
        self.v2i, self.i2v = v2i, i2v
        self.vocab = v2i.keys()

    def sample(self, n):
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx], self.y[b_idx]
        return bx, by

    @property
    def num_word(self):
        return len(self.v2i)


def process_w2v_data(corpus, skip_window=2, method='skip-gram'):
    all_words = [sentence.split(" ") for sentence in corpus]
    all_words = np.array(list(itertools.chain(*all_words)))
    # vocab sort by decreasing frequency for the negative sampling below (nce_loss).
    vocab, v_count = np.unique(all_words, return_counts=True)
    vocab = vocab[np.argsort(v_count)[::-1]]
    print("all vocabularies sorted from more frequent to less frequent:\n", vocab)
    w2i = {w:i for i, w in enumerate(vocab)}
    i2w = {i:w for w, i in w2i.items()}

    # pair data
    pairs = []
    js = [i for i in range(-skip_window, skip_window+1) if i!=0] # 偏移量

    for sentence in corpus:
        words = sentence.split(" ")
        w_idx = [w2i[w] for w in words]
        if method == 'skip_gram':
            pass
            # for i in range(len(w_idx)):
            #     for j in js:
            #         if i+j<0 or i+j>=len(w_idx):
        elif method.lower() == 'cbow':
            for i in range(skip_window, len(w_idx)- skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i+j])
                pairs.append(context + [w_idx[i]])  # (contexts, center) or (feature, target)
        else:
            raise ValueError

    pairs = np.array(pairs)
    print("5 example pairs:\n", pairs[:5])
    if method.lower() == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    elif method.lower() == "cbow":
        x, y = pairs[:, :-1], pairs[:, -1]
    else:
        raise ValueError
    return Dataset(x, y, w2i, i2w)