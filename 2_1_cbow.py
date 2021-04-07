from utils import process_w2v_data
from tensorflow import keras
import tensorflow as tf


corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


class CBOW(keras.Model):
    def __init__(self, w_dim, emb_dim):
        super(CBOW, self).__init__()
        self.w_dim = w_dim
        self.emb_dim = emb_dim
        # network
        self.embedding = keras.layers.Embedding(
            input_dim=self.w_dim, output_dim=emb_dim,
            embeddings_initializer=keras.initializers.RandomNormal(0.0, 0.1)
        )# [n_vocab, emb_dim]
        # noise-contrastive estimation
        # word2vec 给了负采样和层次softmax，nce和前面的一样可以降低复杂度
        self.nce_w = self.add_weight(name="nce_w", shape=[self.w_dim, self.emb_dim], initializer=keras.initializers.TruncatedNormal(0., 0.1))# [n_vocab, emb_dim]
        self.nce_b = self.add_weight(name="nce_b", shape=[self.w_dim, ], initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]
        # optimizer
        self.opt = keras.optimizers.Adam(learning_rate=0.01)

    def call(self, x, training=None, mask=None):
    # x.shape = [n, skip_window*2]
    # skip_window 是一侧的
        o = self.embeddiings(x)
        o = tf.re



def train():
    pass


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method='skip-gram')
    m = CBOW()
    train()

    # TODO: plotting
