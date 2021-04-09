from utils import process_w2v_data
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import Dataset


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
        self.w_dim = w_dim # n_vocab
        self.emb_dim = emb_dim
        # network
        self.embeddings = keras.layers.Embedding(
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
        o = self.embeddings(x)
        o = tf.reduce_mean(o,axis=1) # [n, emb_dim] 取了平均
        return o

    def loss(self, x, y, training=None):
        embedded = self.call(x, training)
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
                inputs=embedded, num_sampled=5, num_classes=self.w_dim)
            )

    def step(self, x, y):
        with tf.GradientTape() as tape:
            loss= self.loss(x, y, training=True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients((zip(grads, self.trainable_variables)))
        return loss.numpy()


def train(model, data):
    for t in range(2500):
        bx, by = data.sample(8)
        loss = model.step(bx, by)
        if t %200 ==0:
            print("step: {} | loss: {}".format(t, loss))


def show_w2v_word_embedding(model, data: Dataset, path):
    word_emb = model.embeddings.get_weights()[0]
    for i in range(data.num_word):
        c = "blue"
        try:
            int(data.i2v[i])
        except ValueError:
            c = "red"
        plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
    plt.xlim(word_emb[:, 0].min() - .5, word_emb[:, 0].max() + .5)
    plt.ylim(word_emb[:, 1].min() - .5, word_emb[:, 1].max() + .5)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    plt.savefig(path, dpi=300, format="png")
    plt.show()


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method='cbow')
    m = CBOW(d.num_word, 2)
    train(m, d)

    # plotting
    show_w2v_word_embedding(m, d, "cbow.png")

    # TODO: plotting
