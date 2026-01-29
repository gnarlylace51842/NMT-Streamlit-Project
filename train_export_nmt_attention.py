import argparse
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_DIR = "/Users/dylanashraf/Downloads"

def standardize(x):
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r"([?.!,¿])", r" \1 ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, enc, hid):
        hid = tf.expand_dims(hid, 1)
        score = self.v(tf.nn.tanh(self.w1(enc) + self.w2(hid)))
        w = tf.nn.softmax(score, axis=1)
        ctx = tf.reduce_sum(w * enc, axis=1)
        return ctx

class Encoder(tf.keras.Model):
    def __init__(self, vocab, emb, units):
        super().__init__()
        self.e = tf.keras.layers.Embedding(vocab, emb, mask_zero=True)
        self.g = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.e(x)
        return self.g(x)

class Decoder(tf.keras.Model):
    def __init__(self, vocab, emb, units):
        super().__init__()
        self.e = tf.keras.layers.Embedding(vocab, emb, mask_zero=True)
        self.g = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.f = tf.keras.layers.Dense(vocab)
        self.a = Attention(units)

    def call(self, x, hid, enc):
        ctx = self.a(enc, hid)
        ctx = tf.expand_dims(ctx, 1)
        x = self.e(x)
        x = tf.concat([ctx, x], axis=-1)
        o, s = self.g(x)
        return self.f(o), s

class Seq2Seq(tf.keras.Model):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def call(self, src, tgt):
        enc_out, enc_state = self.enc(src)
        state = enc_state
        outs = []
        for t in range(tgt.shape[1]):
            o, state = self.dec(tgt[:, t:t+1], state, enc_out)
            outs.append(o)
        return tf.concat(outs, axis=1)

def loss_fn(y, yhat):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, yhat, from_logits=True)
    mask = tf.cast(y != 0, tf.float32)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

class Translator(tf.Module):
    def __init__(self, sv, tv, enc, dec, maxlen):
        super().__init__()
        self.sv = sv
        self.tv = tv
        self.enc = enc
        self.dec = dec
        self.maxlen = maxlen
        self.vocab = tf.constant(tv.get_vocabulary())
        self.start = tf.constant(self.vocab.numpy().tolist().index("[start]"))
        self.end = tf.constant(self.vocab.numpy().tolist().index("[end]"))

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def translate(self, text):
        src = self.sv(text)
        enc_out, enc_state = self.enc(src)
        state = enc_state
        cur = tf.fill([tf.shape(src)[0], 1], self.start)
        out = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        for i in tf.range(self.maxlen):
            logits, state = self.dec(cur, state, enc_out)
            nxt = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
            out = out.write(i, nxt)
            cur = tf.expand_dims(nxt, 1)

        out = tf.transpose(out.stack(), [1, 0])

        def decode(ids):
            idx = tf.where(ids == self.end)
            cut = tf.cond(tf.size(idx) > 0, lambda: idx[0][0], lambda: tf.shape(ids)[0])
            tokens = tf.gather(self.vocab, ids[:cut])
            return tf.strings.reduce_join(tokens, separator=" ")

        return {"translation": tf.map_fn(decode, out, tf.string)}

def main(pair, epochs):
    data, info = tfds.load(
        f"ted_hrlr_translate/{pair}",
        with_info=True,
        data_dir=DATA_DIR
    )

    src, tgt = info.supervised_keys
    train = data["train"]

    sv = tf.keras.layers.TextVectorization(
        standardize=standardize,
        max_tokens=20000,
        output_sequence_length=40
    )
    tv = tf.keras.layers.TextVectorization(
        standardize=standardize,
        max_tokens=20000,
        output_sequence_length=40
    )

    sv.adapt(train.map(lambda x: x[src]).batch(2048))
    tv.adapt(train.map(lambda x: "[start] " + x[tgt] + " [end]").batch(2048))

    def prep(x):
        s = sv(x[src])
        t = tv("[start] " + x[tgt] + " [end]")
        return (s, t[:, :-1]), t[:, 1:]

    ds = train.map(prep).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

    enc = Encoder(len(sv.get_vocabulary()), 256, 512)
    dec = Decoder(len(tv.get_vocabulary()), 256, 512)
    model = Seq2Seq(enc, dec)
    model.compile(optimizer="adam", loss=loss_fn)
    model.fit(ds, epochs=epochs)

    export = Path("export") / pair
    export.mkdir(parents=True, exist_ok=True)

    translator = Translator(sv, tv, enc, dec, 40)
    tf.saved_model.save(translator, export, signatures={"translate": translator.translate})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="pt_to_en")
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    main(a.pair, a.epochs)
