import time
from pathlib import Path
import streamlit as st
import tensorflow as tf

PAIRS = [
    "az_to_en","aztr_to_en","be_to_en","beru_to_en","es_to_pt","fr_to_pt",
    "gl_to_en","glpt_to_en","he_to_pt","it_to_pt","pt_to_en",
    "ru_to_en","ru_to_pt","tr_to_en"
]

EXPORT_ROOT = Path("export")

@st.cache_resource
def load_model(pair):
    model = tf.saved_model.load(str(EXPORT_ROOT / pair))
    if hasattr(model, "signatures") and "translate" in model.signatures:
        return model.signatures["translate"]
    return model

def translate(model, text):
    x = tf.constant([text])
    try:
        out = model(text=x)
    except:
        out = model(x)
    if isinstance(out, dict):
        out = list(out.values())[0]
    y = out[0]
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    if isinstance(y, bytes):
        return y.decode("utf-8")
    return str(y)

st.set_page_config(layout="centered")
st.title("NMT Translation Demo")

pair = st.selectbox("Language pair", PAIRS)
text = st.text_area("Input sentence", height=120)

if st.button("Translate"):
    model = load_model(pair)
    start = time.perf_counter()
    result = translate(model, text)
    end = time.perf_counter()
    st.subheader("Translation")
    st.write(result)
    st.subheader("Inference time (ms)")
    st.write(f"{(end - start) * 1000:.2f}")
