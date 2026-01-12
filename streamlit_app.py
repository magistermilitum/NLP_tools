# streamlit_app.py
import time
import json
import glob
from io import StringIO

import streamlit as st
import numpy as np
from PIL import Image

from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier

from kraken import blla, rpred
from kraken.lib import vgsl, models as kraken_models

# -------------------------
# Streamlit config (arriba)
# -------------------------
st.set_page_config(
    page_title="medieval charter analyze",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Caching: recursos (modelos)
# -------------------------
@st.cache_resource
def load_ner_tagger():
    return SequenceTagger.load("models/best-model_flat_13_03_2022.pt")

@st.cache_resource
def load_discourse_tagger():
    return SequenceTagger.load("models/discours_parts_05_02_2022.pt")

@st.cache_resource
def load_charter_classifier():
    return TextClassifier.load("models/charters_class_05_02_2022.pt")

@st.cache_resource
def load_kraken_seg_model():
    # OJO: en Kraken, .mlmodel aquí parece ser un modelo VGSL; mantengo tu ruta
    model_path = "models/blla.mlmodel"
    return vgsl.TorchVGSLModel.load_model(model_path)

@st.cache_resource
def load_kraken_rec_model():
    rec_model_path = "models/model_36.mlmodel"
    return kraken_models.load_any(rec_model_path)

# -------------------------
# Caching: datos (procesos)
# -------------------------
@st.cache_data
def read_image_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(StringIO(file_bytes.decode("latin-1")))  # NO recomendado

# Mejor: bytes -> Image sin hacks
@st.cache_data
def read_image_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file)

def WORD2HTML(sentence: Sentence):
    # Nota: tu parsing via str(token) es frágil, pero lo dejo igual para no romperte outputs
    CONLL_html = [str(token).split("Token: ")[1].split()[1] for token in sentence]
    tokenized_text = [str(token).split("Token: ")[1].split()[1] for token in sentence]

    index_entities = ["O"] * len(tokenized_text)
    dict_colors = {"PERS": "255FD2", "ORG": "EE4220", "MISC": "19842E", "LOC": "7A1A7A"}

    for entity in sentence.get_spans("ner"):
        x = " ".join([(str(y).split("Token: ")[1]) for y in entity]).split()[::2]
        x = [int(x[0]) - 1, int(x[-1])]

        type_ent = str(entity).split("− Labels: ")[1].split()[0]

        if x[1] - x[0] > 1:
            index_entities[x[0]:x[1]] = ["B-" + type_ent] + ["I-" + type_ent] * (x[1] - x[0] - 1)
            if x[1] - x[0] > 2:
                CONLL_html[x[0]:x[1]] = (
                    [f'<span style="background-color: #{dict_colors[type_ent]}; padding:1px">{CONLL_html[x[0]]}']
                    + [w for w in CONLL_html[x[0] + 1 : x[1] - 1]]
                    + [CONLL_html[x[1] - 1] + "</span>"]
                )
            else:
                CONLL_html[x[0]:x[1]] = [
                    f'<span style="background-color: #{dict_colors[type_ent]}; padding:1px">{CONLL_html[x[0]]}',
                    CONLL_html[x[1] - 1] + "</span>",
                ]
        else:
            index_entities[x[0]:x[1]] = ["B-" + type_ent] * (x[1] - x[0])
            CONLL_html[x[0]:x[1]] = [
                f'<span style="background-color: #{dict_colors[type_ent]}; padding:1px">{CONLL_html[x[0]:x[1]][0]}</span>'
            ]

    return CONLL_html

@st.cache_data
def ner_html(text: str) -> str:
    sent = Sentence(text)
    tagger = load_ner_tagger()
    tagger.predict(sent)
    return " ".join(WORD2HTML(sent))

@st.cache_data
def parts_dis_html(text: str) -> str:
    dis_model = load_discourse_tagger()
    dis_sent = Sentence(text)
    dis_model.predict(dis_sent)

    ner_tagger = load_ner_tagger()
    ner_sent = Sentence(text)
    ner_tagger.predict(ner_sent)
    tagged_sent = WORD2HTML(ner_sent)

    tokenized_text = [str(token).split("Token: ")[1].split()[1] for token in dis_sent]
    parts_discours = []

    for x in dis_sent.get_spans("ner"):
        idx = " ".join([(str(y).split("Token: ")[1]) for y in x]).split()[::2]
        idx = [int(idx[0]) - 1, int(idx[-1])]
        part = str(x).split("[− Labels: ")[1].replace("]", "")
        parts_discours.append([part, " ".join(tagged_sent[idx[0]:idx[1]])])

    html = "<table>"
    for p, frag in parts_discours:
        html += f"<tr><td>{p}</td><td>{frag}</td></tr>"
    html += "</table>"
    return html

@st.cache_data
def classify_charter(text: str) -> str:
    clf = load_charter_classifier()
    sent = Sentence(text)
    clf.predict(sent)
    return f"Most probably type : {sent.labels[0]}"

@st.cache_data
def kraken_segment(img: Image.Image):
    seg_model = load_kraken_seg_model()
    return blla.segment(img, model=seg_model)

@st.cache_data
def kraken_transcribe(img: Image.Image, baseline_seg):
    rec_model = load_kraken_rec_model()
    pred_it = rpred.rpred(network=rec_model, im=img, bounds=baseline_seg)
    pred_char = [f"Lin. {i+1} : {record.prediction}<br>" for i, record in enumerate(pred_it)]
    return " ".join(pred_char)

# -------------------------
# UI helpers
# -------------------------
def p_title(title: str):
    st.markdown(
        f'<h3 style="text-align:left; color:#F63366; font-size:28px;">{title}</h3>',
        unsafe_allow_html=True,
    )

# -------------------------
# Sidebar / Navigation
# (Arreglo: tus if nav == 'Paraphrase text' nunca corrían porque no estaba en el radio)
# -------------------------
st.sidebar.header("Analyze medieval charter")
nav = st.sidebar.radio(
    "",
    [
        "Go to homepage",
        "medieval charter analyze",
        "Paraphrase text",
        "Analyze text",
        "Handwritten text recognition",
        "OCR enriching engine",
    ],
)

# -------------------------
# Pages
# -------------------------
if nav == "Go to homepage":
    st.markdown("<h1 style='text-align:center; color:white; font-size:28px;'>Easy charter analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; font-size:56px;'><p>🤖</p></h3>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; color:grey; font-size:20px;'>Summarize, paraphrase, analyze text & more.</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.write("Use the menu at left to select a task.")

elif nav == "medieval charter analyze":
    st.markdown("<h4 style='text-align:center; color:grey;'>Accelerate knowledge with Streamlit 🤖</h4>", unsafe_allow_html=True)
    p_title("Analyze charter")

    s_example = "In nomine sancte et indiuidue Trinitatis , Amen . ..."

    source = st.radio("How would you like to start?", ("I want to input some text", "I want to upload a file"))

    if source == "I want to input some text":
        input_su = st.text_area(
            "Input your own text in Latin, French or Spanish (1,000–10,000 chars)",
            value=s_example,
            max_chars=10000,
            height=330,
        )

        if st.button("medieval charter analyze"):
            if len(input_su) < 1000:
                st.error("Please enter a text of minimum 1,000 characters")
            else:
                with st.spinner("Processing..."):
                    st.markdown("---")
                    st.write("Named entities in Flat mode")
                    st.write(ner_html(input_su), unsafe_allow_html=True)

                    st.markdown("---")
                    st.success(classify_charter(input_su))

                    st.markdown("---")
                    st.write("Diplomatics parts")
                    st.write(parts_dis_html(input_su), unsafe_allow_html=True)

    else:
        file = st.file_uploader("Upload your file here", type=["jpg", "jpeg", "png"])
        if file is not None:
            with st.spinner("Processing..."):
                img = read_image_upload(file)
                st.image(img, width=250)
                seg = kraken_segment(img)
                st.write(kraken_transcribe(img, seg), unsafe_allow_html=True)

elif nav == "Paraphrase text":
    # IMPORT PEREZOSO: sólo se importa si entras a esta página
    try:
        from googletrans import Translator
        from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter
    except Exception as e:
        st.error(f"Missing dependencies for Paraphrase page: {e}")
        st.stop()

    p_title("Paraphrase")
    p_example = "Health is the level of functional or metabolic efficiency..."
    input_pa = st.text_area("Input your own text in English (max 500 chars)", max_chars=500, value=p_example, height=160)

    if st.button("Paraphrase"):
        with st.spinner("Wait for it..."):
            translator = Translator()
            mid = translator.translate(input_pa, dest="fr").text
            mid2 = translator.translate(mid, dest="de").text
            back = translator.translate(mid2, dest="en").text
            st.markdown("---")
            st.write("Back Translation Model")
            st.success(back)

            e_augmenter = EmbeddingAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
            st.markdown("---")
            st.write("Embedding Augmenter Model")
            st.success(e_augmenter.augment(input_pa))

            w_augmenter = WordNetAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
            st.markdown("---")
            st.write("WordNet Augmenter Model")
            st.success(w_augmenter.augment(input_pa))

elif nav == "Analyze text":
    # IMPORT PEREZOSO
    try:
        import nltk
        import readtime
        import textstat
        from nltk.tokenize import word_tokenize
    except Exception as e:
        st.error(f"Missing dependencies for Analyze page: {e}")
        st.stop()

    p_title("Analyze text")
    a_example = "Artificial intelligence (AI) is intelligence demonstrated by machines..."
    input_me = st.text_area("Input your own text in English (max 10,000 chars)", max_chars=10000, value=a_example, height=330)

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            nltk.download("punkt", quiet=True)
            rt = readtime.of_text(input_me)
            tc = textstat.flesch_reading_ease(input_me)
            tokenized_words = word_tokenize(input_me)
            lr = round(len(set(tokenized_words)) / max(1, len(tokenized_words)), 2)
            n_s = textstat.sentence_count(input_me)

            st.markdown("---")
            st.write("Reading Time"); st.write(rt)
            st.markdown("---")
            st.write("Text Complexity"); st.write(tc)
            st.markdown("---")
            st.write("Lexical Richness"); st.write(lr)
            st.markdown("---")
            st.write("Number of sentences"); st.write(n_s)

else:
    st.info("Page not implemented yet.")
