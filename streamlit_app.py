# streamlit_app.py
# SYNTHIA (modernized Streamlit + fixed Flair token/span handling)

from __future__ import annotations

import json
import time
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# --- Optional / heavy deps (loaded lazily in cached loaders) ---
# flair, kraken, nltk, etc. are imported in the functions that need them.


APP_TITLE = "medieval charter analyze"
MODELS_DIR = Path("models")


# ======================================================================================
# Streamlit config
# ======================================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================================
# Cached resource loaders (models)
# ======================================================================================
@st.cache_resource(show_spinner=False)
def load_ner_tagger():
    from flair.models import SequenceTagger

    model_path = MODELS_DIR / "best-model_flat_13_03_2022.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"NER model not found: {model_path.resolve()}")
    return SequenceTagger.load(str(model_path))


@st.cache_resource(show_spinner=False)
def load_discourse_tagger():
    from flair.models import SequenceTagger

    model_path = MODELS_DIR / "discours_parts_05_02_2022.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Discourse model not found: {model_path.resolve()}")
    return SequenceTagger.load(str(model_path))


@st.cache_resource(show_spinner=False)
def load_charter_classifier():
    from flair.models import TextClassifier

    model_path = MODELS_DIR / "charters_class_05_02_2022.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {model_path.resolve()}")
    return TextClassifier.load(str(model_path))


@st.cache_resource(show_spinner=False)
def load_kraken_segmentation_model():
    # blla segmentation model (VGSL Torch model)
    from kraken.lib import vgsl

    model_path = MODELS_DIR / "blla.mlmodel"
    if not model_path.exists():
        raise FileNotFoundError(f"Kraken segmentation model not found: {model_path.resolve()}")
    return vgsl.TorchVGSLModel.load_model(str(model_path))


@st.cache_resource(show_spinner=False)
def load_kraken_recognition_model():
    from kraken.lib import models

    model_path = MODELS_DIR / "model_36.mlmodel"
    if not model_path.exists():
        raise FileNotFoundError(f"Kraken recognition model not found: {model_path.resolve()}")
    return models.load_any(str(model_path))


# ======================================================================================
# Flair helpers (FIXED: no parsing of str(token))
# ======================================================================================
def _span_label(span, label_type: str = "ner") -> str:
    # Flair modern: span.get_label(label_type).value
    try:
        return span.get_label(label_type).value
    except Exception:
        # fallback
        if getattr(span, "labels", None):
            return span.labels[0].value
        return "MISC"


@st.cache_data(show_spinner=False)
def word2html(sentence_text: str, label_type: str = "ner") -> List[str]:
    """
    Returns token list with <span ...> wrappers for Flair spans.
    Robust to Flair version changes.
    """
    from flair.data import Sentence

    sent = Sentence(sentence_text)
    tagger = load_ner_tagger() if label_type == "ner" else None
    if tagger is None:
        # If someone calls with another label_type, they should predict outside.
        # Kept for compatibility.
        pass
    else:
        tagger.predict(sent)

    tokens = [t.text for t in sent]
    html_tokens = tokens.copy()

    dict_colors = {"PERS": "255FD2", "ORG": "EE4220", "MISC": "19842E", "LOC": "7A1A7A"}

    for span in sent.get_spans(label_type):
        label = _span_label(span, label_type)
        color = dict_colors.get(label, "FFD54F")  # default if unknown label

        # token.idx starts at 1 in Flair
        start = span.tokens[0].idx - 1
        end_excl = span.tokens[-1].idx  # exclusive

        if 0 <= start < len(html_tokens):
            html_tokens[start] = (
                f'<span style="background-color: #{color}; padding:1px">{html_tokens[start]}'
            )
        if 0 <= end_excl - 1 < len(html_tokens):
            html_tokens[end_excl - 1] = f"{html_tokens[end_excl - 1]}</span>"

    return html_tokens


@st.cache_data(show_spinner=False)
def ner_html(text: str) -> str:
    """
    Named Entities HTML using your Flair NER tagger.
    """
    from flair.data import Sentence

    sent = Sentence(text)
    tagger = load_ner_tagger()
    tagger.predict(sent)

    tokens = [t.text for t in sent]
    html_tokens = tokens.copy()

    dict_colors = {"PERS": "255FD2", "ORG": "EE4220", "MISC": "19842E", "LOC": "7A1A7A"}

    for span in sent.get_spans("ner"):
        label = _span_label(span, "ner")
        color = dict_colors.get(label, "FFD54F")

        start = span.tokens[0].idx - 1
        end_excl = span.tokens[-1].idx

        if 0 <= start < len(html_tokens):
            html_tokens[start] = (
                f'<span style="background-color: #{color}; padding:1px">{html_tokens[start]}'
            )
        if 0 <= end_excl - 1 < len(html_tokens):
            html_tokens[end_excl - 1] = f"{html_tokens[end_excl - 1]}</span>"

    return " ".join(html_tokens)


@st.cache_data(show_spinner=False)
def parts_dis_html(text: str) -> str:
    """
    Diplomatic parts (your discourse tagger) + NER highlight inside each part.
    FIXED: uses span.tokens and span.get_label, no string parsing.
    """
    from flair.data import Sentence

    # predict discourse parts
    dis_model = load_discourse_tagger()
    dis_sent = Sentence(text)
    dis_model.predict(dis_sent)

    # predict NER once, then reuse token-level HTML slices
    ner_tagger = load_ner_tagger()
    ner_sent = Sentence(text)
    ner_tagger.predict(ner_sent)

    ner_tokens = [t.text for t in ner_sent]
    ner_html_tokens = ner_tokens.copy()

    dict_colors = {"PERS": "255FD2", "ORG": "EE4220", "MISC": "19842E", "LOC": "7A1A7A"}

    for span in ner_sent.get_spans("ner"):
        label = _span_label(span, "ner")
        color = dict_colors.get(label, "FFD54F")

        start = span.tokens[0].idx - 1
        end_excl = span.tokens[-1].idx

        if 0 <= start < len(ner_html_tokens):
            ner_html_tokens[start] = (
                f'<span style="background-color: #{color}; padding:1px">{ner_html_tokens[start]}'
            )
        if 0 <= end_excl - 1 < len(ner_html_tokens):
            ner_html_tokens[end_excl - 1] = f"{ner_html_tokens[end_excl - 1]}</span>"

    parts_rows: List[Tuple[str, str]] = []
    # IMPORTANT: your discourse tagger stores spans under 'ner' (as in your legacy code).
    # If your discourse model uses a different label type, change "ner" below accordingly.
    for sp in dis_sent.get_spans("ner"):
        part = _span_label(sp, "ner")
        start = sp.tokens[0].idx - 1
        end_excl = sp.tokens[-1].idx
        frag = " ".join(ner_html_tokens[start:end_excl]) if 0 <= start < end_excl else ""
        parts_rows.append((part, frag))

    html = "<table>"
    for part, frag in parts_rows:
        html += f"<tr><td>{part}</td><td>{frag}</td></tr>"
    html += "</table>"
    return html


@st.cache_data(show_spinner=False)
def class_acta(text: str) -> str:
    from flair.data import Sentence

    clf = load_charter_classifier()
    sent = Sentence(text)
    clf.predict(sent)
    if not sent.labels:
        return "Most probably type: (no label)"
    return f"Most probably type: {sent.labels[0]}"


# ======================================================================================
# Kraken helpers
# ======================================================================================
@st.cache_data(show_spinner=False)
def read_image(upload_or_path) -> Image.Image:
    """
    Accepts a Streamlit UploadedFile or a path/bytes, returns PIL Image.
    """
    if hasattr(upload_or_path, "getvalue"):
        data = upload_or_path.getvalue()
        return Image.open(BytesIO(data)).convert("RGB")
    if isinstance(upload_or_path, (bytes, bytearray)):
        return Image.open(BytesIO(upload_or_path)).convert("RGB")
    return Image.open(upload_or_path).convert("RGB")


@st.cache_data(show_spinner=False)
def segmentation(img: Image.Image):
    from kraken import blla

    model = load_kraken_segmentation_model()
    baseline_seg = blla.segment(img, model=model)
    return baseline_seg


@st.cache_data(show_spinner=False)
def transcript(img: Image.Image, baseline_seg) -> str:
    from kraken import rpred

    rec_model = load_kraken_recognition_model()
    pred_it = rpred.rpred(network=rec_model, im=img, bounds=baseline_seg)

    lines = []
    for i, record in enumerate(pred_it):
        lines.append(f"Lin. {i+1} : {record.prediction}<br>")
    return " ".join(lines)


# ======================================================================================
# UI helpers
# ======================================================================================
def p_title(title: str):
    st.markdown(
        f'<h3 style="text-align:left; color:#F63366; font-size:28px;">{title}</h3>',
        unsafe_allow_html=True,
    )


# ======================================================================================
# Sidebar
# ======================================================================================
st.sidebar.header("Analyze medieval charter")

nav = st.sidebar.radio(
    "",
    [
        "Go to homepage",
        "medieval charter analyze",
        "Handwritten text recognition",
        "Paraphrase text",
        "Analyze text",
    ],
)

st.sidebar.write("")
st.sidebar.write("")

expander = st.sidebar.expander("Contact")
expander.write(
    "I'd love your feedback. Want to collaborate? "
    "Find me on [LinkedIn](https://www.linkedin.com/in/lopezyse/), "
    "[Twitter](https://twitter.com/lopezyse) and "
    "[Medium](https://lopezyse.medium.com/)"
)


# ======================================================================================
# Pages
# ======================================================================================
if nav == "Go to homepage":
    st.markdown(
        "<h1 style='text-align:center; color:white; font-size:28px;'>Easy charter analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center; font-size:56px;'><p>🤖</p></h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center; color:grey; font-size:20px;'>"
        "Summarize, paraphrase, analyze text & more. Try our models, browse their source code, and share with the world!"
        "</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("___")
    st.write("⬅️ Use the menu at left to select a task.")
    st.markdown("___")
    st.markdown(
        "<h3 style='text-align:left; color:#F63366; font-size:18px;'><b>What is this App about?</b></h3>",
        unsafe_allow_html=True,
    )
    st.write("Personalized NLP utilities for text + historical documents workflows.")
    st.markdown(
        "<h3 style='text-align:left; color:#F63366; font-size:18px;'><b>Who is this App for?</b></h3>",
        unsafe_allow_html=True,
    )
    st.write("Researchers and practitioners working with historical texts and images.")


# --------------------------------------------------------------------------------------
if nav == "medieval charter analyze":
    st.markdown(
        "<h4 style='text-align:center; color:grey;'>Accelerate knowledge with Streamlit 🤖</h4>",
        unsafe_allow_html=True,
    )
    st.text("")
    p_title("Analyze charter")
    st.text("")

    source = st.radio(
        "How would you like to start?",
        ("I want to input some text", "I want to upload a file"),
    )
    st.text("")

    s_example = (
        "In nomine sancte et indiuidue Trinitatis , Amen . Ludovicus dei gratia Francorum Rex . "
        "Quoniam vita hominum brevis et memoria labilis est dignum est ut controversie que mediantibus "
        "sapientibus viris seu transactione seu pacifica compositione sopita sunt , scriptis insinuentur , "
        "ne in posterum recidivas pariant questiones . Notum igitur facimus universis praesentibus et futuris "
        "quod controversia que inter ecclesiam Beati Dionysii et dominum Paganum de Praellis super nemoribus "
        "de Roseai divitius agitata fuerat , aspirante divina gratia hunc finem sortita est ."
    )

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
                    # NER + classification + discourse parts
                    try:
                        st.markdown("___")
                        st.write("Named entities in Flat mode")
                        st.write(ner_html(input_su), unsafe_allow_html=True)

                        st.markdown("___")
                        st.success(class_acta(input_su))

                        st.markdown("___")
                        st.write("Diplomatics parts")
                        st.write(parts_dis_html(input_su), unsafe_allow_html=True)

                    except Exception as e:
                        st.exception(e)

    if source == "I want to upload a file":
        file = st.file_uploader("Upload your file here", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if file is not None:
            with st.spinner("Processing..."):
                try:
                    img = read_image(file)
                    st.image(img, width=350)

                    seg = segmentation(img)
                    st.write(transcript(img, seg), unsafe_allow_html=True)
                    st.success("Done")

                except Exception as e:
                    st.exception(e)


# --------------------------------------------------------------------------------------
if nav == "Handwritten text recognition":
    st.markdown(
        "<h4 style='text-align:center; color:grey;'>Kraken pipeline (segmentation + recognition)</h4>",
        unsafe_allow_html=True,
    )
    st.text("")
    p_title("Handwritten text recognition")
    st.text("")

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if file is not None:
        with st.spinner("Processing..."):
            try:
                img = read_image(file)
                st.image(img, width=450)
                seg = segmentation(img)
                st.write(transcript(img, seg), unsafe_allow_html=True)
            except Exception as e:
                st.exception(e)


# --------------------------------------------------------------------------------------
if nav == "Paraphrase text":
    st.markdown(
        "<h4 style='text-align:center; color:grey;'>Paraphrase utilities</h4>",
        unsafe_allow_html=True,
    )
    st.text("")
    p_title("Paraphrase")
    st.text("")

    p_example = (
        "Health is the level of functional or metabolic efficiency of a living organism. "
        "In humans, it is the ability of individuals or communities to adapt and self-manage when facing "
        "physical, mental, or social challenges."
    )

    input_pa = st.text_area(
        "Input your own text in English (max 500 chars)",
        max_chars=500,
        value=p_example,
        height=160,
    )

    if st.button("Paraphrase"):
        if not input_pa.strip():
            st.error("Please enter some text")
        else:
            with st.spinner("Processing..."):
                try:
                    from googletrans import Translator
                    from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter

                    translator = Translator()
                    mid = translator.translate(input_pa, dest="fr").text
                    mid2 = translator.translate(mid, dest="de").text
                    back = translator.translate(mid2, dest="en").text

                    st.markdown("___")
                    st.write("Back Translation Model")
                    st.success(back)

                    e_augmenter = EmbeddingAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
                    e_a = e_augmenter.augment(input_pa)

                    st.markdown("___")
                    st.write("Embedding Augmenter Model")
                    st.success(e_a)

                    w_augmenter = WordNetAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
                    w_a = w_augmenter.augment(input_pa)

                    st.markdown("___")
                    st.write("WordNet Augmenter Model")
                    st.success(w_a)

                except Exception as e:
                    st.exception(e)


# --------------------------------------------------------------------------------------
if nav == "Analyze text":
    st.markdown(
        "<h4 style='text-align:center; color:grey;'>Basic readability stats</h4>",
        unsafe_allow_html=True,
    )
    st.text("")
    p_title("Analyze text")
    st.text("")

    a_example = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural "
        "intelligence displayed by humans or animals."
    )

    source = st.radio(
        "How would you like to start?",
        ("I want to input some text", "I want to upload a file"),
    )
    st.text("")

    def _analyze_text(raw: str):
        import nltk
        import readtime
        import textstat
        from nltk.tokenize import word_tokenize

        nltk.download("punkt", quiet=True)

        rt = readtime.of_text(raw)
        tc = textstat.flesch_reading_ease(raw)

        tokenized_words = word_tokenize(raw)
        lr = (len(set(tokenized_words)) / max(len(tokenized_words), 1)) if tokenized_words else 0.0
        lr = round(lr, 2)

        n_s = textstat.sentence_count(raw)

        st.markdown("___")
        st.text("Reading Time")
        st.write(rt)

        st.markdown("___")
        st.text("Text Complexity (Flesch Reading Ease)")
        st.write(tc)

        st.markdown("___")
        st.text("Lexical Richness (distinct words / total words)")
        st.write(lr)

        st.markdown("___")
        st.text("Number of sentences")
        st.write(n_s)

    if source == "I want to input some text":
        input_me = st.text_area(
            "Input your own text in English (max 10,000 chars)",
            max_chars=10000,
            value=a_example,
            height=330,
        )
        if st.button("Analyze"):
            if len(input_me) > 10000:
                st.error("Please enter a text of maximum 10,000 characters")
            else:
                with st.spinner("Processing..."):
                    try:
                        _analyze_text(input_me)
                    except Exception as e:
                        st.exception(e)

    if source == "I want to upload a file":
        file = st.file_uploader("Upload a .txt file", type=["txt"])
        if file is not None:
            with st.spinner("Processing..."):
                try:
                    stringio = StringIO(file.getvalue().decode("utf-8", errors="replace"))
                    string_data = stringio.read()
                    if len(string_data) > 10000:
                        st.error("Please upload a file of maximum 10,000 characters")
                    else:
                        _analyze_text(string_data)
                except Exception as e:
                    st.exception(e)
