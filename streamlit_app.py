"""
streamlit_app.py — Ogu Sentence-Function Classifier (Streamlit Community Cloud)

A permanently hosted, free demo. Loads the pre-trained SVM baseline from a
file in this repo, and the fine-tuned AfroXLM-R model from a separate
Hugging Face model repo, and serves live predictions through a Streamlit
web interface.

Expected files alongside this script (in the same GitHub repo):
    svm_model.joblib      - the fitted TF-IDF vectorizer and SVM classifier

The AfroXLM-R model and tokenizer are downloaded automatically at startup
from:
    https://huggingface.co/Supremehoye/ogu-afroxlmr
"""
import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

AFROXLMR_REPO = "Supremehoye/ogu-afroxlmr"  # separate HF model repo


# ---------------------------------------------------------------- load models
# @st.cache_resource makes sure the models are loaded only ONCE, the first
# time the app starts, rather than reloading on every user interaction.
@st.cache_resource
def load_models():
    svm_bundle = joblib.load("svm_model.joblib")
    svm_vectorizer = svm_bundle["vectorizer"]
    svm_model = svm_bundle["model"]

    afro_tokenizer = AutoTokenizer.from_pretrained(AFROXLMR_REPO)
    afro_model = AutoModelForSequenceClassification.from_pretrained(AFROXLMR_REPO)
    afro_model.eval()
    id2label = afro_model.config.id2label

    return svm_vectorizer, svm_model, afro_tokenizer, afro_model, id2label


svm_vectorizer, svm_model, afro_tokenizer, afro_model, id2label = load_models()


# ---------------------------------------------------------------- prediction
def classify_sentence(sentence):
    if not sentence or not sentence.strip():
        return {}, {}

    # SVM baseline
    X = svm_vectorizer.transform([sentence])
    svm_proba = svm_model.predict_proba(X)[0]
    svm_result = {cls: float(p) for cls, p in zip(svm_model.classes_, svm_proba)}

    # AfroXLM-R primary model
    inputs = afro_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = afro_model(**inputs).logits
    afro_proba = torch.softmax(logits, dim=-1)[0].tolist()
    afro_result = {id2label[i]: float(p) for i, p in enumerate(afro_proba)}

    return svm_result, afro_result


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="Ogu Sentence-Function Classifier")
st.title("Ogu Sentence-Function Classifier")
st.write(
    "Type any Ogu sentence and see how both models classify it across the "
    "four functional classes: Declarative, Interrogative, Imperative, Exclamatory."
)

sentence = st.text_input("Ogu sentence", placeholder="Type an Ogu sentence here...")

if sentence and sentence.strip():
    svm_result, afro_result = classify_sentence(sentence)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SVM baseline")
        for label, prob in sorted(svm_result.items(), key=lambda x: -x[1]):
            st.write(f"{label}: {prob:.1%}")
            st.progress(prob)

    with col2:
        st.subheader("AfroXLM-R (primary)")
        for label, prob in sorted(afro_result.items(), key=lambda x: -x[1]):
            st.write(f"{label}: {prob:.1%}")
            st.progress(prob)
