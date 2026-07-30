import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL_NAME = "sagnik538/distilbert-mcq-solver"

st.set_page_config(
    page_title="AI MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

LABELS = ["A", "B", "C", "D", "E"]


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def predict(prompt, choices):

    encoded = tokenizer(
        [prompt] * 5,
        choices,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoded["input_ids"].unsqueeze(0),
            attention_mask=encoded["attention_mask"].unsqueeze(0)
        )

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    best = np.argmax(probs)

    return best, probs


st.title("🧠 AI MCQ Solver")

st.markdown(
    """
Predict the best answer for any Multiple Choice Question using a fine-tuned DistilBERT model.
"""
)

prompt = st.text_area(
    "Question",
    height=180
)

A = st.text_input("Option A")
B = st.text_input("Option B")
C = st.text_input("Option C")
D = st.text_input("Option D")
E = st.text_input("Option E")

if st.button("Predict Answer"):

    if "" in [prompt, A, B, C, D, E]:

        st.warning("Please fill all fields.")

    else:

        prediction, probs = predict(
            prompt,
            [A, B, C, D, E]
        )

        st.success(
            f"Predicted Answer : **{LABELS[prediction]}**"
        )

        st.subheader("Confidence Scores")

        for i in range(5):

            st.write(
                f"Option {LABELS[i]} : {probs[i]*100:.2f}%"
            )

            st.progress(float(probs[i]))

        st.subheader("Top 3 Predictions")

        top3 = np.argsort(probs)[::-1][:3]

        for rank, idx in enumerate(top3, start=1):

            st.write(
                f"{rank}. {LABELS[idx]} ({probs[idx]*100:.2f}%)"
            )