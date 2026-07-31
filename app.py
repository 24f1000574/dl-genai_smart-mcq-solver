import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# Hugging Face Model Name
MODEL_NAME = "sagnik538/distilbert-mcq-solver"

# Streamlit Page Settings
st.set_page_config(
    page_title="AI MCQ Solver",
    layout="centered"
)

# Labels for the options
labels = ["A", "B", "C", "D", "E"]


# Load model and tokenizer only once
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


# Load the model
tokenizer, model = load_model()


# Function to predict the correct answer
def predict_answer(question, options):

    # Tokenize the question and all options
    inputs = tokenizer(
        [question] * 5,
        options,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Disable gradient calculation
    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"].unsqueeze(0),
            attention_mask=inputs["attention_mask"].unsqueeze(0)
        )

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    probabilities = probabilities.cpu().numpy()[0]

    predicted_index = np.argmax(probabilities)

    return predicted_index, probabilities


# Streamlit UI part 


st.title("AI MCQ Solver")

st.write(
    "Enter a question and five options. The model will predict the best answer."
)

# Question
question = st.text_area(
    "Question",
    height=180
)

# Options
option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")
option_e = st.text_input("Option E")


# Predict Button
if st.button("Predict"):

    # Check if all fields are filled
    if "" in [
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]:

        st.warning("Please enter the question and all five options.")

    else:

        options = [
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        ]

        prediction, probabilities = predict_answer(
            question,
            options
        )

        st.success(
            f"Predicted Answer: Option {labels[prediction]}"
        )

        st.write("---")

        st.subheader("Confidence Scores")

        # Display confidence score for each option
        for i in range(5):

            st.write(
                f"Option {labels[i]} : {probabilities[i] * 100:.2f}%"
            )

            st.progress(float(probabilities[i]))

        st.write("---")

        st.subheader("Top 3 Predictions")

        top_three = np.argsort(probabilities)[::-1][:3]

        for rank, index in enumerate(top_three, start=1):

            st.write(
                f"{rank}. Option {labels[index]} ({probabilities[index] * 100:.2f}%)"
            )
