import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📧 Spam Email Detection System")

# -----------------------------
# LOAD DATASET (THIS IS WHERE FILE NAME IS USED)
# -----------------------------
df = pd.read_csv(r"C:\Users\keert\Downloads\spamdet.xlsx")  # <-- YOUR FILE PATH

emails = df["text"]
labels = df["label"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# ACCURACY
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.title("📊 Model Info")
st.sidebar.metric("Accuracy", f"{accuracy * 100:.2f}%")

# -----------------------------
# INPUT UI
# -----------------------------
st.subheader("Enter Email Text")

user_input = st.text_area("Type email content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")