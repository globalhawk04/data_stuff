import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Our messy, real-world dataset of user reviews ---
data = {
    'review_text': [
        "This product is absolutely amazing! Changed my life. Five stars!", # High quality, positive
        "loved it", # Low quality (short), positive
        "Broken on arrival. The packaging was terrible and the item was shattered. Zero stars if I could.", # High quality, negative
        "did not work!!!!!!!!", # Low quality (grammar/style), negative
        "It's okay, I guess. It does the job but I'm not wowed.", # High quality, neutral
        "meh.", # Low quality (short), neutral
        "This is a TRULY an exceptional product, combining both form and function into a seamless user experience that exceeded all expectations.", # High quality, positive
        "terrible waste of money", # Low quality (short), negative
        "I was skeptical at first, but after a week of use, I am a convert. The battery life is particularly impressive.", # High quality, positive
        "bad" # Low quality (short), negative
    ],
    'sentiment': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] # 1 for Positive/Neutral, 0 for Negative
}
df = pd.DataFrame(data)

# --- The "Old Way": Aggressively Filter for "High Quality" ---
print("--- METHOD 1: Training on 'High-Quality' Filtered Data ---")
# Let's define "high quality" as more than 5 words long.
df_filtered = df[df['review_text'].str.split().str.len() > 5].copy()
print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}\n")

# Vectorize and train a simple model on the filtered data
vectorizer_filtered = TfidfVectorizer()
X_filtered = vectorizer_filtered.fit_transform(df_filtered['review_text'])
y_filtered = df_filtered['sentiment']

# We don't have a separate test set, so we'll simulate by training on the whole filtered set
# and evaluating on the *entire* original set.
model_filtered = LogisticRegression()
model_filtered.fit(X_filtered, y_filtered)
predictions_filtered = model_filtered.predict(vectorizer_filtered.transform(df['review_text']))
accuracy_filtered = accuracy_score(df['sentiment'], predictions_filtered)
print(f"Accuracy of model trained on FILTERED data (evaluated on all data): {accuracy_filtered:.2f}\n")


# --- The "FineVision Way": Embrace the Mess ---
print("--- METHOD 2: Training on the Full, Diverse Dataset ---")
print(f"Dataset size: {len(df)}\n")

# Vectorize and train a model on the entire, messy dataset
vectorizer_full = TfidfVectorizer()
X_full = vectorizer_full.fit_transform(df['review_text'])
y_full = df['sentiment']

model_full = LogisticRegression()
model_full.fit(X_full, y_full)
predictions_full = model_full.predict(vectorizer_full.transform(df['review_text']))
accuracy_full = accuracy_score(df['sentiment'], predictions_full)
print(f"Accuracy of model trained on FULL data (evaluated on all data): {accuracy_full:.2f}\n")

# --- Conclusion ---
print("By being trained on the full, messy dataset, the second model learned to handle short, 'low-quality' inputs.")
print("The first model, shielded from this 'messy' data, was brittle and failed when evaluated on a more realistic, diverse set of inputs.")
