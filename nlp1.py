import re
import tkinter as tk
from tkinter import messagebox, ttk

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the dataset
print("Loading the dataset")
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None, nrows=1000000)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]
df['target'] = df['target'].replace(4, 1)

# Balance the dataset
print("Balancing the dataset")
positive_samples = df[df['target'] == 1]
negative_samples = df[df['target'] == 0]

# Adjust sampling based on available counts
if len(positive_samples) < len(negative_samples):
    df_balanced = pd.concat([positive_samples, negative_samples.sample(n=len(positive_samples), random_state=42)])
else:
    df_balanced = pd.concat([positive_samples.sample(n=len(negative_samples), random_state=42), negative_samples])
print(f"Balanced dataset size: {len(df_balanced)}")

# Initialize SentimentIntensityAnalyzer
print("Initializing Sentiment Intensity Analyzer")
sia = SentimentIntensityAnalyzer()

# Text preprocessing
print("Initiating Text Preprocessing")
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# Feature extraction
print("Initiating Feature Extraction")
def extract_features(text):
    features = {}
    processed_text = preprocess_text(text)

    sentiment = sia.polarity_scores(processed_text)
    features['sentiment_pos'] = sentiment['pos']
    features['sentiment_neg'] = sentiment['neg']
    features['sentiment_neu'] = sentiment['neu']
    features['sentiment_compound'] = sentiment['compound']
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['capital_word_ratio'] = sum(1 for word in text.split() if word.isupper()) / len(text.split())
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)

    return features

df_balanced['features'] = df_balanced['text'].apply(extract_features)

# Use a smaller subset of the dataset for initial testing
print("Using a smaller subset of the dataset for testing")
subset_size = 20000  # Adjust this value as needed
df_subset = df_balanced.sample(n=subset_size, random_state=42)

# Prepare data for training
print("Preparing and Splitting data for training")
X_subset = pd.DataFrame(df_subset['features'].tolist())
y_subset = df_subset['target']

X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_subset_scaled = scaler.fit_transform(X_train_subset)
X_test_subset_scaled = scaler.transform(X_test_subset)

# Hyperparameter tuning with RandomizedSearchCV
print("Initiating hyperparameter tuning for Random Forest Classifier")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Timing the random search
import time
start_time = time.time()
random_search.fit(X_train_subset_scaled, y_train_subset)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time for RandomizedSearchCV: {elapsed_time / 60:.2f} minutes")

# Use the best estimator
model = random_search.best_estimator_
model.fit(X_train_subset_scaled, y_train_subset)

# Evaluation on the subset
print("Evaluation Metrics on Subset")
y_pred_subset = model.predict(X_test_subset_scaled)
print("Subset Accuracy:", accuracy_score(y_test_subset, y_pred_subset))
print("Subset Classification Report:\n", classification_report(y_test_subset, y_pred_subset))

# Extrapolate the execution time for the full dataset based on subset time
estimated_full_time = elapsed_time * (len(df_balanced) / subset_size)
print(f"Estimated full dataset search time: {estimated_full_time / 60:.2f} minutes")

# Prepare the full dataset
print("Preparing the full dataset for final model training")
X = pd.DataFrame(df_balanced['features'].tolist())
y = df_balanced['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the final model on the full dataset using the best parameters
print("Training the final model on the full dataset")
model.fit(X_train_scaled, y_train)

# Final evaluation
print("Final Evaluation Metrics")
y_pred = model.predict(X_test_scaled)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Final Classification Report:\n", classification_report(y_test, y_pred))

# GUI for text analysis
def analyze_text():
    input_text = text_entry.get("1.0", tk.END).strip()
    if input_text:
        features = extract_features(input_text)
        features_df = pd.DataFrame([features])
        features_scaled = scaler.transform(features_df)
        prediction_proba = model.predict_proba(features_scaled)[0]
        sentiment = 'Positive' if prediction_proba[1] > 0.5 else 'Negative'
        confidence = max(prediction_proba)

        exclamation = "High" if features['exclamation_count'] > 1 else "Low"

        sarcasm_score = (features['sentiment_compound'] + 1) * features['capital_word_ratio'] * (features['exclamation_count'] + 1)
        sarcasm = "Likely" if sarcasm_score > 0.5 else "Unlikely"

        result_text = f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n"
        result_text += f"Exclamation: {exclamation}\n"
        result_text += f"Potential Sarcasm: {sarcasm}\n"
        result_text += f"Sentiment Scores:\n"
        result_text += f"  Positive: {features['sentiment_pos']:.2f}\n"
        result_text += f"  Negative: {features['sentiment_neg']:.2f}\n"
        result_text += f"  Neutral: {features['sentiment_neu']:.2f}\n"
        result_text += f"  Compound: {features['sentiment_compound']:.2f}"

        result_label.config(text=result_text, foreground='green' if sentiment == 'Positive' else 'red')
    else:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")

# Tkinter GUI setup
root = tk.Tk()
root.title("Text Analysis Tool")
root.geometry("600x500")
root.resizable(False, False)

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12))
style.configure('TLabel', font=('Helvetica', 12))
style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))

header_label = ttk.Label(root, text="Text Analysis Tool", style='Header.TLabel')
header_label.pack(pady=10)

text_frame = ttk.Frame(root)
text_frame.pack(pady=20)

text_label = ttk.Label(text_frame, text="Enter text to analyze:")
text_label.grid(row=0, column=0, padx=5, pady=5)

text_entry = tk.Text(text_frame, height=5, width=60, font=('Helvetica', 12))
text_entry.grid(row=1, column=0, padx=5, pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

analyze_button = ttk.Button(button_frame, text="Analyze Text", command=analyze_text)
analyze_button.grid(row=0, column=0, padx=5, pady=5)

result_frame = ttk.Frame(root)
result_frame.pack(pady=20)

result_label = ttk.Label(result_frame, text="Analysis Results: ", font=('Helvetica', 14))
result_label.grid(row=0, column=0, padx=5, pady=5)

print("Starting GUI...")
root.mainloop()
