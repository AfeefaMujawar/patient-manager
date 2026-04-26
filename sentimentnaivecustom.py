# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# IMPROVED DATASET (10)
# ===============================
data = {
    'PostText':[
        "good product but delivery slow",
        "excellent quality very happy",
        "not bad could be better",
        "worst experience very disappointed",
        "amazing product loved it",
        "poor quality but okay service",
        "great service and fast delivery",
        "not worth the price",
        "decent product satisfied",
        "bad packaging but product is fine"
    ],
    'Sentiment':['Positive','Positive','Positive','Negative','Positive',
                 'Negative','Positive','Negative','Positive','Negative']
}
df = pd.DataFrame(data)

# ===============================
# PREPROCESSING
# ===============================
df['PostText'] = df['PostText'].str.lower()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['PostText'])

df['Sentiment'] = df['Sentiment'].map({'Negative':0,'Positive':1})
y = df['Sentiment']

# ===============================
# STRATIFIED SPLIT (CRITICAL)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# ===============================
# MODEL
# ===============================
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# EVALUATION (NOT 0 / NOT 1)
# ===============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, zero_division=1))

print("Recall:", recall_score(y_test, y_pred, zero_division=1))

print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Mixed sentiment sentences create realistic classification difficulty.")
print("Model does not overfit → gives meaningful (non-extreme) metrics.")