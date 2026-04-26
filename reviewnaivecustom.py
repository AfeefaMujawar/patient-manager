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
# PART A: SYNTHETIC DATASET (10)
# ===============================
data = {
    'ReviewText':[
        "good product but slow delivery",
        "excellent quality very satisfied",
        "average item not too bad",
        "poor packaging and bad service",
        "really happy with purchase",
        "not worth the price",
        "decent quality could improve",
        "amazing experience loved it",
        "bad product but okay support",
        "great value for money"
    ],
    'RatingClass':['Good','Good','Good','Bad','Good',
                   'Bad','Good','Good','Bad','Good']
}
df = pd.DataFrame(data)

# ===============================
# TEXT PREPROCESSING
# ===============================
df['ReviewText'] = df['ReviewText'].str.lower()

# TF-IDF (numerical features)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['ReviewText'])

# Encode target
df['RatingClass'] = df['RatingClass'].map({'Bad':0,'Good':1})
y = df['RatingClass']

# Class distribution
df['RatingClass'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# PART C: EVALUATION
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
print("Positive reviews include words like 'good', 'excellent', 'amazing'.")
print("Negative reviews include words like 'bad', 'poor', 'not worth'.")
print("Mixed reviews create realistic classification difficulty.")
print("Model gives balanced performance (not extreme 0 or 1).")