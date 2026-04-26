import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
data = [
["Government passes new law in parliament", "Politics"],
["Election results declared by officials", "Politics"],
["Stock market hits record high today", "Business"],
["Company reports huge profit this quarter", "Business"],
["New smartphone launched with advanced features", "Tech"],
["AI technology is transforming industries", "Tech"],
["Team wins championship after thrilling match", "Sports"],
["Player scores century in cricket match", "Sports"],
["Health benefits of yoga and exercise", "Health"],
["Doctors discover new treatment for disease", "Health"]
]

df = pd.DataFrame(data, columns=['NewsText','Category'])

# --- Part A ---
# Text preprocessing
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['NewsText'] = df['NewsText'].apply(clean)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['NewsText'])

# Encode labels
df['Category'] = df['Category'].astype('category').cat.codes
y = df['Category']

# Category distribution
print("\nCategory Distribution:\n", df['Category'].value_counts())

# --- Part B ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Part C ---
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro', zero_division=1))
print("Recall:", recall_score(y_test, y_pred, average='macro', zero_division=1))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Interpretation
print("\nInterpretation:")
print("Model classifies news into categories based on textual patterns using TF-IDF features.")