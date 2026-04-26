# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ===============================
# PART A: DATA GENERATION (150)
# ===============================
np.random.seed(23)

courses = [
    'DSA','ML','AI','DBMS','OS','CN',
    'Python','Java','WebDev','Cloud'
]

transactions = []

for _ in range(150):
    selected = list(np.random.choice(courses, 2, replace=False))
    
    # 🔥 Inject strong patterns (IMPORTANT)
    if 'ML' in selected:
        selected.append('Python')
    if 'AI' in selected:
        selected.append('ML')
    if 'DSA' in selected:
        selected.append('DBMS')
    
    transactions.append(list(set(selected)))

df = pd.DataFrame({
    'StudentID': range(1,151),
    'Courses': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nTransactions Example:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_courses = [c for sub in transactions for c in sub]
course_counts = pd.Series(all_courses).value_counts()

print("\nTop Courses:\n", course_counts.head(10))

# Bar chart
course_counts.head(10).plot(kind='bar')
plt.title("Top 10 Courses")
plt.xlabel("Courses")
plt.ylabel("Frequency")
plt.show()

# ===============================
# PART B: APRIORI
# ===============================
te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df_encoded, min_support=0.03, use_colnames=True)

# Fix compatibility issue
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop 5 Rules:")
print(top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules")
plt.show()

# ===============================
# INTERPRETATION (5 RULES)
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Students prefer related technical courses together such as ML-Python, AI-ML, and DSA-DBMS.")