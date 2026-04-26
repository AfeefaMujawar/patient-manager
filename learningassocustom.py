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
np.random.seed(24)

modules = [
    'Intro','Basics','Advanced','ML','AI',
    'Python','Projects','DataViz','Statistics','DeepLearning'
]

transactions = []

for _ in range(150):
    selected = list(np.random.choice(modules, 2, replace=False))
    
    # 🔥 Inject learning flow patterns
    if 'Intro' in selected:
        selected.append('Basics')
    if 'Basics' in selected:
        selected.append('Advanced')
    if 'ML' in selected:
        selected.append('Python')
    if 'AI' in selected:
        selected.append('ML')
    
    transactions.append(list(set(selected)))

df = pd.DataFrame({
    'UserID': range(1,151),
    'ModulesAccessed': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nTransactions Example:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_modules = [m for sub in transactions for m in sub]
module_counts = pd.Series(all_modules).value_counts()

print("\nTop Modules:\n", module_counts.head(10))

# Bar chart
module_counts.head(10).plot(kind='bar')
plt.title("Top Modules")
plt.xlabel("Modules")
plt.ylabel("Frequency")
plt.show()

# ===============================
# PART B: APRIORI
# ===============================
te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df_encoded, min_support=0.025, use_colnames=True)

# Fix compatibility issue
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.45)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Rules:\n", top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Users follow structured learning paths such as Intro → Basics → Advanced.")
print("Technical modules like ML, AI, and Python are strongly interconnected.")