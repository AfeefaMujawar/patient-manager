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
np.random.seed(28)

books = [
    'DataScience','MachineLearning','ArtificialIntelligence',
    'Python','SQL','DBMS','OperatingSystems',
    'ComputerNetworks','Algorithms','Statistics'
]

transactions = []

for _ in range(150):
    selected = list(np.random.choice(books, 2, replace=False))
    
    # 🔥 Inject strong academic patterns
    if 'MachineLearning' in selected:
        selected.append('Python')
    if 'ArtificialIntelligence' in selected:
        selected.append('MachineLearning')
    if 'DBMS' in selected:
        selected.append('SQL')
    if 'DataScience' in selected:
        selected.append('Statistics')
    
    transactions.append(list(set(selected)))

df = pd.DataFrame({
    'StudentID': range(1,151),
    'BooksBorrowed': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_books = [b for sub in transactions for b in sub]
book_counts = pd.Series(all_books).value_counts()

print("\nTop Borrowed Books:\n", book_counts.head(10))

# Bar chart
book_counts.head(10).plot(kind='bar')
plt.title("Top Borrowed Books")
plt.xlabel("Books")
plt.ylabel("Frequency")
plt.show()

# ===============================
# PART B: APRIORI
# ===============================
te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df_encoded, min_support=0.02, use_colnames=True)

# Fix compatibility issue
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.38)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Book Associations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Library)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Students borrow related academic books together (ML–Python, AI–ML).")
print("Database concepts (DBMS–SQL) show strong association.")
print("Apriori helps libraries recommend relevant books to students.")