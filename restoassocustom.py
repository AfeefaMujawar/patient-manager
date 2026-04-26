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
np.random.seed(25)

menu = [
    'Burger','Fries','Pizza','Pasta','Coke',
    'Coffee','Sandwich','IceCream','Salad','Juice'
]

transactions = []

for _ in range(150):
    order = list(np.random.choice(menu, 2, replace=False))
    
    # 🔥 Inject strong combos (IMPORTANT)
    if 'Burger' in order:
        order.append('Fries')
        order.append('Coke')
    if 'Pizza' in order:
        order.append('Coke')
    if 'Coffee' in order:
        order.append('Sandwich')
    if 'Pasta' in order:
        order.append('Juice')
    
    transactions.append(list(set(order)))

df = pd.DataFrame({
    'OrderID': range(1,151),
    'Items': transactions
})

print("\nSample Orders:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_items = [i for sub in transactions for i in sub]
item_counts = pd.Series(all_items).value_counts()

print("\nTop Items:\n", item_counts.head(10))

# Plot top 10
item_counts.head(10).plot(kind='bar')
plt.title("Top 10 Ordered Items")
plt.xlabel("Items")
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

# Fix compatibility bug
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.40)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Food Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Food Items)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Customers prefer combo meals like Burger–Fries–Coke.")
print("Beverages are frequently paired with main dishes.")
print("Apriori helps identify popular food bundles for marketing.")