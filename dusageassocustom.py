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
np.random.seed(31)

devices = [
    'Smartphone','Laptop','Tablet','Smartwatch',
    'Desktop','SmartTV','GamingConsole','Headphones','Printer','Camera'
]

transactions = []

for _ in range(150):
    used = list(np.random.choice(devices, 2, replace=False))
    
    # 🔥 Inject realistic device ecosystems
    if 'Smartphone' in used:
        used.append('Smartwatch')
    if 'Laptop' in used:
        used.append('Tablet')
    if 'Desktop' in used:
        used.append('Printer')
    if 'GamingConsole' in used:
        used.append('SmartTV')
    
    transactions.append(list(set(used)))

df = pd.DataFrame({
    'UserID': range(1,151),
    'DevicesUsed': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_devices = [d for sub in transactions for d in sub]
device_counts = pd.Series(all_devices).value_counts()

print("\nTop Devices:\n", device_counts.head(10))

# Plot
device_counts.head(10).plot(kind='bar')
plt.title("Top Devices Used")
plt.xlabel("Devices")
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
rules = association_rules(freq_items, metric="confidence", min_threshold=0.36)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Device Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Device Usage)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Users prefer connected ecosystems like Smartphone–Smartwatch.")
print("Work devices (Laptop–Tablet) show strong association.")
print("Apriori helps design device bundling and recommendations.")