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
np.random.seed(27)

apps = [
    'WhatsApp','Instagram','Facebook','YouTube',
    'Netflix','Spotify','Twitter','Snapchat',
    'Telegram','Gmail','Chrome','Maps'
]

transactions = []

for _ in range(150):
    used = list(np.random.choice(apps, 2, replace=False))
    
    # 🔥 Inject realistic usage patterns
    if 'Instagram' in used:
        used.append('Facebook')
    if 'YouTube' in used:
        used.append('Chrome')
    if 'Netflix' in used:
        used.append('Spotify')
    if 'WhatsApp' in used:
        used.append('Telegram')
    
    transactions.append(list(set(used)))

df = pd.DataFrame({
    'UserID': range(1,151),
    'AppsUsed': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_apps = [a for sub in transactions for a in sub]
app_counts = pd.Series(all_apps).value_counts()

print("\nTop Apps:\n", app_counts.head(10))

# Plot
app_counts.head(10).plot(kind='bar')
plt.title("Top Apps Used")
plt.xlabel("Apps")
plt.ylabel("Frequency")
plt.show()

# ===============================
# PART B: APRIORI
# ===============================
te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df_encoded, min_support=0.015, use_colnames=True)

# Fix compatibility issue
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.30)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop App Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (App Usage)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Users tend to use related apps together (e.g., Instagram–Facebook, YouTube–Chrome).")
print("Entertainment apps (Netflix–Spotify) show strong co-usage.")
print("Apriori helps identify app bundles for recommendations.")