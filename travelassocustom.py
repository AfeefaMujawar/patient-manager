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
np.random.seed(30)

services = [
    'Flight','Hotel','Cab','Guide',
    'TourPackage','Insurance','Meal','Train','Bus','CarRental'
]

transactions = []

for _ in range(150):
    selected = list(np.random.choice(services, 2, replace=False))
    
    # 🔥 Inject realistic travel bundles
    if 'Flight' in selected:
        selected.append('Hotel')
    if 'Hotel' in selected:
        selected.append('Cab')
    if 'TourPackage' in selected:
        selected.append('Guide')
    if 'CarRental' in selected:
        selected.append('Insurance')
    
    transactions.append(list(set(selected)))

df = pd.DataFrame({
    'CustomerID': range(1,151),
    'SelectedServices': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_services = [s for sub in transactions for s in sub]
service_counts = pd.Series(all_services).value_counts()

print("\nTop Services:\n", service_counts.head(10))

# Bar chart
service_counts.head(10).plot(kind='bar')
plt.title("Top Services")
plt.xlabel("Services")
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
rules = association_rules(freq_items, metric="confidence", min_threshold=0.42)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Service Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Travel Services)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Customers prefer bundled services like Flight–Hotel–Cab.")
print("Tour packages are strongly associated with guides.")
print("Apriori helps travel platforms design personalized packages.")