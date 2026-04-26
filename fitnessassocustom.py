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
np.random.seed(26)

activities = [
    'Running','Cycling','Yoga','WeightTraining',
    'Cardio','Zumba','Swimming','HIIT','Stretching','Pilates'
]

transactions = []

for _ in range(150):
    act = list(np.random.choice(activities, 2, replace=False))
    
    # 🔥 Inject realistic workout combos
    if 'WeightTraining' in act:
        act.append('Cardio')
    if 'Running' in act:
        act.append('Stretching')
    if 'Yoga' in act:
        act.append('Pilates')
    if 'HIIT' in act:
        act.append('Cardio')
    
    transactions.append(list(set(act)))

df = pd.DataFrame({
    'UserID': range(1,151),
    'Activities': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_acts = [a for sub in transactions for a in sub]
act_counts = pd.Series(all_acts).value_counts()

print("\nTop Activities:\n", act_counts.head(10))

# Plot frequency
act_counts.head(10).plot(kind='bar')
plt.title("Top Activities")
plt.xlabel("Activities")
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
rules = association_rules(freq_items, metric="confidence", min_threshold=0.35)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Activity Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Fitness Activities)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Users prefer structured workouts like Strength + Cardio.")
print("Flexibility exercises (Yoga, Pilates) are strongly linked.")
print("Apriori helps design personalized fitness programs.")