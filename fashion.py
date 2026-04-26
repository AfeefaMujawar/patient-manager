import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset (with patterns) ---
items = ['T-Shirt','Jeans','Shirt','Jacket','Dress',
         'Skirt','Shorts','Sweater','Hoodie','Blazer']

data = []
for i in range(1, 201):
    base = random.sample(items, 2)

    # Inject patterns (important for rules)
    if 'T-Shirt' in base:
        base.append('Jeans')
    if 'Shirt' in base:
        base.append('Blazer')
    if 'Dress' in base:
        base.append('Jacket')

    basket = list(set(base + random.sample(items, 1)))

    for item in basket:
        data.append([i, item])

df = pd.DataFrame(data, columns=['TransactionID','ClothingItem'])

# --- Part A ---
df = df.dropna()

# One-hot encoding
basket = pd.crosstab(df['TransactionID'], df['ClothingItem'])
basket = basket > 0

# Top items
top_items = df['ClothingItem'].value_counts().head(10)

top_items.plot(kind='bar')
plt.title("Top Clothing Items")
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.show()

# --- Part B ---
frequent_items = apriori(basket, min_support=0.018, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.42)
rules = rules[rules['lift'] > 1]

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['lift','confidence'], ascending=False)

    print("\nStrong Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nFashion Buying Behavior:\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If customers buy {set(row['antecedents'])}, they also buy {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Visualization
    top_rules = top_rules.head(30)
    plt.scatter(top_rules['support'], top_rules['confidence'], s=top_rules['lift']*120, alpha=0.7)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Fashion Association Rules")
    plt.show()
else:
    print("No strong rules found")