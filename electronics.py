import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Better Synthetic Dataset (with patterns) ---
products = ['Laptop','Mouse','Keyboard','Headphones','Charger',
            'Monitor','USB Cable','Webcam','Printer','Tablet']

data = []
for i in range(1, 201):
    base = random.sample(products, 2)

    # Add patterns (important)
    if 'Laptop' in base:
        base.append('Mouse')
        base.append('Keyboard')
    if 'Printer' in base:
        base.append('USB Cable')

    items = list(set(base + random.sample(products, 1)))

    for item in items:
        data.append([i, item])

df = pd.DataFrame(data, columns=['OrderID','Product'])

# --- Part A ---
df = df.dropna()

top_products = df['Product'].value_counts().head(10)
top_products.plot(kind='bar')
plt.title("Top Products")
plt.show()

# Transaction format
basket = pd.crosstab(df['OrderID'], df['Product'])
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.40)

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['lift','confidence'], ascending=False)

    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    for _, row in top_rules.head(5).iterrows():
        print(f"{set(row['antecedents'])} → {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Better visualization
    plt.scatter(rules['support'], rules['confidence'], s=rules['lift']*150, alpha=0.7)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Improved Association Rules")
    plt.grid()
    plt.show()
else:
    print("No rules found")