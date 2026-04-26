import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
products = ['Bread','Milk','Eggs','Butter','Cheese','Juice','Apples','Bananas','Chicken','Rice']

data = []
for i in range(1, 201):   # 200 orders
    items = random.sample(products, random.randint(2,5))
    for item in items:
        data.append([i, item])

df = pd.DataFrame(data, columns=['OrderID','ProductName'])

# --- Part A ---
df = df.dropna()

top_products = df['ProductName'].value_counts().head(10)

top_products.plot(kind='bar')
plt.title("Top Selling Products")
plt.xlabel("Products")
plt.ylabel("Frequency")
plt.show()

# Transaction format
basket = df.groupby(['OrderID','ProductName'])['ProductName'].count().unstack().fillna(0)
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.015, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.35)
rules = rules[rules['lift'] > 1]

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['confidence','lift'], ascending=False)

    print("\nTop Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nInterpretation (Top 5 Rules):\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Scatter plot
    top_rules = top_rules.head(30)
    plt.scatter(top_rules['support'], top_rules['confidence'], s=top_rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()
else:
    print("No rules found")