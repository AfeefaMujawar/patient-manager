import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("zomato.csv", encoding='latin1')
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Cuisines'])

# --- Part A ---
# Convert cuisines into transactions (split by comma)
df['Cuisines'] = df['Cuisines'].str.split(', ')

# Transaction format
transactions = df['Cuisines']
print("\nSample Transactions:\n", transactions.head())

# Top dishes (cuisines)
all_items = pd.Series([item for sublist in transactions for item in sublist])
top_items = all_items.value_counts().head(10)

top_items.plot(kind='bar')
plt.title("Top Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Frequency")
plt.show()

# One-hot encoding
basket = pd.get_dummies(transactions.apply(pd.Series).stack()).groupby(level=0).sum()
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.50)

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['lift','confidence'], ascending=False)

    print("\nTop Food Combinations:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nCustomer Preferences:\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If cuisine {set(row['antecedents'])} then {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Visualization
    top_rules = top_rules.head(30)
    plt.scatter(top_rules['support'], top_rules['confidence'], s=top_rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Cuisine Association Rules")
    plt.show()
else:
    print("No strong rules found → reduce support slightly")