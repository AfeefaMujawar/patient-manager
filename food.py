import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv", encoding='latin1')
df.columns = df.columns.str.strip()
df = df.dropna()

# --- Part A ---
# Create transaction items
df['Item1'] = df['Type_of_order']
df['Item2'] = df['Vehicle_condition'].astype(str)
df['Item3'] = df['Road_traffic_density']

# Convert to transaction format
transactions = df[['Item1','Item2','Item3']]

basket = pd.get_dummies(transactions.stack()).groupby(level=0).sum()
basket = basket > 0

# Top items
top_items = transactions.stack().value_counts().head(10)

top_items.plot(kind='bar')
plt.title("Top Items")
plt.show()

# --- Part B ---
frequent_items = apriori(basket, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.30)

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values('lift', ascending=False).head(5)

    print("\nTop Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']])

    print("\nInterpretation:\n")
    for _, row in top_rules.iterrows():
        print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Visualization
    plt.scatter(rules['support'], rules['confidence'], s=rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()
else:
    print("No rules found")