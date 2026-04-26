import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("SuperMarket Analysis.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df = df.dropna()

# --- Part A ---
# Create better transactions using Customer + Date
df['Transaction'] = df['Customer type'] + "_" + df['Date']

basket = pd.crosstab(df['Transaction'], df['Product line'])
basket = basket > 0

# Top categories
top_cat = df['Product line'].value_counts().head(10)
top_cat.plot(kind='bar')
plt.title("Top Product Categories")
plt.show()

# --- Part B ---
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.30)

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['confidence','lift'], ascending=False).head(5)

    print("\nStrong Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']])

    print("\nCustomer Buying Patterns:\n")
    for _, row in top_rules.iterrows():
        print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    plt.scatter(rules['support'], rules['confidence'], s=rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()
else:
    print("Still no rules → dataset limitation")