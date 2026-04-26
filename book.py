import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
books = ['Harry Potter','The Hobbit','1984','The Alchemist','Atomic Habits',
         'Rich Dad Poor Dad','Ikigai','The Power of Now','Think and Grow Rich','Sapiens']

data = []
for i in range(1, 201):   # 200 transactions
    items = random.sample(books, random.randint(2,5))
    for item in items:
        data.append([i, item])

df = pd.DataFrame(data, columns=['TransactionID','BookTitle'])

# --- Part A ---
# Transaction format (list view)
transactions = df.groupby('TransactionID')['BookTitle'].apply(list)
print("\nSample Transactions:\n", transactions.head())

# Popular books
top_books = df['BookTitle'].value_counts().head(10)

# Visualization
top_books.plot(kind='bar')
plt.title("Top 10 Books")
plt.xlabel("Books")
plt.ylabel("Frequency")
plt.show()

# One-hot encoding
basket = pd.crosstab(df['TransactionID'], df['BookTitle'])
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.55)
rules = rules[rules['lift'] > 1]

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['lift','confidence'], ascending=False)

    print("\nTop Rules (by Lift):\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nInterpretation (Top 5):\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If readers choose {set(row['antecedents'])} they also choose {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Visualization
    top_rules = top_rules.head(30)
    plt.scatter(top_rules['support'], top_rules['confidence'], s=top_rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()
else:
    print("No strong rules found")