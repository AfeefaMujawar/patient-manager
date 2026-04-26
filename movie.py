import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
movies = ['Inception','Titanic','Avatar','Interstellar','Joker',
          'Avengers','Matrix','Gladiator','Batman','Spiderman']

data = []
for i in range(1, 201):   # 200 users
    watched = random.sample(movies, random.randint(2,5))
    for m in watched:
        data.append([i, m])

df = pd.DataFrame(data, columns=['UserID','MovieTitle'])

# --- Part A ---
# User-wise transactions
transactions = df.groupby('UserID')['MovieTitle'].apply(list)
print("\nSample Transactions:\n", transactions.head())

# Most watched movies
top_movies = df['MovieTitle'].value_counts().head(10)

# Visualization
top_movies.plot(kind='bar')
plt.title("Top Movies")
plt.xlabel("Movies")
plt.ylabel("Frequency")
plt.show()

# One-hot encoding
basket = pd.crosstab(df['UserID'], df['MovieTitle'])
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.005, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.25)
rules = rules[rules['lift'] > 1]

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['lift','confidence'], ascending=False)

    print("\nStrong Movie Associations:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nInterpretation (Top 5):\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If users watch {set(row['antecedents'])}, they also watch {set(row['consequents'])} "
              f"(Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Visualization
    top_rules = top_rules.head(30)
    plt.scatter(top_rules['support'], top_rules['confidence'], s=top_rules['lift']*100, alpha=0.6)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Movie Association Rules")
    plt.show()
else:
    print("No strong rules found")