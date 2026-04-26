import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Create Synthetic Dataset ---
meds = ['Paracetamol','Aspirin','Ibuprofen','Amoxicillin','Cetrizine',
        'Azithromycin','Metformin','Atorvastatin','Pantoprazole','VitaminC']

data = []
for i in range(1, 201):   # 200 bills
    items = random.sample(meds, random.randint(2,5))
    for item in items:
        data.append([i, item])

df = pd.DataFrame(data, columns=['BillNo','MedicineName'])

# --- Part A ---
df = df.dropna()

# Transaction list format (display)
transactions = df.groupby('BillNo')['MedicineName'].apply(list)
print("\nSample Transactions:\n", transactions.head())

# Top medicines
top_meds = df['MedicineName'].value_counts().head(10)

top_meds.plot(kind='bar')
plt.title("Top Medicines")
plt.xlabel("Medicine")
plt.ylabel("Frequency")
plt.show()

# One-hot encoding
basket = pd.crosstab(df['BillNo'], df['MedicineName'])
basket = basket > 0

# --- Part B ---
frequent_items = apriori(basket, min_support=0.025, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.45)
rules = rules[rules['lift'] > 1]

# --- Part C ---
if not rules.empty:
    top_rules = rules.sort_values(['confidence','lift'], ascending=False)

    print("\nStrong Medicine Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nInterpretation (Top 5):\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
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