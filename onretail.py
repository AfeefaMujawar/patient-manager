import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("OnlineRetail.csv", encoding='latin1')

# --- Part A: Preprocessing & Exploration ---
# Remove cancelled + invalid
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]

# Reduce size (important for Apriori to run)
top_items = df['Description'].value_counts().head(100).index
df = df[df['Description'].isin(top_items)]

# Most frequent products
top_products = df['Description'].value_counts().head(10)

# Bar chart
top_products.plot(kind='bar')
plt.title("Top 10 Products")
plt.xlabel("Products")
plt.ylabel("Frequency")
plt.show()

# Transaction format
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket > 0   # boolean

# --- Part B: Model Building ---
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.40)
rules = rules[rules['lift'] > 1]

# --- Part C: Evaluation & Interpretation ---
if rules.empty:
    print("No strong rules found. (Data filtered for performance)")
else:
    top_rules = rules.sort_values(['confidence','lift'], ascending=False)

    print("\nTop Rules:\n")
    print(top_rules[['antecedents','consequents','support','confidence','lift']].head())

    print("\nInterpretation (Top 5 Rules):\n")
    for _, row in top_rules.head(5).iterrows():
        print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
              f"(Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

    # Scatter plot
   # Take only top rules (clean view)
top_rules = rules.sort_values(['confidence','lift'], ascending=False).head(30)

plt.scatter(top_rules['support'], top_rules['confidence'],
            s=top_rules['lift']*100, alpha=0.6)

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Top Association Rules")
plt.grid()
plt.show()