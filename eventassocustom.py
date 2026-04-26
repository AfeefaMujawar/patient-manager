# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ===============================
# PART A: DATA GENERATION (150)
# ===============================
np.random.seed(32)

events = [
    'TechFest','Workshop','Seminar','Hackathon',
    'CodingContest','Networking','Webinar',
    'StartupMeet','AIConference','GamingEvent'
]

transactions = []

for _ in range(150):
    attended = list(np.random.choice(events, 2, replace=False))
    
    # 🔥 Inject realistic event patterns
    if 'TechFest' in attended:
        attended.append('Workshop')
    if 'Hackathon' in attended:
        attended.append('CodingContest')
    if 'Seminar' in attended:
        attended.append('Networking')
    if 'AIConference' in attended:
        attended.append('Workshop')
    
    transactions.append(list(set(attended)))

df = pd.DataFrame({
    'ParticipantID': range(1,151),
    'EventsAttended': transactions
})

print("\nSample Data:\n", df.head())

# ===============================
# TRANSACTION FORMAT
# ===============================
print("\nSample Transactions:\n", transactions[:5])

# ===============================
# FREQUENCY ANALYSIS
# ===============================
all_events = [e for sub in transactions for e in sub]
event_counts = pd.Series(all_events).value_counts()

print("\nTop Events:\n", event_counts.head(10))

# Bar chart
event_counts.head(10).plot(kind='bar')
plt.title("Top Events Attended")
plt.xlabel("Events")
plt.ylabel("Frequency")
plt.show()

# ===============================
# PART B: APRIORI
# ===============================
te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Fix compatibility issue
freq_items['itemsets'] = freq_items['itemsets'].apply(
    lambda x: frozenset(str(i) for i in x)
)

print("\nFrequent Itemsets:\n", freq_items.head())

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.30)

print("\nRules:\n", rules[['antecedents','consequents','support','confidence','lift']])

# ===============================
# PART C: EVALUATION
# ===============================
top_rules = rules.sort_values(by=['lift','confidence'], ascending=False).head(5)

print("\nTop Event Combinations:\n",
      top_rules[['antecedents','consequents','confidence','lift']])

# Scatter plot
plt.scatter(rules['confidence'], rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.title("Association Rules (Events)")
plt.show()

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")

for _, r in top_rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])} "
          f"(confidence={r['confidence']:.2f}, lift={r['lift']:.2f})")

print("\nConclusion:")
print("Participants attend related events like Hackathon–CodingContest.")
print("Workshops are commonly linked with TechFest and AI events.")
print("Apriori helps organizers plan bundled or sequential events.")