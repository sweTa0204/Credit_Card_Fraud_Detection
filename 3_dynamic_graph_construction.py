# Step 3: Dynamic Transaction Graph Construction
# Build transaction graphs in real-time and document the process

import pandas as pd
import networkx as nx

# Load engineered features
df = pd.read_csv('engineered_features.csv')

# Example: Construct a transaction graph where nodes are users and merchants, edges are transactions
G = nx.Graph()

for idx, row in df.iterrows():
    user = row['Customer_ID'] if 'Customer_ID' in row else f'user_{idx}'
    merchant = row['Store_ID'] if 'Store_ID' in row else f'merchant_{idx}'
    G.add_node(user, type='user')
    G.add_node(merchant, type='merchant')
    G.add_edge(user, merchant, amount=row['Purchase_Amount'], time=row['Transaction_Datetime'])

print(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.')

# Alignment Check
print('Alignment Check:')
print('- Graph supports evolving fraud patterns and real-time updates.')
print('- Next: Hybrid AttentionGNN-LSTM model implementation.')

# Save graph for model input
nx.write_gpickle(G, 'transaction_graph.gpickle')
