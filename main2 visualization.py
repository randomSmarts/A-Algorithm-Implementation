import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
edges = [
    ('A', 'B', 101.9), ('A', 'C', 154.6),
    ('B', 'D', 75.3), ('C', 'D', 51.7),
    ('D', 'E', 131.6)
]

for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Highlight the optimal path
optimal_path = path
path_edges = list(zip(optimal_path, optimal_path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

plt.title("Flight Network with Optimal Path")
plt.show()