import networkx as nx
import matplotlib.pyplot as plt


# Define the graph as an adjacency list with weights for cost, time, and preference.
graph = {
    'A': {'B': {'cost': 200, 'time': 3, 'preference': 5},
          'C': {'cost': 300, 'time': 4, 'preference': 4}},
    'B': {'D': {'cost': 150, 'time': 2, 'preference': 3}},
    'C': {'D': {'cost': 100, 'time': 1, 'preference': 2},
          'E': {'cost': 250, 'time': 3, 'preference': 6}},
    'D': {'E': {'cost': 250, 'time': 3, 'preference': 6}},
    'E': {}
}

# Define the weight factors for the composite weight calculation
alpha = 0.5  # Weight for cost
beta = 0.3   # Weight for time
gamma = 0.2  # Weight for preference

def calculate_weight(cost, time, preference):
    return alpha * cost + beta * time + gamma * preference

# Define the heuristic based on estimated cost and time
heuristic_estimates = {
    'A': 400,  # Estimated cost + time to reach E
    'B': 300,
    'C': 200,
    'D': 100,
    'E': 0
}

def heuristic(node):
    return heuristic_estimates[node]


import heapq


def a_star(graph, start, goal):
    # Priority queue for open set
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Track g(n), f(n), and visited nodes
    g_scores = {node: float('inf') for node in graph}
    g_scores[start] = 0

    f_scores = {node: float('inf') for node in graph}
    f_scores[start] = heuristic(start)

    came_from = {}  # To reconstruct the path

    while open_set:
        # Pop the node with the lowest f(n)
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_scores[goal]

        # Explore neighbors
        for neighbor, attributes in graph[current].items():
            weight = calculate_weight(attributes['cost'], attributes['time'], attributes['preference'])
            tentative_g_score = g_scores[current] + weight

            if tentative_g_score < g_scores[neighbor]:
                # Update g(n) and f(n)
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_scores[neighbor], neighbor))

    return None, float('inf')  # Return None if no path found

path, total_cost = a_star(graph, 'A', 'E')
print("Optimal Path:", path)
print("Total Cost:", total_cost)

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