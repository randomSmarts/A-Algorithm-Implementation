import math
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# Define the graph as a dictionary with neighbors and weights
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'C': 3},
    'C': {'D': 4},
    'D': {'E': 1},
    'E': {}
}

# Coordinates for visualization and heuristic calculation
coordinates = {
    'A': (0, 0),
    'B': (2, 0),
    'C': (5, 1),
    'D': (7, 2),
    'E': (8, 2)
}

# Heuristic function (Euclidean distance)
def heuristic(node, goal):
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# A* Algorithm
def a_star_search(graph, start, goal):
    open_set = {start}  # Nodes to be explored
    came_from = {}  # To reconstruct the path

    # Initialize g(n) and f(n)
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_set:
        # Node with the lowest f(n)
        current = min(open_set, key=lambda node: f_score[node])

        # If the goal is reached, reconstruct the path
        if current == goal:
            return reconstruct_path(came_from, current), f_score[goal]

        open_set.remove(current)

        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                # Update path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None, float('inf')  # Return None if no path is found

# Reconstruct the path from start to goal
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Run the algorithm
path, total_cost = a_star_search(graph, 'A', 'E')

# Output the results
print("Path:", path)
print("Total Cost:", total_cost)

# Visualization
# Enhanced Visualization
def visualize_graph(graph, path, coordinates):
    G = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = coordinates  # Node positions
    labels = nx.get_edge_attributes(G, 'weight')  # Edge weights

    # Set up the figure
    plt.figure(figsize=(10, 8))
    plt.title("Graph Representation with A* Path Highlighted", fontsize=16, fontweight='bold')

    # Draw the graph with enhancements
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        node_color='skyblue',
        font_size=12,
        font_weight='bold',
        edge_color='gray',
        width=2
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10, font_color='black')

    # Highlight the shortest path
    if path:
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges_in_path,
            edge_color='red', width=3, style='solid'
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=path,
            node_color='orange', node_size=1200
        )

    # Create a custom legend
    legend_elements = [
        mpatches.Patch(color='skyblue', label='Explored Nodes'),
        mpatches.Patch(color='orange', label='Shortest Path Nodes'),
        mpatches.Patch(color='gray', label='Explored Edges', linewidth=2),
        mpatches.Patch(color='red', label='Shortest Path Edges', linewidth=2)
    ]
    plt.legend(
        handles=legend_elements,
        loc='upper left', fontsize=10, frameon=True, title='Legend', title_fontsize='12'
    )

    plt.axis('off')  # Turn off axes for a clean graph
    plt.show()

# Visualize the graph with the improved design
visualize_graph(graph, path, coordinates)