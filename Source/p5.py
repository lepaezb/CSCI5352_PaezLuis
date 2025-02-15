import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from a GML file
G = nx.read_gml('Data/p5/medici_network.gml')

# Compute the harmonic centrality for the original graph.
centrality_data = nx.harmonic_centrality(G)

# Get a sorted list of nodes for consistent ordering in printing and plotting.
nodes = sorted(G.nodes())
n = len(nodes)
m = G.number_of_edges()

# Print the harmonic centrality for each node.
print("Node\tHarmonic Centrality")
for node in nodes:
    print(f"{node}\t{centrality_data[node]:.3f}")

# -------------------------------
# Part B: Double-Edge Swap Null Model
# -------------------------------
def experiment_double_edge_swap(G, nodes, centrality_data, m, num_random=1000):
    # Store the null differences for each node (C_i^null - C_i^data).
    null_diff = {node: [] for node in nodes}
    r = 20 * m  # double-edge swaps to produce a random graph

    for i in range(num_random):
        G_rand = G.copy()
        try:
            nx.double_edge_swap(G_rand, nswap=r, max_tries=r * 10)
        except nx.NetworkXError as e:
            print(f"Double edge swap failed on iteration {i}: {e}")
        # Compute the harmonic centrality for the randomized graph.
        centrality_rand = nx.harmonic_centrality(G_rand)
        # Record the difference for each node.
        for node in nodes:
            diff = centrality_rand[node] - centrality_data[node]
            null_diff[node].append(diff)

    # Prepare the data for boxplot.
    data_to_plot = [null_diff[node] for node in nodes]
    
    # Create the boxplot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data_to_plot, positions=range(len(nodes)), widths=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Node index (sorted order)')
    ax.set_ylabel('Null Harmonic Centrality Difference (C_null - C_data)')
    ax.set_title('Distribution of Harmonic Centrality Differences\n(Double-Edge Swap Null Model)')
    plt.xticks(range(len(nodes)), nodes, rotation=90)
    
    plt.tight_layout()
    plt.savefig("Results/p5/p5_b.png")
    plt.close()

# -------------------------------
# Part C: Stub-Matching (Configuration Model) Null Model
# -------------------------------
def experiment_stub_matching(G, nodes, centrality_data, num_random=1000):
    # Store the null differences for each node (C_i^null - C_i^data).
    null_diff = {node: [] for node in nodes}
    # Obtain the degree sequence for stub-matching, using the sorted node order.
    degree_seq = [G.degree(n) for n in nodes]

    for i in range(num_random):
        # Generate a stub-labeled loopy multigraph using the configuration model.
        G_stub = nx.configuration_model(degree_seq, create_using=nx.MultiGraph)
        # Simplify the graph by converting to a simple graph:
        # - Collapses multiple edges and removes self-loops.
        G_simple = nx.Graph(G_stub)
        G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
        # Relabel nodes so that node i corresponds to the i-th node in the sorted list.
        mapping = {i: nodes[i] for i in range(len(nodes))}
        G_simple = nx.relabel_nodes(G_simple, mapping)
        
        # Compute the harmonic centrality for the null model graph.
        centrality_rand = nx.harmonic_centrality(G_simple)
        # Record the difference for each node.
        for node in nodes:
            diff = centrality_rand[node] - centrality_data[node]
            null_diff[node].append(diff)

    # Prepare the data for boxplot.
    data_to_plot = [null_diff[node] for node in nodes]
    
    # Create the boxplot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data_to_plot, positions=range(len(nodes)), widths=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Node index (sorted order)')
    ax.set_ylabel('Null Harmonic Centrality Difference (C_null - C_data)')
    ax.set_title('Distribution of Harmonic Centrality Differences\n(Stub-Matching Null Model)')
    plt.xticks(range(len(nodes)), nodes, rotation=90)
    
    plt.tight_layout()
    plt.savefig("Results/p5/p5_c.png")
    plt.close()

# -------------------------------
# Run Both Experiments
# -------------------------------
experiment_double_edge_swap(G, nodes, centrality_data, m, num_random=1000)
experiment_stub_matching(G, nodes, centrality_data, num_random=1000)

# -------------------------------
# Plot the Graphs
# -------------------------------
# Generate a random graph using double-edge swaps (Experiment 1)
num_swaps = 20 * m
G_rand_swap = G.copy()
try:
    nx.double_edge_swap(G_rand_swap, nswap=num_swaps, max_tries=num_swaps * 10)
except nx.NetworkXError as e:
    print(f"Double edge swap failed during final plotting: {e}")

# Generate a random graph using stub-matching (Experiment 2)
degree_seq = [G.degree(n) for n in nodes]
G_stub = nx.configuration_model(degree_seq, create_using=nx.MultiGraph)
G_simple = nx.Graph(G_stub)
G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
mapping = {i: nodes[i] for i in range(len(nodes))}
G_rand_stub = nx.relabel_nodes(G_simple, mapping)

# Plot all three graphs side by side.
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the original graph.
pos_original = nx.kamada_kawai_layout(G)
nx.draw(G, pos_original, ax=axs[0], with_labels=True, node_size=300, font_size=8)
axs[0].set_title("Original Graph")

# Plot the random graph from double-edge swap.
pos_swap = nx.kamada_kawai_layout(G_rand_swap)
nx.draw(G_rand_swap, pos_swap, ax=axs[1], with_labels=True, node_size=300, font_size=8)
axs[1].set_title("Random Graph (Double-Edge Swap Algorithm)")

# Plot the random graph from stub-matching.
pos_stub = nx.kamada_kawai_layout(G_rand_stub)
nx.draw(G_rand_stub, pos_stub, ax=axs[2], with_labels=True, node_size=300, font_size=8)
axs[2].set_title("Random Graph (Stub-Matching Algorithm)")

plt.tight_layout()
plt.savefig("Results/p5/graphs.png")
plt.close()