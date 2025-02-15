import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Load the Berkeley13.txt Network
# ----------------------------
G_full = ig.Graph.Read_Edgelist("Data/p3/Berkeley13.txt", directed=False)
G_full.simplify(multiple=True, loops=True)
m = G_full.ecount()
n = G_full.vcount()
print("Original Full Graph: nodes =", n, "edges =", m)

# ----------------------------
# 2. Compute the full network metrics
# ----------------------------
# Clustering coefficient
C_orig = G_full.transitivity_undirected()

# Extract largest component
components = G_full.connected_components()
largest_cc = components.giant()

# Mean path length
l_orig = largest_cc.average_path_length()

# Print metrics
print("Original Graph:")
print("Clustering Coefficient: {:.4f}".format(C_orig))
print("Mean Path Length (Largest Component): {:.4f}".format(l_orig))

# ----------------------------
# 3. Double-Edge Swaps
# ----------------------------
# Number of swaps r from 0 up to 20*m, using an increasing step size.
num_points = 30 
r_values = np.unique(np.round(np.logspace(0, np.log10(20 * m), num=num_points)).astype(int))

C_values = []
l_values = []

for r in r_values:
    G_random = G_full.copy()
    
    # Perform r double-edge swaps on the full graph
    G_random.rewire(n=r, mode="simple")
    G_random.simplify(multiple=True, loops=True)  # Ensure edges are not duplicated post-rewiring
    
    # Compute the clustering coefficient
    C = G_random.transitivity_undirected()
    
    # Extract the largest connected component
    components = G_random.connected_components()
    largest_cc = components.giant()
    
    # Compute the mean shortest path length
    try:
        l = largest_cc.average_path_length()
    except:
        l = np.nan
    
    C_values.append(C)
    l_values.append(l)
    
    print("Swaps: {:d}, Clustering: {:.4f}, Mean Path Length: {:.4f}".format(r, C, l))

# The configuration model baseline is when r = 20*m.
C_config = C_values[-1]
l_config = l_values[-1]

# ----------------------------
# 4. Plot the Results
# ----------------------------
plt.figure(figsize=(12, 5))

# Plot for Clustering Coefficient
plt.subplot(1, 2, 1)
plt.xscale('log')
plt.plot(r_values, C_values, 'o-', label="C(r)")
plt.axhline(y=C_config, color='r', linestyle='--', label="Config Model C (r=20*m)")
plt.xlabel("Number of Double-Edge Swaps (r)")
plt.ylabel("Clustering Coefficient")
plt.title("Randomization of Clustering Coefficient")
plt.legend()

# Plot for Mean Path Length
plt.subplot(1, 2, 2)
plt.xscale('log')
plt.plot(r_values, l_values, 'o-', label="⟨l⟩(r)")
plt.axhline(y=l_config, color='r', linestyle='--', label="Config Model ⟨l⟩ (r=20*m)")
plt.xlabel("Number of Double-Edge Swaps (r)")
plt.ylabel("Mean Path Length")
plt.title("Randomization of Mean Path Length")
plt.legend()

plt.tight_layout()
plt.savefig("Results/p3/p3.png")
plt.close()