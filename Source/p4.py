import os
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

# Define input and output directories:
input_folder = "Data/p4/"
output_folder = "Results/p4/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Helper function to plot an igraph graph on a given matplotlib axis:
def plot_igraph_on_ax(G, ax, title):
    layout = G.layout("fr")
    coords = np.array(layout)
    
    # Draw all edges:
    for edge in G.get_edgelist():
        x0, y0 = coords[edge[0]]
        x1, y1 = coords[edge[1]]
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5)
        
    # Draw vertices:
    ax.scatter(coords[:, 0], coords[:, 1], color="blue", s=10)
    ax.set_title(title)
    ax.set_axis_off()

# Loop through all files in the input folder:
for filename in os.listdir(input_folder):
    if filename.endswith(".gml") or filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        
        # Read the graph based on its extension:
        if filename.endswith(".gml"):
            G_orig = ig.Graph.Read_GML(file_path)
        elif filename.endswith("foodweb.txt"):
            G_orig = ig.Graph.Read_Edgelist(file_path, directed=True)
        elif filename.endswith("slavko.txt"):
            G_orig = ig.Graph.Read_Edgelist(file_path, directed=False)
        
        # Ensure the graph is undirected and simple:
        if G_orig.is_directed():
            G_orig.as_undirected(mode="collapse")
        G_orig.simplify(multiple=True, loops=True)
        
        m = G_orig.ecount()
        n = G_orig.vcount()
        print(f"Network {filename}: nodes = {n}, edges = {m}")
        
        # Parameters for the experiment:
        num_samples = 1000           # number of configuration model graphs to generate
        num_swaps = 20 * m           # number of double-edge swaps to perform
        
        # Compute empirical measures:
        try:
            empirical_clustering = G_orig.transitivity_undirected()
        except Exception:
            empirical_clustering = np.nan
        try:
            empirical_path_length = G_orig.average_path_length()
        except Exception:
            empirical_path_length = np.nan
        
        # Lists to store null distribution values:
        clustering_vals = []
        path_length_vals = []
        
        # Generate null distribution by randomizing the graph (preserving degree sequence):
        for i in range(num_samples):
            G_rand = G_orig.copy()
            G_rand.rewire(n=num_swaps, mode="simple")
            try:
                clustering_vals.append(G_rand.transitivity_undirected())
            except Exception:
                clustering_vals.append(np.nan)
            try:
                path_length_vals.append(G_rand.average_path_length())
            except Exception:
                path_length_vals.append(np.nan)
        
        # Plot the null distributions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].hist(clustering_vals, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
        axes[0].axvline(empirical_clustering, color='red', linestyle='dashed', linewidth=2,
                        label=f'Empirical C = {empirical_clustering:.3f}')
        axes[0].set_title("Null Distribution of Clustering Coefficient (C)")
        axes[0].set_xlabel("Clustering Coefficient")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        
        axes[1].hist(path_length_vals, bins=30, color='lightgreen', edgecolor='black', alpha=0.8)
        axes[1].axvline(empirical_path_length, color='red', linestyle='dashed', linewidth=2,
                        label=(r'Empirical ($\langle \ell \rangle$) =' f'{empirical_path_length:.3f}'))
        axes[1].set_title(r"Null Distribution of Mean Path Length ($\langle \ell \rangle$)")
        axes[1].set_xlabel("Mean Path Length")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        
        plt.tight_layout()
        # Save the null distribution figure:
        null_fig_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_null.png")
        plt.savefig(null_fig_path)
        plt.close()
        
        # Generate a single configuration model instance:
        G_conf = G_orig.copy()
        G_conf.rewire(n=num_swaps, mode="simple")
        
        # Plot the original graph next to its configuration model:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
        plot_igraph_on_ax(G_orig, axes2[0], "Original Graph")
        plot_igraph_on_ax(G_conf, axes2[1], "Configuration Model Graph")
        
        plt.tight_layout()
        # Save the comparison figure:
        comp_fig_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_comparison.png")
        plt.savefig(comp_fig_path)
        plt.close()