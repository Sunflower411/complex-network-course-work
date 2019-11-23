# In order to understand how the code works, it is a good idea to check the
# final section of the file that starts with
#   if __name__ == '__main__'
#
# Your task is essentially to replace all the parts marked as TODO or as
# instructed through comments in the functions, as well as the filenames
# and labels in the main part of the code which can be found at the end of
# this file.
#
# The raise command is used to help you out in finding where you still need to
# write your own code. When you successfully modified the code in that part,
# remove the `raise` command.
from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# ====================== FUNCTIONS USED BY THE MAIN CODE ===================
#
# ====================== FOR THE MAIN CODE SCROLL TO THE BOTTOM ============

def ring(n, m):
    """
    This function creates the basic ring (to be rewired) with n nodes
    in which each node is connected to m nodes on the left and right.

    Parameters
    ----------
    n : int
      Number of nodes
    m : int
      Number of neighbors to connect left and right

    Returns
    -------
    network : graph
             The basic ring before rewiring
    """
    network = nx.Graph()
    # YOUR CODE HERE
    return network

def ws(n, m, p):
    """
    This function call the ring() function to make a basic ring and then
    rewires each link with  probability p and also prints the total number of
    links and the number of rewired links.
    Note self-loops are not allowed when rewiring (check that you do not rewire
    the end of a link to the node at its other end!)

    Parameters
    ----------
    n : int
      Number of nodes
    m : int
      Number of neighbors to connect left and right
    p : float
        Rewiring probability

    Returns
    -------
    network : graph
        The Watts-Strogatz small-world network

    """
    network = ring(n, m)
    edges = network.edges()
    rewired_num = 0 # tracks the number of rewired links
    total_num = 0 # tracks the total number of links in the network
    # YOUR CODE HERE
    # You should rewire each edge if: numpy.random.rand() < p
    # Also, avoid duplicate links (rewiring to a neighbor of the other node)
    # as well as self-links (rewiring one endpoint to the other)
    # You might find these useful: random.choice and nx.non_neighbors
    # The latter yields an iterator: NN=nx.non_neighbors(G,i) lists all nodes
    # that are not connected to i (and are not i)
    #print total number of links in the graph and number of rewired links:
    print("total number of links:")
    print(total_num)
    print("number of rewired links:")
    print(rewired_num)
    return network

# =========================== MAIN CODE BELOW ==============================

if __name__ == "__main__":
    np.random.seed(42)
    #visualizing the rings for p = 0 ...
    graph1 = ws(15, 2, .1)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    nx.draw_circular(graph1)

    figure_filename = 'Small_world_ring.pdf'


    fig1.savefig(figure_filename)
    # or just use plt.show() and save manually

    total_num_edges = len(list(graph1.edges()))
    print("Total number of edges for n = 15, m = 2, p = 0 :")
    print(total_num_edges)
    #... and p = 0.5
    graph2 = ws(100, 2, 0.5)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    nx.draw_circular(graph2, node_size=20)

    figure_filename = './Small_world_rewired.pdf'


    fig2.savefig(figure_filename)
    # or just use plt.show() and save manually

    # Produce the basic ring network and calculate the average clustering
    # coefficient and average shortest path of the network
    basic_ring = ring(1000, 5)
    # YOUR CODE HERE
    c_basic = None # Replace!
    l_basic = None # Replace!

    probability = [0.001*(2**n) for n in range(11)] #[0.001, 0.002, 0.004, ...]
    relative_c = []
    relative_l = []

    for p in probability:
        smallworld = ws(500, 2, p)

        # gets all connected components; mostly there is just one:
        components = nx.connected_component_subgraphs(smallworld)

        # finds the largest to be used for the average path length:
        largest_component = max(components, key=len)

        # YOUR CODE HERE
        c_rewired = None # Replace!
        l_rewired = None # Replace!
        # Update relative_c and relative_l

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    ax.semilogx(probability, relative_c, marker='o', ls='', color='b',
                label='relative avg c')
    ax.semilogx(probability, relative_l, marker='o', ls='', color='r',
                label='relarive avg shortest path')
    # YOUR CODE HERE
    # Label the axes

    figure_filename = 'WS_relative_c_and_l.pdf'


    fig3.savefig(figure_filename)

    # or just use plt.show() and save manually
