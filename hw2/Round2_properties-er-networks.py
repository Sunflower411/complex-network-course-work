#!/usr/bin/env python
# coding: utf-8

# # Round 2. Properties of Erdös-Rényi (ER) networks
# 
# Erdös-Rényi networks are random networks where $N$ nodes are randomly connected such that the probability that a pair of nodes are linked is $p$. In network science, the ER random graphs are important because they provide the simplest reference to which one can compare real-world networks.
# 
# In this exercise, we will analyze some of the properties of ER graphs.
# 
# To get you started, you may use this notebook or the accompanying Python template `properties_of_er_networks.py` available in MyCourses. **The usage of the notebook or template is fully optional**. Then you only need to fill in the required functions. Some of the functions do NOT need modifications. You may start your solution after the subtitle "**Begin of the Exercise**" down below. 

# Use NetworkX to calculate estimates for the ensemble averages $\left< k \right>$, $\left< c \right>$, and $\left< d^* \right>$  defined in the Exercise 1 of the Assignment Round 2. Do this by generating 100 ER networks for each value of $p$ in range $p=\left[0.00, 0.05, ..., 0.95, 1\right]$ and $N=3$. An estimate for the ensemble average $\left< X \right>$ can be calculated for each value of $p$ by calculating the average value of $X$ over the 100 realisations. 
# 
# - Plot the quantities, such that $p$ values are on the x-axis and $\left< X \right>$ values are on the y-axis. If you solved the part b of Exercise 1 (Assignment Round 2), you can check the correctness of your results by including your analytical solution to these plots. 
# - Repeat the same exercise with $N=100$.
# 
# You would need to modify the functions `ER_properties` and `ER_properties_theoretical` for this purpose. The figures will be automatically saved in the notebook folder.
# 
# **Hints**:
# - You can use the function `nx.fast_gnp_random_graph(N, p)` for generating ER networks.
# - You can use the function `nx.diameter()` to calculate the diameter of a network. However, this function only accepts graphs that have a single connected component. Use the function `nx.connected_component_subgraphs()` to iterate over the components, calculate the diameter for each of them, and find the largest value $d^*$.

# In[5]:


import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[6]:


def plot_er_values(n, p_list):
    '''
    This function calculates the theoretical clustering coefficient, average
    degree and diameter for ER network with parameters n and p and plots them
    against the expected values from an ensemble of 100 realizations.

    Parameters
    ----------
    n : int
      Number of nodes
    p_list : list of floats
      where each member is the probability that a pair of nodes are linked.

    Returns
    -------
    fig: matplotlib Figure
                plots of expected values against theoretical values
    '''

    k_list = [] # list for degrees
    c_list = [] # list for clustering coeff values
    d_list = [] # list for diameter values
    c_list_theory = []
    k_list_theory = []
    d_list_theory = []

    for p in p_list:
        print("calculating for n=%d p=%f" % (n, p), file=sys.stderr)
        average_c, average_k, average_d = ER_properties(n, p)
        k_list.append(average_k)
        c_list.append(average_c)
        d_list.append(average_d)

        if n is 3:
            c_theory, k_theory, d_theory = ER_properties_theoretical(p)
            c_list_theory.append(c_theory)
            k_list_theory.append(k_theory)
            d_list_theory.append(d_theory)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1, 3, 1) # Three subplots for <K>, <C> and <d*> [(1,3,1) means 1 row, three columns, first subplot)

    ax.plot(p_list, c_list, 'r-', label="Simulated", marker='o')
    if len(c_list_theory) > 0 and c_list_theory[0] is not None:
        ax.plot(p_list, c_list_theory, 'b-', label="Theoretical", marker='o')
    ax.set_xlabel('p')
    ax.set_ylabel('Clustering coefficient')
    ax.legend(loc=0)
    ax.set_title('N = %d' % n, size=18)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(p_list, k_list, 'r-', label="Simulated", marker='o')
    if len(k_list_theory) > 0 and k_list_theory[0] is not None:
        ax2.plot(p_list, k_list_theory, 'b-', label="Theoretical", marker='o')
    ax2.set_xlabel('p')
    ax2.set_ylabel('Average degree')
    ax2.legend(loc=0)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(p_list, d_list, 'r-', label="Simulated", marker='o')
    if len(d_list_theory) > 0 and d_list_theory[0] is not None:
        ax3.plot(p_list, d_list_theory, 'b-', label="Theoretical", marker='o')
    ax3.set_xlabel('p')
    ax3.set_ylabel('Diameter')
    ax3.legend(loc=0)

    fig.tight_layout()

    return fig


# # Begin of the Exercise
# Write your code here to analyze some of the properties of ER graphs.

# In[22]:


def ER_properties(n, p):
    '''
    This function builds 100 ER networks with the given parameters and averages
    over them to estimate the expected values of average clustering coefficient,
    average degree and diameter of an ER network with the given parameters. The
    diameter is always computed for the largest ("giant") connected component.

    Parameters
    ----------
    n : int
      Number of nodes
    p : float
      the probability that a pair of nodes are linked is p.

    Returns
    -------
    expected_c: float
                expected value of average clustering coefficient
    expected_k: float
                expected value of average degree
    expected_d: float
    expected value of d*



    Hints:
        The following functions might be useful:
        nx.fast_gnp_random_graph(n, p),
        nx.average_clustering(graph, count_zeros=True),
        nx.connected_component_subgraphs(graph),
        nx.diameter(giant)

        nx.connected_component_subgraphs gives you a list of subgraphs
        To pick the largest, the fastest way is to use max with a key: max(x,key=len)
        returns the longest/largest element of the list x

        For computing averages over realizations, you can e.g. collect your
        values to three lists, c,k,d, and use np.mean to get the average.
    '''
    c = [] # change these!
    k = [] # change these!
    d = [] # change these!
    for i in range(100):
        net = None
        net = nx.fast_gnp_random_graph(n, p)
        degree = sum(nx.average_neighbor_degree(net).values())/n
        k.append(degree)
        cc = nx.average_clustering(net, count_zeros=True)
        c.append(cc)
        connet = None
        connet = max(nx.connected_component_subgraphs(net), key=len)
        diameter = nx.diameter(connet)
        d.append(diameter)
    
    expected_c = np.mean(c) # change these!
    expected_k = np.mean(k)# change these!
    expected_d = np.mean(d) # change these!
    
    # YOUR CODE HERE
    

    return expected_c, expected_k, expected_d


# In[23]:


def ER_properties_theoretical(p):
    '''
    This function calculates the theoretical values for clustering coefficients,
    average degree, and diameter for ER networks of size 3 and link probability p.
    The theoretical values can be viewed as expectations, or ensemble averages.
    Therefore, e.g., the expected diameter doesn't have to be integer, although it of
    course always is for a single ER network.

    Parameters
    ----------
    p : float
      the probability that a pair of nodes are linked is p.

    Returns
    -------
    c_theory: float
                theoretical value of average clustering coefficient
    k_theory: float
                theoretical value of average degree
    d_theory: float
                Theoretical value of diameter
    '''

    # Calculate the theoretical values for and ER network with parameters n, p
    c_theory = p**3
    k_theory = 2*p
    d_theory = -2*(p**3)+3*p
    # YOUR CODE HERE
    
    return c_theory, k_theory, d_theory


# In[24]:


ps = np.arange(0, 1.01, 0.05)

fig = plot_er_values(n=3, p_list=ps)

figure_filename = './ER_properties_3_nodes.pdf'
fig.savefig(figure_filename)
# or just use plt.show() and save manually


# Do the same steps (calculation and visulization) for n=100
fig = plot_er_values(n=100, p_list=ps)

figure_filename = './ER_properties_100_nodes.pdf'
fig.savefig(figure_filename)
# or just use plt.show() and save manually


# In[ ]:




