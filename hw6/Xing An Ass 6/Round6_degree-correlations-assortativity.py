#!/usr/bin/env python
# coding: utf-8

# # Round 6. Degree correlations and assortativity
# 
# In this problem, we consider degree correlations and assortativity of two real-world networks: the Zachary karate club network `karate_club_network_edge_file.edg` (W. W. Zachary, 1977, “An information flow model for conflict and fission in small groups”) and a snowball-sampled subgraph of a Facebook friendships network `facebook-wosn-links_subgraph.edg` (http://konect.uni-koblenz.de/networks/facebook-wosn-links, B. Viswanath, A. Mislove, M. Cha, and K. P. Gummadi, 2009, “On the evolution of user interaction in facebook”). 
# 
# To get started, you can use the provided Python template `degree_correlations_assortativity.py` OR this notebook for reference. The usage of the notebook or template is **optional**. Then you only need to fill in the required functions. Some of the functions do NOT need modifications. You may start your solution after the subtitle "**Begin of the Exercise**" down below. 
# 
# In addition to returning a short report of your results (including the visualizations), return also your commented Python code or notebook. Remember to label the axes in your figures!

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib as mpl


# In[2]:


def create_scatter(x_degrees, y_degrees, network_title):
    """
    For x_degrees, y_degrees pair, creates and
    saves a scatter of the degrees.

    Parameters
    ----------
    x_degrees: np.array
    y_degrees: np.array
    network_title: str
        a network-referring title (string) for figures

    Returns
    -------
    no output, but scatter plot (as pdf) is saved into the given path
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    alpha = 0.5
    ax.plot(x_degrees, y_degrees, 'r', ls='', marker='o', ms=5, alpha=alpha)
    ax.set_xlabel(r'Degree $k$')
    ax.set_ylabel(r'Degree $k$')

    ax.set_title(network_title)

    return fig


# In[3]:


def create_heatmap(x_degrees, y_degrees, network_title):
    """
    For x_degrees, y_degrees pair, creates and
    saves a heatmap of the degrees.

    Parameters
    ----------
    x_degrees: np.array
    y_degrees: np.array
    network_title: str
        a network-referring title (string) for figures

    Returns
    -------
    no output, but heatmap figure (as pdf) is saved into the given path
    """
    k_min = np.min((x_degrees, y_degrees))
    k_max = np.max((x_degrees, y_degrees))

    n_bins = k_max-k_min+1
    values = np.zeros(x_degrees.size)

    statistic = binned_statistic_2d(x_degrees,y_degrees, values,
                                    statistic='count', bins=n_bins)[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(statistic, extent=(k_min-0.5, k_max+0.5, k_min-0.5, k_max+0.5),
              origin='lower', cmap='hot', interpolation='nearest')
    ax.set_title(network_title)
    ax.set_xlabel(r'Degree $k$')
    ax.set_ylabel(r'Degree $k$')
    cmap = plt.get_cmap('hot')
    norm = mpl.colors.Normalize(vmin=np.min(statistic), vmax=np.max(statistic))
    scm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(scm, ax=ax)
    return fig


# In[4]:


def visualize_nearest_neighbor_degree(degrees, nearest_neighbor_degrees, bins, bin_averages,
                                      network_title):
    """
    Visualizes the nearest neighbor degree for each degree as a scatter and
    the mean nearest neighbor degree per degree as a line.

    Parameters
    ----------
    degrees: list-like
        an array of node degrees
    nearest_neighbor_degrees: list-like
        an array of node nearest neighbor degrees in the same order as degrees
    bins: list-like
        unique degree values
    bin_averages: list-like
        the mean nearest neighbor degree per unique degree value
    network_title: str
        network-referring title (string) for figure

    Returns
    -------
    fig : figure object
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(degrees, nearest_neighbor_degrees,
              ls='', marker='.', label=r'$k_{nn}$($k$)')
    ax.loglog(bins, bin_averages,
              color='r', label=r'<$k_{nn}$>($k$)')
    ax.set_title(network_title)
    ax.set_xlabel(r'Degree $k$')
    ax.set_ylabel(r'Average nearest neighbor degree $k_{nn}$')
    ax.legend(loc=0)
    return fig


# ## Data
# Let us load the data from the right folder and assign the names for all the plots we will save. If you run this notebook in your machine, please specify the right folder.

# In[5]:


# Select data directory
import os
if os.path.isdir('/coursedata'):
    course_data_dir = '/coursedata'
elif os.path.isdir('../data'):
    course_data_dir = '../data'
else:
    # Specify course_data_dir on your machine
    course_data_dir = 'some_path'
    # YOUR CODE HERE
    #raise NotImplementedError()

print('The data directory is %s' % course_data_dir)


# ### a. Scatter plot
# Create a scatter plot of the degrees of pairs of connected nodes. That is, take each connected pair of nodes $(i,j)$, take their degrees $k_i$ and $k_j$ , plot the point $(k_i,k_j)$ on two axes with degrees as their units, and repeat for all pairs of connected nodes. Because the network is undirected, the plot should be symmetrical, containing points $(k_i,k_j)$ and $(k_j,k_i)$ for all connected pairs $(i,j)$.
# 
# For this purpose, you will need to modify the `get_x_and_y_degrees` function.

# In[6]:


def get_x_and_y_degrees(network):
    """
    For the given network, creates two arrays (x_degrees
    and y_degrees) of the degrees of "start" and "end" nodes of each edge in
    the network. For undirected networks, each edge is considered twice.

    Parameters
    ----------
    network: a NetworkX graph object

    Returns
    -------
    x_degrees: np.array
    y_degrees: np.array
    """
    edges = network.edges()
    n_edges = len(edges)
    x_degrees = np.zeros(2 * n_edges)
    y_degrees = np.zeros(2 * n_edges)
    for i,(u,v) in enumerate(edges):
        x_degrees[i] = network.degree[u]
        y_degrees[i] = network.degree[v]
    for i in range(n_edges):
        x_degrees[n_edges+i] = y_degrees[i]
        y_degrees[n_edges+i] = x_degrees[i]
    #TODO: write a correct definition for x_arrays and y_arrays
    # check the excercise sheet for more information
    # YOUR CODE HERE
    #raise NotImplementedError()
    return x_degrees, y_degrees


# In[7]:


network_paths = ['karate_club_network_edge_file.edg', 'facebook-wosn-links_subgraph.edg']
network_names = ['karate', 'facebook-wosn']
network_titles = ['Karate club network', 'Facebook friendships network']
scatter_figure_base = './edge_degree_correlation_scatter_'
heatmap_figure_base = './edge_degree_correlation_'
nearest_neighbor_figure_base = './nearest_neighbor_degree_'


for network_path, network_name, network_title in zip(network_paths, network_names, network_titles):
    network_pname = os.path.join(course_data_dir, network_path)
    network = nx.read_weighted_edgelist(network_pname)
    x_degrees, y_degrees = get_x_and_y_degrees(network)

    fig = create_scatter(x_degrees, y_degrees, network_title)
    fig.savefig(scatter_figure_base+network_name+'.pdf')


# ### b. Heatmap
# Produce a heat map of the degrees of all connected nodes. The heat map uses the same information as you used in a), that is, the degrees of pairs of connected nodes. However, no points are plotted: rather, the two degree axes are binned and the number of degree pairs $(k_i,k_j)$ in each bin is computed. Then, the bin is colored according to this number (e.g., red = many connected pairs of nodes with degrees falling in the bin). What extra information do you gain by using a heatmap instead of just a scatter plot (if any)?

# In[8]:


for network_path, network_name, network_title in zip(network_paths, network_names, network_titles):
    network_pname = os.path.join(course_data_dir, network_path)
    network = nx.read_weighted_edgelist(network_pname)
    x_degrees, y_degrees = get_x_and_y_degrees(network)

    fig = create_heatmap(x_degrees, y_degrees, network_title)
    fig.savefig(heatmap_figure_base+network_name+'.pdf')


# ### c. Assortativity
# The assortativity coefficient is defined as the Pearson correlation coefficient of the degrees of pairs of connected nodes. Calculate the assortativity coefficient of the network using `scipy.stats.pearsonr` and compare your result with the output of NetworkX function `degree_assortativity_coefficient`. As mentioned in the lecture, social networks typically are assortative. Does this hold for these two social networks? What could explain this result?
# 
# For this purpose, you will need to modify the `assortativity` function.

# In[13]:


from scipy import stats
def assortativity(x_degrees, y_degrees):
    """
    Calculates assortativity for a network, i.e. Pearson correlation
    coefficient between x_degrees and y_degrees in the network.

    Parameters
    ----------
    x_degrees: np.array
    y_degrees: np.array

    Returns
    -------
    assortativity: float
        the assortativity value of the network as a number
    """
    assortativity = 0 # to be replaced
    assortativity = stats.pearsonr(x_degrees,y_degrees)[0]
    #TODO: write code for calculating assortativity
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    return assortativity


# In[14]:


for network_path, network_name, network_title in zip(network_paths, network_names, network_titles):
    network_pname = os.path.join(course_data_dir, network_path)
    network = nx.read_weighted_edgelist(network_pname)
    x_degrees, y_degrees = get_x_and_y_degrees(network)
    
    # assortativities
    assortativity_own = assortativity(x_degrees, y_degrees)
    assortativity_nx = nx.degree_assortativity_coefficient(network)
    print("Own assortativity for " + network_title + ": " +
            str(assortativity_own))
    print("NetworkX assortativity for " + network_title + ": " +
            str(assortativity_nx))


# ### d. Average nearest neighbor
# For each node, compute the average nearest neighbour degree $k_nn$ and make a scatter plot of $k_{nn}$ as a function of $k$. In the same plot, plot also the curve of $\langle k_{nn} \rangle(k)$ as a function of $k$, *i.e.* the averaged $k_{nn}$ for each $k$. Comment the result from the viewpoint of assortativity.
# 
# For this purpose, you will need to modify the `get_nearest_neighbor_degree` and `get_simple_bin_average` functions.

# In[68]:


def get_nearest_neighbor_degree(network):
    """
    Calculates the average nearest neighbor degree for each node for the given
    list of networks.

    Parameters
    ----------
    network: a NetworkX graph objects

    Returns
    -------
    degrees: list-like
        an array of node degree
    nearest_neighbor_degrees: list-like
        an array of node average nearest neighbor degree in the same order
        as degrees
    """
    
    degrees = [] #to be replaced
    nearest_neighbor_degrees = [] #to be replaced
    degrees_d = nx.degree(network)
    nnd_dic = nx.average_neighbor_degree(network)
    for node in network:
        degrees.append(degrees_d[node])
        nearest_neighbor_degrees.append(nnd_dic[node])
    #for i in sorted (nnd_dic.keys()):
        #nearest_neighbor_degrees.append(nnd_dic[i])
    #print(nearest_neighbor_degrees)
    #print(degrees)
    #TODO: write code for calculating degrees and average nearest
    #  neighbor degrees of the networks.
    # Hint: if using nx.degree() and nx.average_neighbor_degree(), remember
    # that key-value pairs of dictionaries are not automatically in a fixed
    # order!
    # YOUR CODE HERE
    #raise NotImplementedError()
    return degrees, nearest_neighbor_degrees


# In[69]:


def get_simple_bin_average(x_values, y_values):
    """
    Calculates average of y values for each x-value bin. The binning used is the
    most simple one: each unique x value is a bin of it's own.

    Parameters
    ----------
    x_values: an array of x values
    y_values: an array of corresponding y values

    Returns
    -------
    bins: an array of unique x values
    bin_average: an array of average y values per each unique x
    """
    #TODO: set proper bins and compute bin-averages
    #bins = np.array([])# replace
    #print(x_values)
    bins = np.unique(x_values)
    #bins = list(set(x_values))
    #print(bins)
    bin_average = []
    for i in range(len(bins)):
        sum_y = 0
        sum_x = 0
        for j in range(len(x_values)):
            #print(x_values[j])
            if(x_values[j] == bins[i]):
                #print(bin_average)
                sum_y += y_values[j]
                sum_x += 1
        average_y = sum_y/sum_x
        
        bin_average.append(average_y)
                
                
                
    
     # replace
    # YOUR CODE HERE
    #raise NotImplementedError()
    return bins, bin_average


# In[70]:


for network_path, network_name, network_title in zip(network_paths, network_names, network_titles):
    network_pname = os.path.join(course_data_dir, network_path)
    network = nx.read_weighted_edgelist(network_pname)
    x_degrees, y_degrees = get_x_and_y_degrees(network)
    
    # nearest neighbor degrees
    degrees, nearest_neighbor_degrees = get_nearest_neighbor_degree(network)
    unique_degrees, mean_nearest_neighbor_degrees = get_simple_bin_average(degrees, nearest_neighbor_degrees)
    fig = visualize_nearest_neighbor_degree(degrees,
                                            nearest_neighbor_degrees,
                                            unique_degrees,
                                            mean_nearest_neighbor_degrees,
                                            network_title)
    fig.savefig(nearest_neighbor_figure_base + network_name + '.pdf')


# In[ ]:




