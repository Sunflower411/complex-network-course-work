#!/usr/bin/env python
# coding: utf-8

# # Round 7. Weight–topology correlations in social networks 
# 
# In this exercise, we will do some weighted network analysis using a social network data set describing private messaging in a Facebook-like web-page (http://toreopsahl.com/datasets/ ). In the network, each node corresponds to a user of the website and link weights describe the total number of messages exchanged between users.
# 
# In the ﬁle `OClinks_w_undir.edg`, the three entries of each row describe one link: `(node_i node_j w_ij)`, where the last entry `w_ij` is the weight of the link between nodes `node_i` and `node_j`.
# 
# You can use these notebook or the accompanying Python template (weight_topology_correlations.py) to get started. The notebook and template have some functions that will help you. Do NOT modify these functions. You may start modifying the code after the subtitle **Begin of the exercise**.
# 
# `scipy.stats.binned_statistic` function is especially useful throughout this exercise.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import binned_statistic


# In[2]:


def create_linbins(start, end, n_bins):
    """
    Creates a set of linear bins.

    Parameters
    -----------
    start: minimum value of data, int
    end: maximum value of data, int
    n_bins: number of bins, int

    Returns
    --------
    bins: a list of linear bin edges
    """
    bins = np.linspace(start, end, n_bins)
    return bins


# In[3]:


def create_logbins(start, end, n_log, n_lin=0):
    """
    Creates a combination of linear and logarithmic bins: n_lin linear bins 
    of width 1 starting from start and n_log logarithmic bins further to
    max.

    Parameters
    -----------
    start: starting point for linear bins, float
    end: maximum value of data, int
    n_log: number of logarithmic bins, int
    n_lin: number of linear bins, int

    Returns
    -------
    bins: a list of bin edges
    """
    if n_lin == 0:
        bins = np.logspace(np.log10(start), np.log10(end), n_log)
    elif n_lin > 0:
        bins = np.array([start + i for i in range(n_lin)] + list(np.logspace(np.log10(start + n_lin), np.log10(end), n_log)))
    return bins


# ## Data
# Let us load the data from the right folder and assign the names for all the plots we will save. If you run this notebook in your machine, please specify the right folder.

# In[4]:


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

network_path = os.path.join(course_data_dir, 'OClinks_w_undir.edg')

net_name = 'fb_like'
#TODO: replace with a path where to save the 1-CDF plot
path = './ccdfs_' + net_name + '.png'
#TODO: replace with a base path where to save the average link weight scatter
# A scale-related suffix will be added to this base path so the figures will not overwritte
base_path = './s_per_k_vs_k_'
#TODO: replace with a base path where to save the link neighborhood overlap plot
save_path_linkneighborhoodplot = './O_vs_w_' + net_name + '.png'


# In[5]:


#Let's read the network file
network = nx.read_weighted_edgelist(network_path)


# # Begin of the exercise
# Write your code here to analyze the social network dataset. 

# ### a. Complementary cumulative distribution
# 
# Before performing more sophisticated analysis, it is always good to get some idea on how the network is like. To this end, plot the complementary cumulative distribution (1-CDF) for node degree *k*, node strength *s* and link weight *w*.
# 
# - Show all three distributions in one plot using loglog-scale
# - Briefly describe the distributions: are they Gaussian, power laws or something else?
# - Based on the plots, roughly estimate the 90th percentiles of the degree, strength, and weight distributions.
# 
# To achieve this, you will need to modify two functions: `get_link_weights` and  `plot_ccdf` first.
# 
# **Hints**:
# - See the binning tutorial for help on computing the 1-CDFs.
# - For getting node strengths, use `strengths = nx.degree(net, weight="weight")`

# In[6]:


def get_link_weights(net):

    """
    Returns a list of link weights in the network.

    Parameters
    -----------
    net: a networkx.Graph() object

    Returns
    --------
    weights: list of link weights in net
    """

    # TODO: write a function to get weights of the links
    # Hints:
    # to get the links with their weight data, use net.edges(data=True)
    # to get weight of a single link, use (i, j, data) for each edge,
    # weight = data['weight']
    weights = []
    edgewithweight = net.edges(data=True)
    for i,j,d in edgewithweight:
        weights.append(d['weight'])
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    return weights


# In[35]:


#from itertools import accumulate
#from statsmodels.distributions.empirical_distribution import ECDF
def plot_ccdf(datavecs, labels, xlabel, ylabel, num, path):

    """
    Plots in a single figure the complementary cumulative distributions (1-CDFs)
    of the given data vectors.

    Parameters
    -----------
    datavecs: data vectors to plot, a list of iterables
    labels: labels for the data vectors, list of strings
    styles = styles in which plot the distributions, list of strings
    xlabel: x label for the figure, string
    ylabel: y label for the figure, string
    num: an id of the figure, int or string
    path: path where to save the figure, string
    """
    styles = ['-', '--', '-.']
    fig = plt.figure(num)
    ax = fig.add_subplot(111)
    for datavec, label, style in zip(datavecs,labels, styles):
        #TODO: calculate 1-CDF of datavec and plot it with ax.loglog()
        sorted_datavec = sorted(datavec)
        ccdf = 1 - np.linspace(0, 1, len(sorted_datavec),endpoint=False)
        #ccdf = 1 - (np.cumsum(sorted_datavec)/sum(sorted_datavec))
        # YOUR CODE HERE
        #https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
        ax.loglog(sorted_datavec,ccdf,style,label = label)
    #print(ccdf)
    #raise NotImplementedError()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=0)
    ax.grid()

    return fig


# In[36]:


# Get the node degrees and strengths
degrees = nx.degree(network)
strengths = nx.degree(network,weight='weight')
#TODO: get network degrees
#TODO: get network strength

# YOUR CODE HERE
#raise NotImplementedError()

#Now, convert the degree and strength into lists.
degree_vec = []
strength_vec = []
for node in network.nodes():
    degree_vec.append(degrees[node])
    strength_vec.append(strengths[node])

# Then, compute the weights
weights = get_link_weights(network)

# Now let's create 1-CDF plots
datavecs = [degree_vec, strength_vec, weights]
num = 'a)' + net_name # figure identifier

#TODO: set the correct labels
labels = ['Degree', 'Strength ', ' Weight'] ### TIP: set the proper labels
xlabel = 'degree/strength/weight ' ### TIP: label the axis!
ylabel = '1-CDF ' ### TIP: label the axis!

fig=plot_ccdf(datavecs, labels, xlabel, ylabel, num, path)
fig.savefig(path)
print('1-CDF figure saved to ' + path)


# ### b. Average link weight per node
# Next, we will study how the average link weight per node $\langle w \rangle =\frac{s}{k}$ behaves as a function of the node degree $k$.
# - Compute $s$, $k$, and $\langle w \rangle = \frac{s}{k}$ for each node. 
# - Make a scatter plot of $\langle w \rangle$ as a function of $k$. 
# - Create two versions of the scatter plot: one with linear and one with logarithmic x-axes.

# In[37]:


# average link weight per node
av_weight = []
for i in range(len(degree_vec)):
    av_weight.append(strength_vec[i]/degree_vec[i])
#TODO: calculate average link weight per node
# YOUR CODE HERE
#raise NotImplementedError()
#print(av_weight)
# Since b and c solution plots can be drawn in one figure for linear and one figure for logarithmic, 
# let's plot the scatters and bin averages in one figure in the c part.


# ### c. 
# The large variance of the data can make the scatter plots a bit messy. To make the relationship between $\langle w \rangle$ and $k$ more visible, create bin-averaged versions of the plots, i.e. divide nodes into bins based on their degree and calculate the average $\langle w \rangle$ in each bin. Now, you should be able to spot a trend in the data.
# 
# **Hints**:
# - For the linear scale, use bins with constant width. For the logarithmic scale, use logarithmic bins. If in trouble, see the binning tutorial for help.
# - Use the number of bins you ﬁnd reasonable. Typically, it is better to use too many than too few bins.
# - An example of how the scatter and bin-averaged plots may look like is shown in Fig.1 of the assignment PDF. Note that the results will vary according to the number of bins.

# In[38]:


n_bins = 50 #TIP: use the number of bins you find reasonable
min_deg = min(degree_vec)
max_deg = max(degree_vec)
linbins = create_linbins(min_deg, max_deg, n_bins)
logbins = create_logbins(0.5, max_deg, n_bins, n_lin=10)
print(logbins)
print(max_deg)
num = 'b) ' + net_name + "_"
alpha = 0.1 # transparency of data points in the scatter

for bins, scale in zip([linbins, logbins], ['linear', 'log']):
    fig = plt.figure(num + scale)
    ax = fig.add_subplot(111)
    # mean degree value of each degree bin
    degree_bin_means, _, _ = binned_statistic(degree_vec,degree_vec,statistic = 'mean', bins = bins)
    # TODO: use binned_statistic to get mean degree of each bin 
    # YOUR CODE HERE
    #raise NotImplementedError()

    # mean strength value of each degree bin
    strength_bin_means, _, _ = binned_statistic(degree_vec,strength_vec,statistic = 'mean', bins = bins)
    # TODO: use binned_statistic to get mean strength of each bin)
    # YOUR CODE HERE
    #raise NotImplementedError()

    # number of points in each degree bin
    counts, _, _ = binned_statistic(degree_vec,av_weight,statistic = 'count', bins = bins)
    # TODO: use binned_statistic to get number of data points
    # YOUR CODE HERE
    #raise NotImplementedError()

    # b: plotting all points (scatter)
    ax.scatter(degree_vec, av_weight, marker='o', s=1.5, alpha=alpha)
    # calculating the average weight per bin
    bin_av_weight = strength_bin_means / degree_bin_means

    # c: and now, plotting the bin average
    # the marker size is scaled by number of data points in the bin
    ax.scatter(degree_bin_means,
                bin_av_weight,
                marker='o',
                color='g',
                s=np.sqrt(counts) + 1,
                label='binned data')
    ax.set_xscale(scale)
    min_max = np.array([min_deg, max_deg])
    ax.set_xlabel('Degree ') #TIP: Do not forget to label the axis
    ax.set_ylabel('Av weight ') #TIP: Do not forget to label the axis
    ax.grid()

    ax.legend(loc='best')
    plt.suptitle('avg. link weight vs. strength:' + net_name)
    save_path = base_path + scale + '_' + net_name + '.png'
    fig.savefig(save_path)
    print('Average link weight scatter saved to ' + save_path)


# ### d. What does it mean?
# Based on the plots created in b), answer the following questions questions:
# - Which of the two approaches (linear or logarithmic x-axes) suits better for presenting $\langle w \rangle$ as a function of $k$? Why?
# - In social networks, $\langle w \rangle$ typically decreases as a function of the degree due to time constraints required for taking care of social contacts. Are your results in accordance with this observation? If not, how would you explain this?
# 
# **Hints**:
# - Check your results from a). Is there an equal number of nodes with each degree and strength? Nonequal distribution of observations may obscure the results.
# - You are dealing with real data that may be noisy. So, interpretation of results may be confusing at ﬁrst - do not worry!

# ### e. Link neighborhood overlap
# Lets consider a link between nodes $i$ and $j$. For this link, *link neighborhood overlap* $O_{ij}$ is defined as the fraction of common neighbors of $i$ and $j$ out of all their neighbors: $O_{ij}=\frac{n_{ij}}{\left(k_i-1\right)+\left(k_j-1\right)-n_{ij}}.$
# 
# According to the Granovetter hypothesis, link neighborhood overlap is an increasing function of link weight in social networks.  Next, your task is now to find out whether this is the case also for the present data set. To this end:
# - Calculate the link neighborhood overlap for each link. To do this, you will need to modify the `get_link_overlap` function.
# - Create a scatter plot showing the overlaps as a function of link weight.
# - As in c), produce also a bin-averaged version of the plot. Use a binning strategy that is most suitable for this case.
# 
# In the end, you should be able to spot a subtle trend in the data. Based on your plot, answer the following questions:
# - Is this trend in accordance with the Granovetter hypothesis? If not, how would you explain your ﬁndings?

# In[65]:


def get_link_overlap(net):
    """
    Calculates link overlap: 
    O_ij = n_ij / [(k_i - 1) + (k_j - 1) - n_ij]

    Parameters
    -----------
    net: a networkx.Graph() object

    Returns
    --------
    overlaps: list of link overlaps in net
    """

    # TODO: write a function to calculate link neighborhood overlap
    # Hint: for getting common neighbors of two nodes, use
    # set datatype and intersection method

    overlaps = []
    for (i,j) in net.edges():
        n_ij = len(set(net.neighbors(i)).intersection(set(net.neighbors(j))))
        denominator = net.degree[i]+net.degree[j]-2-n_ij
        if denominator == 0:
            o_ij = 0
        else:
            o_ij = n_ij / denominator
        overlaps.append(o_ij)
    # YOUR CODE HERE
    #raise NotImplementedError()
    return overlaps


# In[66]:


# Get link neighborhood overlaps
overlaps = get_link_overlap(network)

# creating link neighborhood overlap scatter
num = 'd) + net_name'
fig = plt.figure(num)
ax = fig.add_subplot(111)

n_bins = 30 #TIP: use the number of bins you find reasonable
min_w = np.min(weights)
max_w = np.max(weights)

linbins = create_linbins(min_w, max_w, n_bins)
logbins = create_logbins(min_w, max_w, n_bins)

#TODO: try both linear and logarithmic bins, select the best one
bins = logbins

# mean weight value of each weight bin
weight_bin_means, _, _ = binned_statistic(weights,weights,statistic = 'mean', bins = bins)
#TODO: use binned_statistic to get mean weight of each bin
# YOUR CODE HERE
#raise NotImplementedError()

# mean link neighborhood overlap of each weight bin
overlap_bin_means, _, _ = binned_statistic(weights,overlaps,statistic = 'mean', bins = bins)
#TODO: use binned_statistic to get mean overlap of each bin 
# YOUR CODE HERE
#raise NotImplementedError()
#print(weight_bin_means)
# number of points in each weigth bin
counts, _, _ = binned_statistic(weights,overlaps,statistic='count',bins = bins)
#TODO: use binned_statistic to get number of data points
# YOUR CODE HERE
#raise NotImplementedError()

# plotting all points (overlap)
ax.scatter(weights, overlaps, marker="o", s=1.5, alpha=alpha)
# plotting bin average, marker size scaled by number of data points in the bin
ax.scatter(weight_bin_means,
            overlap_bin_means,
            s=np.sqrt(counts) + 2,
            marker='o',
            color='g')

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.set_xlabel('Weight ') #TIP: Do not forget to label the axis
ax.set_ylabel('Overlap Neighbour') #TIP: Do not forget to label the axis
fig.suptitle('Overlap vs. weight:' + net_name)
fig.savefig(save_path_linkneighborhoodplot)
print('Link neighborhood overlap scatter saved as ' + save_path_linkneighborhoodplot)


# In[ ]:




