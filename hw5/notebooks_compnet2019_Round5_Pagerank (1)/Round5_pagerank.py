#!/usr/bin/env python
# coding: utf-8

# # Round 5. Pagerank
# 
# PageRank, a generalization of eigenvector centrality for directed networks, is used by *e.g.* Google to determine the centrality of web pages. If we consider a random walker that with probability $d$ moves to one of the neighbors of the current node and with probability $1 - d$ teleports to a random node, PageRank of each node equals the fraction of time the random walker spent in that node. 
# 
# In this exercise, we investigate the behavior of PageRank in both a simple directed model network (see fig below) and an extract from the Wikipedia hyperlink network. To get started, you can use the provided Python template `pagerank.py` or this notebook for reference. The usage of the notebook or template is **optional**. Then you only need to fill in the required functions. Some of the functions do NOT need modifications. You may start your solution after the subtitle "**Begin of the Exercise**" down below. 
# 
# In addition to returning a short report of your results (including the visualizations), return also your commented Python code or notebook. Remember to label the axes in your figures!
# 

# In[2]:


from IPython.display import Image

fig = Image(filename=('/coursedata/pagerank_network.PNG'))
fig


# In[3]:


import timeit
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import networkx as nx


# In[4]:


def add_colorbar(cvalues, cmap='OrRd', cb_ax=None):
    """
    Add a colorbar to the axes.

    Parameters
    ----------
    cvalues : 1D array of floats

    """
    eps = np.maximum(0.0000000001, np.min(cvalues)/1000.)
    vmin = np.min(cvalues) - eps
    vmax = np.max(cvalues)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scm = mpl.cm.ScalarMappable(norm, cmap)
    scm.set_array(cvalues)
    if cb_ax is None:
        plt.colorbar(scm)
    else:
        cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation='vertical')


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

network_path = os.path.join(course_data_dir, 'pagerank_network.edg')


# # Begin of the exercise
# Write your code here to compute the pagerank

# ### a. Network visualization
# Load the network given in file `pagerank_network.edg` and, as a sanity check, visualize it with `nx.draw`. You would need to modify the function `visualize_network` for this purpose.
# 
# **Hint**:
# To load the directed network, use parameter `create_using=nx.DiGraph()` when reading the edge list.
# NetworkX visualization of directed graphs is somewhat ugly but sufficient for the present purposes. In fact, the spring layout algorithm in NetworkX, which is its default algorithm for computing node positions, works only well with undirected graphs, so for computing the layout, it's better to feed the algorithm the undirected version of the network. In addition, the algorithm can give different results on different runs, so it may be useful to plot the network a few times until the result looks good.

# In[6]:


def visualize_network(network, node_positions, cmap='OrRd',
                      node_size=3000, node_colors=[], with_labels=True, title=""):
    """
    Visualizes the given network using networkx.draw and saves it to the given
    path.

    Parameters
    ----------
    network : a networkx graph object
    node_positions : a dict of positions of nodes, obtained by e.g. networkx.graphviz_layout
    cmap : colormap
    node_size : int
    node_colors : a list of node colors
    with_labels : should node labels be drawn or not, boolean
    title: title of the figure, string
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    if node_colors:
        #TODO: write code to visualize the networks with nodes colored by PageRank.
        # use nx.draw and make use of parameters pos, node_color,
        # cmap, node_size and with_labels
        # YOUR CODE HERE
        nx.draw(network,pos = node_positions, node_color = node_colors, node_size = node_size, cmap = cmap, with_labels = with_labels)
        #raise NotImplementedError()
        add_colorbar(node_colors)
    else:
        #TODO: write code to visualize the networks without node coloring.
        # use nx.draw and make use of parameters pos, cmap, node_size and
        # with_labels
        # YOUR CODE HERE
        nx.draw(network,pos = node_positions, node_size = node_size, with_labels = with_labels)
        #raise NotImplementedError()
    ax.set_title(title)
    plt.tight_layout()

    return fig


# In[7]:


network =  nx.read_edgelist(network_path, create_using=nx.DiGraph())

# Visualization of the network (note that spring_layout is intended to be used with undirected networks):
node_positions = nx.spring_layout(network.to_undirected())
cmap = 'OrRd'
node_size = 3000

fig=visualize_network(network, node_positions, cmap=cmap, node_size=node_size, title="Network")

fig.savefig('./network_visualization.pdf')


# ### b. Compute the pagerank
# 
# Write a function that computes the PageRank on a network by simulating a random walker. In more detail:
# 1. Initialize the PageRank of all nodes to 0.
# 2. Pick the current node (the starting point of the random walker) at random.
# 3. Increase the PageRank of the current node with 1.
# 4. Select the node, to which the random walker will move next:
#     - Draw a random number $p \in [0, 1]$.
#     - If $p < d$, the next node is one of the successors of the current one (i.e. nodes linked *to* by the current node). Pick it randomly.
#     - Else, the random walker will teleport. Pick the next node randomly from all the network nodes.
# 5. Repeat 3-4 $N_{steps}$ times.
# 6. Normalize the PageRank values by $N_{steps}$.
# 
# Use your function to compute PageRank in the example network.
# Visualize the result on the network: update your visualization from a) by using the PageRank values as node color values.  Compare your results with `nx.pagerank` by plotting both results as a function of node index. Note that the above algorithm is only a naive way of computing PageRank. The actual algorithm behind the success of Google, introduced by its founders, Larry Page and Sergey Brin, is based on power iteration (Brin, S and Page, L, 2012, Reprint of: The anatomy of a large-scale hypertextual web search engine).
# 
# **Hint**:
# The damping factor is normally set to $d = 0.85$. $N_{steps} = 10 000$ is a reasonable choice.
# 

# In[8]:


def pageRank(network, d, n_steps):
    """
    Returns the PageRank value, calculated using a random walker, of each
    node in the network. The random walker teleports to a random node with
    probability 1-d and with probability d picks one of the neighbors of the
    current node.

    Parameters
    -----------
    network : a networkx graph object
    d : damping factor of the simulation
    n_steps : number of steps of the random walker

    Returns
    --------
    page_rank: dictionary of node PageRank values (fraction of time spent in
               each node)
    """

    # Initializing PageRank dictionary:
    pageRank = {}
    nodes = list(network.nodes())

    #TODO: write code for calculating PageRank of each node
    # Use the random walker algorithm.
    # Some pseudocode:

    # 1) Initialize PageRank of each node to 0
    pageRank = dict.fromkeys(nodes,0)
    # 2) Pick a random starting point for the random walker (Hint: np.random.choice)
    randomWalker = np.random.choice( a = nodes )
    # 3) Random walker steps, at each step:
    for n in range(n_steps):
        pageRank[randomWalker] += 1
        p = np.random.rand()
        if(p < d):
            randomWalker = np.random.choice(list(network.neighbors(randomWalker)))
        else:
            randomWalker = np.random.choice(nodes)
        
    #   1) Increase the PageRank of current node by 1
    #   2) Check if the random walker will teleport or go to a neighbor
    #   3) Pick the next node either randomly or from the neighbors
    #   4) Update the current node variable to point to the next node
    # 4) Repeat random walker steps 1-4 n_steps times
    # 5) Normalize PageRank by n_steps
    for key in pageRank:
        pageRank[key] /= n_steps 
    # YOUR CODE HERE
    #raise NotImplementedError()
    return pageRank


# In[9]:


nodes = network.nodes()
n_nodes = len(nodes)

n_steps = 10000 # TODO: replace: set a reasonable n_steps
d = 0.85 # TODO: replace: set a reasonable d; nx.pagerank uses d = 0.85
# YOUR CODE HERE
#raise NotImplementedError()
    
pageRank_rw = pageRank(network, d, n_steps)

# Visualization of PageRank on network:
node_colors = [pageRank_rw[node] for node in nodes]

fig = visualize_network(network, node_positions, cmap=cmap, node_size=node_size,
                        node_colors=node_colors, title="PageRank random walker")
fig.savefig('./network_visualization_pagerank_rw.pdf')

# PageRank with networkx:
pageRank_nx = nx.pagerank(network)

# Visualization to check that results from own function and nx.pagerank match:

pageRank_rw_array = np.zeros(n_nodes)
pageRank_nx_array = np.zeros(n_nodes)
for node in nodes:
    pageRank_rw_array[int(node)] = pageRank_rw[node]
    pageRank_nx_array[int(node)] = pageRank_nx[node]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(0, n_nodes), pageRank_rw_array, 'k+', label=r'Random walker')
plt.plot(range(0, n_nodes), pageRank_nx_array, 'rx', label='networkx')
ax.set_xlabel(r'Node index')
ax.set_ylabel(r'PageRank')
ax.set_title(r'PageRank with different methods')
ax.legend(loc=0)
plt.tight_layout()

fig.savefig('./network_visualization_pagerank_nx.pdf')


# ### c. Pagerank based on power iteration
# 
# The above algorithm is a naive way of computing PageRank. The actual algorithm behind the success of Google, introduced by its founders, Larry Page and Sergey Brin, is based on power iteration (Brin, S and Page, L, 2012).
# 
# The power iteration can be shown to find the leading eigenvector for the "Google matrix" (or other matrices) very fast under certain conditions. An intuitive way of thinking about the power iteration algorithm is to think that at time $t-1$ you have a vector $x(t-1)$ where each element gives the probability of finding the walker. You use the rules of the random walk/teleportation process to find out what are the probabilities of finding the random walkers at each node at time $t$. That is you increase the time $t$ and calculate $x(t)$ based on $x(t-1)$ until the vector $x$ doesn't change any more. 
# 
# Your task is to write a function that computes the PageRank by using power iteration. In more detail:
# 
# 1. Initialize the PageRank of all nodes to $\frac{1}{n}$, where $n$ is the number of nodes in the network. That is, at the iteration $t=0$ your PageRank vector contains the same value for each node, and it is equally likely to find the walker in each node. (Any other initialization strategy is possible as long as the sum of all elements is one, and the closer the initial vector is to the final vector the faster you will find the final PageRank values.)
# 2. Increase the iteration number $t$ by one and create a new empty PageRank vector $x(t)$.
# 3. Fill in each element of the new vector PageRank vector $x(t)$ using the old PageRank vector $x(t-1)$ and the formula: $x_i(t)=(1-d)\frac{1}{n}+d\sum_{j \in \nu_i}\frac{x_j(t-1)}{k_j^{\mathrm{out}}}$, where $\nu_i$ is the set of nodes that have a directed link ending at $i$, and for each such node $j \in \nu_i$, $k_j^{\mathrm{out}}$ is $j$'s out-degree. In summary, for each node $i$ you need to calculate their entry in the new PageRank vector $x(t)$ as a sum of two parts:
#     - probability that the walker will teleport into the node $(1-d)\frac{1}{n}$ and
#     - probability that the walker will move from a neighbor $j$ to node $i$. Iterate over each in-neighbor $j$ of the node $i$ (i.e., there is a link from $i$ to $j$) and add the neighbors contribution $d\frac{x_j(t-1)}{k_j^{\mathrm{out}}}$ to the entry of the node $i$ in the new PageRank vector $x(t)$.
# 4. Repeat steps 2 and 3 $N_{\mathrm{iterations}}$ times.
# 
# Use your function to compute PageRank} in the example network and
# Visualize the result on the network as in b).
# 
# **Hints**:
# - The damping factor is normally set to $d = 0.85$. 
# - You can monitor the progress of the power iteration by printing out the change in the PageRank vector $\Delta(t)=\sum_i | x_{i}(t) - x_{i}(t-1)|$ after each iteration step. The change $\Delta(t)$ should be decreasing function of $t$. $N_{\mathrm{iterations}} = 10$ should be more than enough in most cases.
# - You can list the incoming edges to node $i$ with the function \code{net.in_edges(i)}, where \code{net} is the network object. Alternatively, you can use the function \code{net.predecessors(i)}, which returns an iterator over predecessors nodes of node $i$. 
# - The sum of all elements in the PageRank vector should always equal to one. There might be slight deviations from this due to numerical errors, but much larger or smaller values is an indication that something is wrong with the code.

# In[10]:


def pagerank_poweriter(g, d, iterations):
    """
    Uses the power iteration method to calculate PageRank value for each node
    in the network.

    Parameters
    -----------
    g : a networkx graph object
    d : damping factor of the simulation
    iterations : number of iterations to perform

    Returns
    --------
    pr_old : dict where keys are nodes and values are PageRank values
    """
    print("Running function for obtaining PageRank by power iteration...")
    
    #TODO: write code for calculating power iteration PageRank

    # Some pseudocode:
    # 1) Create a PageRank dictionary and initialize the PageRank of each node
    #    to 1/n where n is the number of nodes.
    pageRank = {}
    n = g.number_of_nodes()
    pageRank = dict.fromkeys(list(g.nodes),1/n)
    
    # 2) For each node i, find nodes having directed link to i and calculate
    #    sum(x_j(t-1)/k_j^out) where the sum is across the node's neighbors
    #    and x_j(t-1) is the PageRank of node  .
    for t in range(iterations):
        f_pageRank = pageRank.copy()
        for i in g.nodes:
            pageRank[i] = (1 - d) / n
            neighbour_contribution = sum(
                f_pageRank[j] / g.out_degree[j] for j in g.predecessors(i)
            )
            pageRank[i] += d*neighbour_contribution
            
        
    # 3) Update each node's PageRank to (1-d)*1/n + d*sum(x_j(t-1)/k_j^out).
    # 4) Repeat 2-3 n_iterations times.
    
    # sanity checks in each itteration:
    #print('PageRank sums to ')
    #print(sum(pr_new.values()))
    #print(sum(pageRank.values()))
    #print('PageRank difference since last iteration:')
    #print(sum([abs(pr_new[i]-pr_old[i]) for i in g]))
    #print(sum([abs(pageRank[i]-f_pageRank[i]) for i in g]))
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    return pageRank # TODO replace this!


# In[11]:


# PageRank with power iteration
n_iterations = 10
pageRank_pi = pagerank_poweriter(network, d, n_iterations)

# Visualization of PageRank by power iteration

node_colors = [pageRank_pi[node] for node in nodes]
fig = visualize_network(network, node_positions, cmap=cmap, node_size=node_size,
                        node_colors=node_colors, title="PageRank power iteration")

fig.savefig('./network_visualization_pagerank_pi.pdf')


# ### d. Time estimations
# The Google search engine indexes billions of websites and the algorithm for calculating the PageRank needs to be extremely fast. In the original paper about PageRank (Brin, S and Page, L, 2012), by Google founders Larry Page and Sergey Brin, they claim that their "iterative algorithm" is able to calculate the PageRank for 26 million webpages in a few hours using a normal desktop computer (in 1998). 
# 
# Come up with a rough estimate of how long it would take for your power iteration algorithm (part c) and naive random walker algorithm (part b) to do the same. You can assume that the average degree of the 26 million node network is small and that the power iteration converges in the same number of steps as it does for your smaller networks. For the random walk you can assume that you need to run enough steps that the walker visits each node on average $1000$ times. You can also omit any considerations of fitting the large network in memory or the time it takes to read it from the disk etc. With these assumption you can simply calculate the time it takes to run the algorithm in a reasonable size network and multiply the result by the factor that the 26 million node network is bigger than your reasonable sized network.
# 
# 
# **Hints**:
# - There are several ways of timing your code. In Linux you can run your script using the command ``time python myscript.py`` instead of ``python myscript.py`` and read out the "user" value. Even better is to use IPython and run a function calculating everything with the command ``\%timeit calculate\_everything()``, or a script with command ``\%timeit \%run myscript.py``. You can also use the Python ``timeit`` module.
# - The small example network is probably going to be to small to test out the speed of your function especially if you measure the time it takes to run a Python script. (In this case your function might take milliseconds to run but running the whole script might still take a second or so because of starting Python and loading various modules.) You should aim for a network for which it takes several seconds to run the PageRank function. You might find it useful to use network model in networkx to run your code. For example, ``net=nx.directed\_configuration\_model(10**4*[5],10**4*[5],create\_using=nx.DiGraph())`` will produce network with 10000 nodes where each node has in and out degrees of 5 using the configuration model.
# - Don't feel bad if you cannot beat Larry and Sergey in speed when using Networkx and Python. These tools are not meant for speed of computation and even modern computers might not be enough to help. Also, your competition invented Google.

# In[15]:


# Investigating the running time of the power iteration fuction
num_tests = 3 

k5net = nx.directed_configuration_model(10**4*[5],10**4*[5],create_using=nx.DiGraph())
 # TODO: replace with a test network of suitable size
import timeit
t_poweriteration = (timeit.timeit('pagerank_poweriter(k5net,0.85,10)',globals=globals(),number=num_tests)/num_tests)
print(f"Power Iteration for a 10**4 nodes network takes {t_poweriteration} seconds" )
print(f"Power Iteration for a 26*10**6 nodes network takes {((t_poweriteration*26*10**6)/(10**4))} seconds")

# TODO: Print results: how many seconds were taken for the test network of
# 10**4 nodes, how many hours would a 26*10**6 nodes network take? 
# Run num_tests times and use the average or minimum running time in your calculations.

# YOUR CODE HERE
#raise NotImplementedError()

# Investigating the running time of the random walker function
n_nodes = 10**3

n_steps = n_nodes * 1000 # TODO: set such number of steps that each node gets visited on average 1000 times
# YOUR CODE HERE
#raise NotImplementedError()
k5net_n = nx.directed_configuration_model(n_nodes*[5],n_nodes*[5],create_using=nx.DiGraph())
t_naive = (timeit.timeit('pageRank(k5net_n,0.85,n_steps)',globals=globals(),number=num_tests)/num_tests)
print(f"Naive random walker for a 10**3 nodes network takes {t_naive} seconds" )
print(f"Naive random walker for a 26*10**6 nodes network takes {((t_naive*26*10**6)/n_nodes)} seconds")


# ### e. What does it mean?
# Describe how the structure of the network plotted in earlier tasks relates to PageRank. What is the connection between degree $k$ or in-degree $k_{in}$ and PageRank? How does PageRank change if the node belongs to a strongly connected component? How could this information be used in improving the power iteration algorithm given in part c)? 
# 
# **Hints**:
# - Are there ways to incoporate this information to make the algorithm converge faster?

# ### f. The damping factor
# Investigate the role of the damping factor $d$. Repeat the PageRank calculation with *e.g.* 5 different values of $d \in [0,1]$ and plot the PageRank as a function of node index (plots of all values of  $d$ in the same figure). How does the change of $d$ affect the rank of the nodes and the absolute PageRank values?

# In[19]:


def investigate_d(network, ds, colors, n_steps):
    """
    Calculates PageRank at different values of the damping factor d and
    visualizes and saves results for interpretation

    Parameters
    ----------
    network : a NetworkX graph object
    ds : a list of d values
    colors : visualization color for PageRank at each d, must have same length as ds
    n_steps : int; number of steps taken in random walker algorithm
    """
    #import pdb; pdb.set_trace()
    n_nodes = len(network.nodes())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #TODO: write a for loop to obtain node PageRank values at each d and to plot the PageRank.
    # use zip to loop over ds and colors at once
    for d,color in zip(ds,colors):
        pageRankValue = pageRank(network,d,n_steps)
        ax.plot([pageRankValue[str(node)] for node in range(n_nodes)],c=color, label="%.1f" % d)
        
    # YOUR CODE HERE
    #raise NotImplementedError()
    ax.set_xlabel(r'Node index')
    ax.set_ylabel(r'PageRank')
    ax.set_title(r'PageRank with different damping factors')
    ax.legend(loc=0)
    plt.tight_layout

    return fig


# In[21]:


# Investigating effects of d:
ds = np.arange(0, 1.2, 0.2)
colors = ['b', 'r', 'g', 'm', 'k', 'c']
fig = investigate_d(network,ds,colors,1000)
fig.savefig('./network_visualization_d.pdf')
# YOUR CODE HERE
#raise NotImplementedError()


# ### g. Real network scenario
# Now, let's see how PageRank works in a real network. File ``wikipedia_network.edg`` contains the strongly connected component of the Wikipedia hyperlink network around the page Network Science (extracted on May 2nd 2012). 
# - Load the network and list the five most central nodes and their centrality score in terms of PageRank, in-degree, and out-degree. Here you should use ``nx.pagerank``, as the naive algorithm implemented in (a) converges very slowly for a network of this size.
# - Comment and interpret the differences and similarities between the three lists of most central pages.

# In[17]:


# Wikipedia network:
network_path_wp = os.path.join(course_data_dir,'wikipedia_network.edg')
network_wp = nx.read_edgelist(network_path_wp,create_using=nx.DiGraph())# TODO: replace with the network loaded with nx.read_edgelist.

# YOUR CODE HERE
#raise NotImplementedError()


# In[18]:


pageRank_wp = dict(nx.pagerank(network_wp))
indegree_wp = dict(network_wp.in_degree())
outdegree_wp = dict(network_wp.out_degree())
if pageRank_wp is not {}:
    highest_pr = sorted(pageRank_wp, key=lambda k: pageRank_wp[k])[::-1][0:5]
    print('---Highest PageRank:---')
    for p in highest_pr:
        print(pageRank_wp[p], ":", p)
if indegree_wp is not {}:
    highest_id = sorted(indegree_wp, key=lambda k: indegree_wp[k])[::-1][0:5]
    print('---Highest In-degree:---')
    for p in highest_id:
        print(indegree_wp[p], ":", p)
if outdegree_wp is not {}:
    highest_od = sorted(outdegree_wp, key=lambda k: outdegree_wp[k])[::-1][0:5]
    print('---Highest Out-degree:---')
    for p in highest_od:
        print(outdegree_wp[p], ":", p)


# In[ ]:




