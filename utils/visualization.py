import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from netgraph import Graph

from src.evolution import *

def draw_graph(g, pos):
    groups = set(nx.get_node_attributes(g,'state').values())
    mapping = {0: 0, 1: 1}
    nodes=g.nodes
    colors=[mapping[nodes[n]['state']] for n in nodes]
    ec = nx.draw_networkx_edges(g, pos, alpha=1)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.Greys)
    
def animation(G, rule, total_frames=100, interval=200, name='animation.gif'):
    fig, ax = plt.subplots()
    g=Graph(G)
    cmap = plt.cm.Greys_r
    def update(i):
        for node, artist in g.node_artists.items():
            value = G.nodes[node]['state']
            artist.set_facecolor(cmap(value*0.99))
            #artist.set_edgecolor(cmap(value*0.99))
        step(G, rule)
        return g.node_artists.values()

    animation = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=True)
    animation.save(name)
    
def draw_graph_evolution(G, rule, nsteps=100, title=None):
    # This should be simplified:
    N=0
    for node in G.nodes:
        N+=1
        
    values=np.zeros((nsteps, N))
    for i in range(nsteps):
        for j in range(N):
            values[i,j]=G.nodes[j]['state']
        step(G, rule)
        
    fig=plt.figure()
    ax=plt.axes()
    ax.set_axis_off()
    fig.suptitle(title)
    ax.grid(color='k', linestyle='-', linewidth=1)
    ax.imshow(values, interpolation='none', cmap='RdPu');

def plot_ECA(states, title=None):
    fig=plt.figure()
    ax=plt.axes()
    ax.set_axis_off()
    fig.suptitle(title)
    ax.grid(color='k', linestyle='-', linewidth=1)
    ax.imshow(states, interpolation='none', cmap='RdPu');