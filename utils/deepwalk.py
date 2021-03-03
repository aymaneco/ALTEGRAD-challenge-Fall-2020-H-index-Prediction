import os
import numpy as np
import ast
import re
from random import randint
from gensim.models import Word2Vec

# These functions are modified version of the functions defined in Lab5 of the course.

# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length, weighted=True):
    """
        G : the graph,
        node : the node from which the generated walk starts,
        walk_length : the length of the walk
        weighted : If True we get a weighted Deep Walk, else: unweighted
        walk : The generated walk

    """
    walk = [node]
    for _ in range(walk_length):
        neighbors = list(G.neighbors(walk[-1]))
        n = len(neighbors)
        if weighted:
            if n > 0:
              weights = np.zeros((n))
              for i in range(n):
                  weights[i] = G[walk[-1]][neighbors[i]]["weight"]
              weights = weights / np.sum(weights)
              random_neighbor = np.random.choice(np.arange(0, n), size=1, p=weights)[0]
              walk.append(neighbors[random_neighbor])
            else:
                break
        else:
          random_neighbor = neighbors[randint(0, n-1)]
          walk.append(random_neighbor)
    walk = [str(node) for node in walk]
    return walk

# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length,weighted=True):
    """
        G : the graph,
        num_walks : the number of walks to be generated,
        walk_length : the length of the walks
        walks : the generated walks

    """
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        permuted_nodes = np.random.permutation(nodes)
        for node in permuted_nodes:
            walks.append(random_walk(G, node, walk_length,weighted))
    return walks

# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim,weighted=True):
    """
        G : the graph,
        num_walks : the number of walks to be generated
        walk_length : the length of the walks,
        n_dim : the nodes embeddings dimension
    """
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length,weighted)
    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)
    return model
