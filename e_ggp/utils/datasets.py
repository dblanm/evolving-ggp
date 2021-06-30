# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import abc
import igraph
import pickle
import numpy as np
from scipy.spatial import KDTree


class Graph:

    def __init__(self, nodes, edges, edge_weights=None, e_rcvs=None, e_send=None):

        self.nodes = nodes

        self.edges = edges

        self.edge_weights = edge_weights

        self.e_rcvs = e_rcvs

        self.e_sends = e_send

        self.ig = igraph.Graph()

        str_vertx = [np.array2string(v) for v in nodes]
        edges_tuple = tuple(map(tuple, edges))

        self.ig.add_vertices(str_vertx)
        self.ig.add_edges(edges_tuple)

    @property
    def degree(self):
        return np.diag(self.ig.degree())  # Get the diagonal degree matrix

    @property
    def adjacency(self):
        return np.array(self.ig.get_adjacency(type=igraph.GET_ADJACENCY_BOTH))

    @property
    def adj_list(self):
        ig_list = self.ig.get_adjlist()
        adj_list = []
        for i in range(0, len(ig_list)):
            adj_list.append(ig_list[i])
        return np.array(adj_list, dtype=object)

    @property
    def laplacian(self):
        return np.array(self.ig.laplacian())

    def plot_graph(self):
        self.ig.vs["label"] = [i for i in range(0, self.nodes.shape[0])]

        layout = self.ig.layout("kk")  # Kamada-Kawai layout
        igraph.plot(self.ig, layout=layout)


class GraphDataset:
    """Abstract class for graph datasets.
    Given a numpy file name, load the file and create a graph structure based on a connectivity distance Rnn.
    """
    # We want to learn the mapping f(G): G_t -> G_{t+1}

    def __init__(self, fd, keep_adj=False):
        self.raw_dataset = np.load(fd).astype(np.float32)
        self.timesteps = None
        self.node_nr = None
        self.node_dim = None  # Nr of nodes
        self.keep_adj = keep_adj

        self.last_pos_dim_idx = 3

        # Create the graph, node and edge list
        self.graph_list = []
        self.node_list = []
        self.edge_list = []
        # Create single edge list in case there is a prior on the graph connectivity
        self.single_edge_list = []

    def load_dataset(self, node_fd, edge_fd):
        with open(node_fd, "rb") as fn:
            node_list = pickle.load(fn)
        with open(edge_fd, "rb") as fe:
            edge_list = pickle.load(fe)

        for i in range(0, len(node_list)):  # Uncomment this for regular dataset
            self.graph_list.append(Graph(node_list[i], edge_list[i]))

    def create_dataset(self):
        data_stack = np.stack(self.raw_dataset)

        self.timesteps = data_stack.shape[0]
        self.node_nr = data_stack.shape[1]  # Nr of nodes
        self.node_dim = data_stack.shape[2]  # X + Z

        for t in range(0, self.timesteps):  # Loop over all timesteps
            nodes = np.zeros((self.node_nr, self.node_dim))  # Create an array of nodes (tuple XY) per each timestep

            for i in range(0, self.node_nr):  # Loop over all nodes
                data = tuple([data_stack[t, i, d] for d in range(0, self.node_dim)])
                nodes[i, :] = np.hstack((data))  # Set the XY tuple

            self.node_list.append(nodes)  # Append the XY tuples to the list

            # Build the kdtree for each timestep
            if self.keep_adj:
                edge_arr = self.single_edge_list.copy()
            else:
                edge_arr = self.create_edge_connections(nodes)
            self.edge_list.append(edge_arr)  # Add the edge list to the edges list

            # Add a graph to the list using the node and edge arrays
            self.graph_list.append(Graph(nodes, edge_arr))

    @abc.abstractmethod
    def create_edge_connections(self, nodes):
        """ You should implement here a method that based on the input nodes creates a list of the
        edges. The edges should not repeat.
        :param nodes: array of nodes with 2D+d attributes for a timestep.
        :return: list of non-repeating edges
        """
        raise NotImplementedError


class GraphInteraction(GraphDataset):

    def __init__(self, fd, prior=True, keep_adj=False):
        """
        Instance of the 2D graph interaction dataset, rope falling on a static ball.
        :param fd: file descriptor
        :param prior: whether prior knowledge of the connectivity is used or not
        :param keep_adj: whether to keep or not the same adjacency on all the timesteps
        """

        super(GraphInteraction, self).__init__(fd=fd, keep_adj=keep_adj)
        self.conn_r = 0.043  # rope

        self.prior = prior

        self.k_folding = 2  # Maximum two when in contact
        self.k_grid = 2  # Maximum two on the side

        if self.prior or self.keep_adj:  # We define the prior connectivity of the rope, Rope: 0-14, Ball: 15-30
            self.single_edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                              [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [15, 16], [15, 30],
                              [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23],
                              [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30]]
            # In case of prior knowledge re-define the maximum connectivity
            self.k_folding = 1  # Maximum one when in contact
            self.k_grid = 1  # Maximum one on the side
        else:  # Empty single edge list
            self.single_edge_list = []

        # Maximum k-nn connectivity
        self.knn = self.k_folding + self.k_grid + 1  # K-neighbours, + itself

    def create_edge_connections(self, nodes):

        position_nodes = nodes[:, :self.last_pos_dim_idx]
        tree = KDTree(position_nodes, leafsize=10)

        edge_list = self.single_edge_list.copy()

        # Check all the neighbours for each node
        for i in range(0, position_nodes.shape[0]):
            q_node = position_nodes[i, :]
            neighbs = tree.query(q_node, self.knn, p=2, distance_upper_bound=self.conn_r)
            k_nn = neighbs[1]
            d_nn = neighbs[0]

            copy_dnn = d_nn.copy()
            copy_dnn[copy_dnn >= 1E308] = 0

            for k_node in range(0, self.knn):
                k_idx = k_nn[k_node]  # Index of the neighbour
                if k_idx == i:  # Skip the node
                    continue
                elif d_nn[k_node] == np.inf:
                    continue
                else:
                    # Order the min idx first to check the list
                    if k_idx > i:
                        edge = [i, k_idx]
                    else:
                        edge = [k_idx, i]
                    # Check if the connection already exists
                    if edge in edge_list:
                        continue
                    else:
                        edge_list.append(edge)

        # Create an array from the edge list
        edge_arr = np.array(edge_list)

        return edge_arr


class IsolatedSubgraphs(GraphDataset):

    def __init__(self, fd, prior=False, keep_adj=False, conn_r=0.015, k_nn=10):
        """
        Instance of the Isolated Sub-graphs data set: 2D water from Taichi.
        The dataset is composed by the following attributes:
        [x, y, x_dot, y_dot, dist_to_right, dist_to_top, dist_to_left, dist_to_bottom]
        :param fd: file descriptor
        :param prior: whether prior knowledge of the connectivity is used or not
        :param keep_adj: whether to keep or not the same adjacency on all the timesteps
        """

        super(IsolatedSubgraphs, self).__init__(fd=fd, keep_adj=keep_adj)
        self.last_pos_dim_idx = 2  #
        # self.pose_data = [0, 1]  # We use the position for checking the connectivity
        self.conn_r = conn_r  # 0.003

        self.prior = prior

        if self.prior or self.keep_adj:  # We define the prior connectivity of the rope, Rope: 0-14, Ball: 15-30
            self.single_edge_list = []
        else:  # Empty single edge list
            self.single_edge_list = []

        self.knn = k_nn  # 10 neighbours at maximum

    def create_edge_connections(self, nodes):

        position_nodes = nodes[:, :self.last_pos_dim_idx]
        tree = KDTree(position_nodes, leafsize=10)

        edge_list = self.single_edge_list.copy()

        # Check all the neighbours for each node
        for i in range(0, position_nodes.shape[0]):
            q_node = position_nodes[i, :][None, :]  # Shape 1x2
            neighbs = tree.query(q_node, self.knn, p=2, distance_upper_bound=self.conn_r)

            k_nn = neighbs[1]
            d_nn = neighbs[0]

            for k_node in range(0, self.knn):
                k_idx = k_nn[0, k_node]  # Index of the neighbour
                if k_idx == i:  # Skip the node
                    continue
                elif d_nn[0, k_node] == np.inf:
                    continue
                else:
                    # Order the min idx first to check the list
                    if k_idx > i:
                        edge = [i, k_idx]
                    else:
                        edge = [k_idx, i]
                    # Check if the connection already exists
                    if edge in edge_list:
                        continue
                    else:
                        edge_list.append(edge)

        # Create an array from the edge list
        edge_arr = np.array(edge_list)

        return edge_arr

    def create_dataset(self):
        data_stack = np.stack(self.raw_dataset)

        self.timesteps = data_stack.shape[0]
        self.node_nr = data_stack.shape[1]  # Nr of nodes
        self.node_dim = data_stack.shape[2]  # X + Z

        for t in range(0, self.timesteps):  # Loop over all timesteps
            nodes = np.zeros((self.node_nr, self.node_dim))  # Create an array of nodes (tuple XY) per each timestep

            for i in range(0, self.node_nr):  # Loop over all nodes
                data = tuple([data_stack[t, i, d] for d in range(0, self.node_dim)])
                nodes[i, :] = np.hstack((data))  # Set the XY tuple

            self.node_list.append(nodes)  # Append the XY tuples to the list

            # Build the kdtree for each timestep
            if self.keep_adj and t == 0:  # Get the connectivity at t=0
                edge_arr = self.create_edge_connections(nodes)
                self.single_edge_list = edge_arr.copy()
            elif self.keep_adj:
                edge_arr = self.single_edge_list.copy()
            else:
                edge_arr = self.create_edge_connections(nodes)
            self.edge_list.append(edge_arr)  # Add the edge list to the edges list

            # Add a graph to the list using the node and edge arrays
            self.graph_list.append(Graph(nodes, edge_arr))







