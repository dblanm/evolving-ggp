# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

from .datasets import GraphInteraction, IsolatedSubgraphs

import numpy as np
import matplotlib
from matplotlib import rcParams

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.style.use('seaborn-white')
rcParams['figure.figsize'] = 8, 8
rcParams['figure.dpi'] = 300
rcParams['figure.subplot.left'] = 0.15
rcParams['figure.subplot.right'] = 0.9
rcParams['figure.subplot.bottom'] = 0.15
rcParams['figure.subplot.top'] = 0.9
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = ":"
rcParams['xtick.major.bottom'] = True
rcParams['xtick.bottom'] = True
rcParams['ytick.left'] = True
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.size'] = 0.5
rcParams['ytick.minor.width'] = 0.4
rcParams['ytick.major.left'] = True
# Font
rcParams['text.usetex'] = True
rcParams['ps.fonttype'] = 42

# Lines
rcParams['lines.linewidth'] = 3
rcParams['lines.markersize'] = 6

font_value = 24  # Use 50 for the variance plots
# font_value = 50  # Use 50 for the variance plots

rcParams['font.size'] = font_value
# Set specific values for the font relative to font.size
# ## small, medium, large, x-large, xx-large, larger, or smaller
rcParams['axes.titleweight'] = "bold"
rcParams['legend.title_fontsize'] = 100



def plot_velocities(dataset, train_idx, target_id, output_folder):

    if isinstance(dataset, IsolatedSubgraphs):
        if target_id == 2:
            test_id_string = "V_x"
        elif target_id == 3:
            test_id_string = "V_y"
    else:
        if target_id == 3:
            test_id_string = "V_x"
        elif target_id == 4:
            test_id_string = "V_y"
        elif target_id == 5:
            test_id_string = "V_z"

    nodes_list = []

    for i in range(dataset.timesteps):
        nodes_list.append(dataset.graph_list[i].nodes)

    nodes_arr = np.array(nodes_list)

    node_data = nodes_arr[:, :, target_id]

    plot_nodes_data(node_data, test_id_string, train_idx, title="Velocity",
                    fig_title="Graph_velocity", ylabel="Velocity (m/s)",
                    output_folder=output_folder)


def plot_nodes_data(data, label, training_idxs, title, fig_title, ylabel, output_folder):
    x = np.linspace(0, data.shape[0], data.shape[0])

    fig, ax = plt.subplots()
    for j in range(data.shape[1]):  # Nr of nodes
        node_vel = data[:, j]

        ax.plot(x, node_vel, label="Node " + str(j))

    for x_id in training_idxs:
        ax.axvline(x=x_id, color="red", linestyle="--")

    plt.legend(frameon=True, prop={'size': 6})
    plt.grid()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("timestep")
    plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)

    plt.savefig(output_folder + fig_title + "_" + label)

    plt.close(fig)


def plot_2d(ax, nodes, adjacency, color, label, x_axis=0, y_axis=1, marker_size=0.5, line_width=0.3):

    if color == "darkred":
        alpha_color = "lightcoral"
    elif color == "darkgreen":
        alpha_color = "limegreen"
    elif color == "darkorange":
        alpha_color = "gold"
    elif color == "blue":
        alpha_color = "aqua"
    else:
        alpha_color = color

    for i in range(0, len(adjacency)):  # Node i
        x_i = nodes[i, [x_axis, y_axis]]
        for j in adjacency[i]:  # Neighbour nodes to i
            if i > 14 and j > 14:
                continue
            x_nn = nodes[j, [x_axis, y_axis]]
            x_conn = np.vstack((x_i, x_nn))
            ax.plot(x_conn[:, 0], x_conn[:, 1], marker='None', color=alpha_color, ls='-',
                    linewidth=line_width, zorder=1)

    ax.scatter(nodes[:15, x_axis], nodes[:15, y_axis], s=marker_size, label=label, edgecolor=color, color=color,
               zorder=5)


def plot_2d_sphere_only(ax, nodes, adjacency, marker_size=0.5, line_width=0.3):
    color = "blue"
    alpha_color = "lightskyblue"
    x_axis = 0
    y_axis = 2

    for i in range(0, len(adjacency)):  # Node i

        x_i = nodes[i, [x_axis, y_axis]]
        for j in adjacency[i]:  # Neighbour nodes to i
            if i > 14 and j > 14:
                pass
            else:
                continue
            x_nn = nodes[j, [x_axis, y_axis]]
            x_conn = np.vstack((x_i, x_nn))

            ax.plot(x_conn[:, 0], x_conn[:, 1], marker='None', color=alpha_color, ls='-',
                    linewidth=line_width, zorder=1)

    ax.scatter(nodes[15:, x_axis], nodes[15:, y_axis], s=marker_size, edgecolor=color, color=color,
               zorder=3)


def plot_2d_dataset(dataset, graph_idxs, output_folder, save_add="", legend=True, dataset_list_names=""):
    tab_colours = mcolors.TABLEAU_COLORS.keys()  # List of 10 colors
    colors = []
    for tab_c in tab_colours:
        color = tab_c.replace('tab:', '')
        colors.append(color)
    css_colors = ['darksalmon', 'goldenrod', 'greenyellow', 'lime',
                  'aquamarine', 'turquoise', 'orchid']
    colors.extend(css_colors)
    colors[0] = "darkred"
    colors[1] = "darkorange"
    colors[2] = "darkgreen"

    fig, ax = plt.subplots()

    name_dataset = ""
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    # Define the index where to get the X/Y or X/Z axis
    x_axis = 0
    y_axis = 1
    marker_size = 70  # Origin

    line_width = 3 # Origin
    if isinstance(dataset, IsolatedSubgraphs):
        conn_r = str(dataset.conn_r).replace(".", "_")
        k_nn = str(dataset.knn)
        name_dataset = "Isolated_dataset_conn_" + conn_r + "_Knn_" + k_nn

        x_min, x_max, y_min, y_max = 0, 1.0, 0.0, 1.0
    elif isinstance(dataset, GraphInteraction):
        name_dataset = "Graph_interaction_dataset"
        # x_min, x_max, y_min, y_max = -0.23, 0.23, 0.37, 0.83
        # x_min, x_max, y_min, y_max = -0.23, 0.23, 0.2, 0.9
        x_min, x_max, y_min, y_max = -0.23, 0.23, 0.4, 0.72
        y_axis = 2
    else:
        print("Unknown dataset type")

    if isinstance(dataset, list):
        colors = ['darkcyan',
                  'lightseagreen',
                  'forestgreen',
                  'darkgreen',
                  'darkgoldenrod',
                  'darkorange',
                  'darkred',
                  'crimson',
                  'darkmagenta']
        list_tests = []
        name_dataset += "Multiple"
        for idx in range(len(dataset)):
            test_values = ""
            test_name = dataset_list_names[idx]
            if "rope_x0_y0" in test_name:
                test_values = 0
            elif "rope_offset_Z_0_1" in test_name:
                test_values = 0.1
            elif "rope_offset_Z_0_2" in test_name:
                test_values = 0.2
            elif "rope_offset_Z_0_3" in test_name:
                test_values = 0.3
            elif "rope_offset_Z_0_05" in test_name:
                test_values = 0.05
            elif "rope_offset_Z_0_15" in test_name:
                test_values = 0.15
            elif "rope_offset_Z_0_25" in test_name:
                test_values = 0.25
            elif "rope_offset_Z_min_0_1" in test_name:
                test_values = -0.1
            elif "rope_offset_Z_min_0_05" in test_name:
                test_values = -0.05
            list_tests.append(test_values)
        test_list_array = np.array(list_tests)
        test_list_order = (-test_list_array).argsort()
        dataset = [dataset[test_list_order[i]] for i in range(len(test_list_order))]
        list_tests = [list_tests[test_list_order[i]] for i in range(len(test_list_order))]
        # colors = [colors[test_list_order[i]] for i in range(len(test_list_order))]

    if isinstance(dataset, list):
        name_dataset += "Multiple"
        for idx in range(0, len(dataset)):
            if isinstance(dataset[idx], GraphInteraction):
                name_dataset = "Rope_dataset"
                x_min, x_max, y_min, y_max = -0.32, 0.25, -0.1, 1.1
                y_axis = 2
            for j in range(len(graph_idxs)):
                graph_t = dataset[idx].graph_list[graph_idxs[j]]
                nodes = graph_t.nodes
                adjacency = graph_t.adj_list
                if isinstance(dataset[idx], GraphInteraction):
                    plot_2d_sphere_only(ax, nodes, adjacency, marker_size=marker_size, line_width=line_width)
                plot_2d(ax, nodes, adjacency=adjacency, color=colors[idx], label="Offset " + str(list_tests[idx]),
                        x_axis=x_axis, y_axis=y_axis, marker_size=marker_size, line_width=line_width)
    else:
        for idx in range(len(graph_idxs)):
            graph_t = dataset.graph_list[graph_idxs[idx]]
            nodes = graph_t.nodes
            adjacency = graph_t.adj_list
            if isinstance(dataset, GraphInteraction):
                plot_2d_sphere_only(ax, nodes, adjacency, marker_size=marker_size, line_width=line_width)
            plot_2d(ax, nodes, adjacency=adjacency, color=colors[idx], label="Timestep " + str(graph_idxs[idx]),
                    x_axis=x_axis, y_axis=y_axis, marker_size=marker_size, line_width=line_width)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)

    if legend:
        plt.legend(frameon=True)

    plt.savefig(output_folder + name_dataset + "_graph_timesteps_" + save_add + str(graph_idxs[0])+".eps")

    plt.close(fig)


def plot_compare_2d_dataset(gt_graphs, gt_adjs, pred_graphs, pred_adjs,
                            name_dataset, save_folder, save_add=""):
    title_dataset = ""
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    # Define the index where to get the X/Y or X/Z axis
    x_axis = 0
    y_axis = 1
    if "isolated" in name_dataset:
        title_dataset = "2D Isolated sub-graph scene "

        x_min, x_max, y_min, y_max = 0, 1.0, 0.0, 1.0
    elif "rope" in name_dataset:
        title_dataset = "2D Graph interaction scene"
        x_min, x_max, y_min, y_max = -0.23, 0.23, 0.2, 0.9
        y_axis = 2
    else:
        print("Unknown dataset type")
    marker_size = 1
    line_width = 0.5

    for i in range(0, len(gt_graphs)):
        fig, axes = plt.subplots(1, 2)
        # Plot the ground-truth
        plot_2d(axes[0], gt_graphs[i], adjacency=gt_adjs[i], color="blue", label="Timestep " + str(i),
                x_axis=x_axis, y_axis=y_axis, marker_size=marker_size, line_width=line_width)
        plot_2d(axes[1], pred_graphs[i], adjacency=pred_adjs[i], color="green", label="Timestep " + str(i),
                x_axis=x_axis, y_axis=y_axis, marker_size=marker_size, line_width=line_width)
        fig.suptitle(title_dataset)
        for ax in axes:
            ax.set_ylabel("Z")
            ax.set_xlabel("X")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend()

        plt.savefig(save_folder + name_dataset + "_comparison_" + save_add + str(i))

        plt.close(fig)


def plot_mean_and_variance(mean, var, gt, model_name, target_name, dataset_name, main_folder, plot_variance=False):

    folder = main_folder + dataset_name + "/" + model_name + "/" + target_name + "/"
    folder = folder.replace(" ", "_")

    if var.min() < 0:
        print("Found negative variance, this should not happen")
        var = np.abs(var)

    for node in range(0, mean.shape[1]):
        fig = plt.figure()
        fig.set_size_inches(20, 20, forward=True)
        fig.clf()
        axes = fig.gca()

        axes.plot(range(len(mean)), gt[:, node], color='blue', label='Target', marker='+', zorder=3)

        axes.plot(range(len(mean)), mean[:, node], lw=2, label='Estimated', marker='+', color='darkorange', zorder=3)
        if plot_variance:
            mean_node = mean[:, node]
            var_node = var[:, node]
            var_1 = (mean_node - 2 * np.sqrt(var_node))
            var_2 = (mean_node + 2 * np.sqrt(var_node))
            axes.fill_between(range(len(mean)), var_1, var_2, color='gold', zorder=0)
        plt.grid(True)
        plt.ylabel("Velocity m/s")
        plt.xlabel("Timestep")
        axes.legend()
        plt.savefig(folder + "Node_prediction_" + str(node))
        plt.close(fig)
