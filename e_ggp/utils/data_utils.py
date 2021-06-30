# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import os
import re
import dill
import torch
import numpy as np
import pandas as pd
from csv import writer
from datetime import datetime

from .datasets import IsolatedSubgraphs, GraphInteraction
from .plot_utils import plot_nodes_data, plot_mean_and_variance


def create_csv_file(output_folder):
    file_name = output_folder + "results.csv"
    # Check if the directory exists
    isdir = os.path.isdir(output_folder)
    if not isdir:
        os.mkdir(output_folder)
    try:
        f = open(file_name)
        f.close()
    except IOError:
        print("CSV does not exist, creating it")
        with open(file_name, 'w') as csvfile:
            filewriter = writer(csvfile, delimiter=',',
                                quotechar='|')
            # Test data to write =
            # "Date/Time", "Target", "Dataset", "Model name", "Data size", "RMSE", "MAPE", "MAE", "NLL"
            filewriter.writerow(["Date-Time", "Target", "Dataset", "Model name", "Data size",
                                 "RMSE", "MAPE", "MAE", "NLL"])


def get_model_and_writer_name(args, main_folder, model_type, kernel_1, kernel_2):
    if args.full_data:
        data_str = "full_data"
    else:
        data_str = "data_" + str(args.data_limit)
    priors_str = "_priors_" + str(args.priors) + "_fixed_adj_" + str(args.fixed_adj) + "_"
    if "isolated" in args.data_file:
        priors_str += "conn_" + str(args.conn_r).replace(".", "_") + "_"
        priors_str += "k_nn_" + str(args.k_nn) + "_"
    model_name = model_type + data_str + priors_str + kernel_1 + "_" + kernel_2

    dateTimeObj = datetime.now()

    date_test = str(dateTimeObj.month) + "m_" + str(dateTimeObj.day) + "d_" + str(dateTimeObj.hour) \
                + "h_" + str(dateTimeObj.minute) + "m"
    writer_name = main_folder + model_name + "_" + date_test

    return model_name, writer_name


def create_training_data(args):
    df_train = args.data_file
    train_pickle = df_train[:-4] + '_dataset_priors_' + str(args.priors) + \
                   '_adj_fixed_' + str(args.fixed_adj)
    if "isolated" in df_train:
        train_pickle += "_conn_" + str(args.conn_r).replace(".", "_") + "_"
        train_pickle += "_k_nn_" + str(args.k_nn) + "_"

    train_pickle += '.pkl'
    train_dataset_name = re.split('/', df_train)
    train_dataset_name = train_dataset_name[-1][:-4]

    # Create the graph
    if "isolated" in df_train:
        train_dataset = IsolatedSubgraphs(df_train, prior=args.priors, keep_adj=args.fixed_adj,
                                          conn_r=args.conn_r, k_nn=args.k_nn)
    else:
        train_dataset = GraphInteraction(df_train, prior=args.priors, keep_adj=args.fixed_adj)

    if args.new_dataset:
        print("Creating train dataset")
        train_dataset.create_dataset()
        if not args.no_pkl:
            with open(train_pickle, 'wb') as fn:
                dill.dump(train_dataset, fn)
    else:
        print("Loading train dataset")
        # Load the dataset from the saved pickle
        with open(train_pickle, 'rb') as fn:
            train_dataset = dill.load(fn)

    return train_dataset, train_dataset_name


def create_test_data(args):
    df_train = args.data_file
    test_files, test_pickles, test_dataset_names, test_dataset = [], [], [], []
    for file in os.listdir(args.data_test_folder):
        if ".npy" in file:
            test_files.append(args.data_test_folder + file)
            fd_pickle = args.data_test_folder + file[:-4] + '_dataset_priors_' + \
                        str(args.priors) + '_adj_fixed_' + str(args.fixed_adj)
            if "isolated" in df_train:
                fd_pickle += "_conn_" + str(args.conn_r).replace(".", "_") + "_"
                fd_pickle += "_k_nn_" + str(args.k_nn) + "_"
            fd_pickle += '.pkl'
            test_pickles.append(fd_pickle)
            test_dataset_names.append(file[:-4])
        else:
            continue

    if args.new_dataset:
        print("Creating test dataset")
        # Create the dataset and dump the dataset into the pkl
        test_dataset = []
        for file in test_files:
            if "isolated" in df_train:
                test_dataset.append(IsolatedSubgraphs(file, prior=args.priors, keep_adj=args.fixed_adj,
                                                      conn_r=args.conn_r, k_nn=args.k_nn))
            else:
                test_dataset.append(GraphInteraction(file, prior=args.priors, keep_adj=args.fixed_adj))

        # Same for the test data
        for i in range(len(test_dataset)):
            test_dataset[i].create_dataset()
            if not args.no_pkl:
                with open(test_pickles[i], 'wb') as fn:
                    dill.dump(test_dataset[i], fn)
    else:
        print("Loading test dataset")
        # Load the dataset from the saved pickle
        for i in range(len(test_pickles)):
            with open(test_pickles[i], 'rb') as fn:
                test_pickle_tmp = dill.load(fn)
                test_dataset.append(test_pickle_tmp)

    return test_dataset, test_dataset_names


def save_torch_model(model, folder, extra_name=""):

    torch.save(model.state_dict(), folder+"/model_state"+extra_name+".pth")


def check_nodes_diff(nodes_arr, target_id):
    target_nodes = nodes_arr[:, :, target_id]

    # Get the derivative of the nodes
    nodes_diff = np.diff(target_nodes[:, :], axis=0)

    # Get the absolute values and check which node has a highest value at each timestep
    diff_nodes_max = np.abs(nodes_diff).argmax(axis=1)
    rows_idxs = np.linspace(0, nodes_diff.shape[0] - 1, nodes_diff.shape[0], dtype=int)
    max_diff_per_node = np.abs(nodes_diff[rows_idxs, diff_nodes_max])
    # Sort the values
    diff_sorted = np.flip(max_diff_per_node.argsort())

    return nodes_diff, diff_sorted


def get_new_training_points(diff_sorted_in, train_list, data_limit, space_idxs):
    """ Get new training points based on the derivative sorted
    :param diff_sorted_in: array that contains the derivative of each training point sorted
    :param train_list: list where to append the new training points
    :param data_limit: requested data limit
    :param space_idxs: space
    :return:
    """
    diff_sorted = diff_sorted_in.copy()
    while len(train_list) < data_limit:

        if diff_sorted.shape[0] < 2:
            print("There is no more training data ", str(space_idxs), " timesteps from each other")
            print("Current size of training data is:=", len(train_list))
            print("Requested size of training data is:=", data_limit)
            break

        new_idx = diff_sorted[0]
        # Check we are adding the first item
        if len(train_list) == 0:
            train_list.append(new_idx)
            # Remove the points close to this
            diff_sorted = diff_sorted[diff_sorted != new_idx]
            continue
        # Remove the current index from the existing indexes
        test_array = np.abs(np.array(train_list) - new_idx)
        any_existing = np.any(test_array < space_idxs)
        # Check if the index is too close (10 timesteps)
        if any_existing:
            diff_sorted = diff_sorted[diff_sorted != new_idx]
        else:
            # If not, append it and remove it from the current indexes
            train_list.append(new_idx)
            diff_sorted = diff_sorted[diff_sorted != new_idx]

    done = len(train_list) == data_limit

    return train_list, done


def get_target_name(dataset, target_idx):
    target_name = ""
    if isinstance(dataset, IsolatedSubgraphs):
        if target_idx == 2:
            target_name = "V_x"
        elif target_idx == 3:
            target_name = "V_y"
    else:
        if target_idx == 3:
            target_name = "V_x"
        elif target_idx == 4:
            target_name = "V_y"
        elif target_idx == 5:
            target_name = "V_z"
    return target_name


def get_data_targets(dataset, data_limit, target_id, plot, output_folder):

    test_id_string = get_target_name(dataset, target_id)

    # Create the nodes array data
    nodes_list = []
    for i in range(dataset.timesteps):
        nodes_list.append(dataset.graph_list[i].nodes)
    nodes_arr = np.array(nodes_list)

    nodes_diff, diff_sorted = check_nodes_diff(nodes_arr, target_id)

    # Half of the data-limit are the points where the derivative is highest
    train_idx = []

    for i in range(20, 0, -2):
        train_idx, done = get_new_training_points(diff_sorted, train_idx, data_limit, space_idxs=i)
        if done:
            break
    print("The final size of the training input is=", len(train_idx))
    # Plot the derivative
    if plot:
        plot_nodes_data(nodes_diff, test_id_string, train_idx, title="Derivative of Velocity",
                        fig_title="Node_diff", ylabel="Velocity change", output_folder=output_folder)

    return np.array(train_idx)


def get_xy_from_data(idx, step_size, dataset, target_idx):
    query_test = idx
    query_target = query_test + step_size
    x_test = dataset.graph_list[query_test]
    y_test = dataset.graph_list[query_target]

    # Create the test data
    nodes = x_test.nodes
    adj = x_test.adj_list
    # Just one dimension of the nodes target
    target = y_test.nodes[:, target_idx]
    target = np.atleast_2d(target).T

    return nodes, adj, target


def extend_train_dataset(graph_dataset, step_size, dataset_train_idx, target_idx, batch=False, con_attr=False):

    X = []
    Y = []
    adj_list = []
    counter = 1

    for idx in dataset_train_idx:

        nodes_train, adj_train, nodes_target = get_xy_from_data(idx, step_size, graph_dataset, target_idx)

        D = nodes_train.shape[1]
        # If batch is enabled the shape of X will be (M, N, D), where M is the number of samples,
        # N is the number of nodes, and D is the number of attributes of each node
        if batch:
            # We can construct the input as a vector that concatenates the features of all the nodes
            if con_attr:
                nodes_attr_train = nodes_train.flatten()
                nodes_attr_target = nodes_target.flatten()
                # Since we have concatenated the features we need to repeat the adjacency list for each attribute
                # adj_attr = np.repeat(adj_train, D)
                adj_attr = adj_train  # Testing keeping the same adjacency as it was

            else:
                nodes_attr_train = nodes_train
                nodes_attr_target = nodes_target
                adj_attr = adj_train

            if counter > 1:
                adj_list = np.vstack((adj_list, [[nn + len(X) for nn in adj_nn] for adj_nn in
                                                 adj_attr]))  # Modify the index of the adj list
                X = np.concatenate((X, nodes_attr_train[None, :]), axis=0)
                Y = np.concatenate((Y, nodes_attr_target[None, :]), axis=0)
            else:
                X = nodes_attr_train[None, :]
                Y = nodes_attr_target[None, :]
                adj_list = adj_attr

        else:
            if counter > 1:
                # Modify the index of the adj list
                adj_list = np.hstack((adj_list, [[nn + len(X) for nn in adj_nn] for adj_nn in adj_train]))
                X = np.vstack((X, nodes_train))
                Y = np.vstack((Y, nodes_target))
            else:
                X = nodes_train
                Y = nodes_target
                adj_list = adj_train

        counter += 1

    return X, Y, adj_list


def save_onestep_data(main_folder, error_pred_lists_dict, model_name, dataset_name, target_name, dataset_size,
                      plot, plot_variance):
    """

    """
    folder = main_folder + dataset_name + "/" + model_name + "/" + target_name + "/"
    folder = folder.replace(" ", "_")
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Prediction of nodes and variances
    gp_pred_arr = np.array(error_pred_lists_dict['gp_pred'])[:, :]

    gp_var_arr = np.array(error_pred_lists_dict['gp_var'])
    gt_arr = np.array(error_pred_lists_dict['gt'])[:, :, 0]
    if len(gp_pred_arr.shape) > 2:
        gp_pred_arr = gp_pred_arr[:, :, 0]
    if len(gp_var_arr.shape) > 2:
        gp_var_arr = gp_var_arr[:, :, 0]

    # Error arrays
    rmse_arr = np.array(error_pred_lists_dict['rmse'])
    mape_arr = np.array(error_pred_lists_dict['mape'])
    mae_arr = np.array(error_pred_lists_dict['mae'])
    nll_arr = np.array(error_pred_lists_dict['nll'])

    if plot:
        plot_mean_and_variance(gp_pred_arr, gp_var_arr, gt_arr,
                               model_name, target_name, dataset_name, main_folder=main_folder,
                               plot_variance=plot_variance)
        # Plot the variance as well in a different folder
        target_variance = target_name + "_variance"
        folder_variance = main_folder + dataset_name + "/" + model_name + "/" + target_variance + "/"
        folder_variance = folder_variance.replace(" ", "_")
        if not os.path.isdir(folder_variance):
            os.makedirs(folder_variance)

        plot_mean_and_variance(gp_pred_arr, gp_var_arr, gt_arr,
                               model_name, target_variance, dataset_name, main_folder=main_folder,
                               plot_variance=True)
    # Save the results
    save_results(main_folder, target_name, dataset_name, model_name, dataset_size,
                 rmse_arr, mape_arr, mae_arr, nll_arr)

    # Save results data to a pickle
    df = pd.DataFrame([gp_pred_arr, gp_var_arr, gt_arr, rmse_arr, mape_arr, mae_arr, nll_arr])
    filename = folder + model_name + " " + dataset_name + target_name + ".pkl"
    filename = filename.replace(" ", "_")

    df.to_pickle(filename)


def save_results(main_folder, target_name, dataset_name, model_name, data_size,
                 rmse_arr, mape_arr, mae_arr, nll_arr):
    # "Date/Time", "Target", "Dataset", "Model name", "Data size", "RMSE", "MAPE", "MAE", "NLL"
    file_name = main_folder + "results.csv"
    with open(file_name, 'a+', newline='') as csvfile:
        csvwriter = writer(csvfile, delimiter=',', quotechar='|')
        dateTimeObj = datetime.now()
        date_test = str(dateTimeObj.month) + "m_" + str(dateTimeObj.day) + "d_" + str(dateTimeObj.hour) \
                    + "h_" + str(dateTimeObj.minute) + "m"

        row_data = [date_test, target_name, dataset_name, model_name, data_size,
                    str(rmse_arr.mean()), str(mape_arr.mean()), str(mae_arr.mean()), str(nll_arr.mean())]
        # Write it to the CSV file
        csvwriter.writerow(row_data)
