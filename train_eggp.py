# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

from e_ggp.utils.datasets import GraphInteraction, IsolatedSubgraphs
from e_ggp.evolving_gp import eGGP
from e_ggp.utils.kernel_utils import get_phi_mappings, get_graph_kernel
from e_ggp.utils.data_utils import extend_train_dataset, get_data_targets, save_onestep_data, \
    create_csv_file, get_xy_from_data, save_torch_model, get_target_name, get_model_and_writer_name, \
    create_training_data, create_test_data

from e_ggp.utils.plot_utils import plot_velocities, plot_2d_dataset

import argparse
import numpy as np
import gpytorch


from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

import torch

float_type = torch.float64


np.random.seed(123)

main_folder = "runs/"


def load_cloth_data(file):
    data = np.load(file).astype(np.float64)

    return data


def main(args):
    # Modify the main folder
    global main_folder
    main_folder = args.output_folder

    # Create the CSV file for storing the results
    create_csv_file(main_folder)

    # Training file argument
    train_dataset, train_dataset_name = create_training_data(args)
    test_dataset, test_dataset_names = create_test_data(args)

    # Test generalisation
    train_test_eGGP(args, train_dataset, test_dataset, train_dataset_name, test_dataset_names)


def train_test_eGGP(args, graph_dataset, test_dataset, train_dataset_name, test_dataset_names):

    # Dimension of the attributed node
    attr_dim = graph_dataset.node_dim

    # Get X, Y and the adj list
    step_size = 1
    dataset_size = np.linspace(0, graph_dataset.timesteps - step_size - 1,
                               graph_dataset.timesteps - step_size - 1, dtype=int)

    if args.graph_plot and isinstance(graph_dataset, GraphInteraction):
        print("Plotting 2d GraphInteraction")
        graph_idxs = [0]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder, legend=False, save_add="_origin")

        graph_idxs = [180, 200, 300]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

        graph_idxs = [400, 405, 410, 415, 420]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)
    if args.graph_plot and isinstance(graph_dataset, IsolatedSubgraphs):
        print("Plotting Isolated sub-graphs")
        graph_idxs = [10]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

        graph_idxs = [38]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

        graph_idxs = [70]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

        graph_idxs = [100]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

        graph_idxs = [150]
        plot_2d_dataset(graph_dataset, graph_idxs, args.output_folder)

    if args.shuffle:
        print("Shuffle data enabled")
        np.random.shuffle(dataset_size)

    # For the isolated subgraphs we learn target_idxs = [2, 3]  # V_x and V_y
    # For the graph interactions we learn target_idxs = [3, 5]  # V_x and V_z
    target_idxs = args.targets

    for id_target in target_idxs:
        # Create the training and test idxs
        if args.full_data:
            print("Full data enabled")
            dataset_train_idx = dataset_size.copy()
        else:
            print("Full data disabled")
            dataset_train_idx = get_data_targets(graph_dataset, args.data_limit, id_target,
                                                 args.vel_plot, args.output_folder)

            if len(dataset_train_idx) < 18 and args.graph_plot:  # Maximum colors that we can plot
                plot_2d_dataset(graph_dataset, dataset_train_idx.tolist(), args.output_folder, save_add="_train_")

        dataset_test_idx = dataset_size.copy()

        # We can plot the velocities once the training dataset is defined
        if args.vel_plot:
            print("Plotting target")
            plot_velocities(graph_dataset, dataset_train_idx, id_target, args.output_folder)

        obs, target, adj_train = extend_train_dataset(graph_dataset, step_size, dataset_train_idx, target_idx=id_target)

        # Get the phi mapping
        phi_node = get_phi_mappings(args)
        # Train the model
        ggp_model, model_name, ggp_writer = train_eGGP(attr_dim, obs.copy(), target.copy(), adj_train.copy(),
                                                       graph_dataset, id_target, args, phi_node=phi_node)
        perform_one_step_predictions(model=ggp_model, model_name=model_name, writer=ggp_writer,
                                     train_dataset=graph_dataset, train_dataset_idx=dataset_test_idx,
                                     train_dataset_name=train_dataset_name, test_dataset=test_dataset,
                                     test_dataset_names=test_dataset_names, step_size=step_size,
                                     target_idx=id_target, adj_train=adj_train, plot=args.plot,
                                     plot_variance=args.plot_variance)


def predict_test_data(model, dataset, dataset_test_idx, step_size, target_idx, model_name, writer,
                      dataset_name, adj_train=None, plot=True, plot_variance=True, only_rope=False):
    # Create the lists for storing the results
    lists_dict = {'rmse': [], 'mae': [], 'nll': [], 'mape': [],
                  'gt': [], 'gp_pred': [], 'gp_var': []}
    # Define what is the target name
    target_name = get_target_name(dataset, target_idx)

    name = dataset_name + "test data " + target_name

    dataset_test_idx.sort()

    if only_rope:
        name = name + "_only_rope"
        dataset_name = dataset_name + "_only_rope"

    # Predict for the test nodes In a foor loop
    for idx in dataset_test_idx:

        nodes_test, adj_test, gt_target = get_xy_from_data(idx, step_size, dataset, target_idx)

        test = torch.from_numpy(nodes_test)

        # Predict mean and variance
        f_pred = model(test, adj_list_1=adj_test, adj_list_2=adj_train)
        mean_gp = f_pred.mean.detach().numpy()
        var_gp = f_pred.variance.detach().numpy()

        lists_dict = compute_onestep_metrics(gt_target=gt_target, mean_gp=mean_gp, var_gp=var_gp,
                                              lists_dict=lists_dict, writer=writer,
                                              name=name, idx=idx, only_rope=only_rope)

    global main_folder
    save_onestep_data(main_folder, lists_dict,
                      model_name, dataset_name, target_name, dataset_size=args.data_limit,
                      plot=plot, plot_variance=plot_variance)


def compute_onestep_metrics(gt_target, mean_gp, var_gp,
                            lists_dict, writer, name, idx,
                            multioutput="raw_values", only_rope=False):
    if only_rope:  # In case of the graph-interactiosn we only predict the rope
        gt_target = gt_target[:15, :]
        mean_gp = mean_gp[:15]
        var_gp = var_gp[:15]

    # Compute the RMSE
    rmse_gp = mean_squared_error(gt_target, mean_gp, multioutput=multioutput, squared=False)
    # Compute the MAPE
    mape_gp = mean_absolute_percentage_error(gt_target, mean_gp, multioutput=multioutput)
    # Compute the MAE
    mae_gp = mean_absolute_error(gt_target, mean_gp, multioutput=multioutput)

    # Sum over the nodes
    sum_result = np.sum(0.5 * np.log(2 * np.pi * var_gp) + (gt_target - mean_gp)**2 / (2 * var_gp), axis=1)
    nll = np.mean(sum_result, axis=0)  # Mean over the timesteps

    writer.add_scalar("RMSE " + name, rmse_gp, idx)
    writer.add_scalar("MAE " + name, mae_gp, idx)
    writer.add_scalar("MAPE " + name, mape_gp, idx)
    writer.add_scalar("NLL " + name, nll, idx)

    lists_dict['rmse'].append(rmse_gp)
    lists_dict['mae'].append(mae_gp)
    lists_dict['nll'].append(nll)
    lists_dict['mape'].append(mape_gp)
    lists_dict['gt'].append(gt_target)
    lists_dict['gp_pred'].append(mean_gp)
    lists_dict['gp_var'].append(var_gp)

    return lists_dict


def perform_one_step_predictions(model, model_name, writer, train_dataset, train_dataset_idx, train_dataset_name,
                                 test_dataset, test_dataset_names, step_size, target_idx,
                                 adj_train, plot, plot_variance):
    if isinstance(train_dataset, IsolatedSubgraphs):
        only_rope = False
    else:
        only_rope = True

    # Test the model
    predict_test_data(model, train_dataset, train_dataset_idx,
                      step_size, target_idx, model_name, writer, train_dataset_name,
                      adj_train=adj_train,
                      plot=args.plot, plot_variance=args.plot_variance, only_rope=only_rope)

    # Test the generalisation
    test_dataset_idx = np.linspace(0, test_dataset[0].timesteps - step_size - 1,
                                   test_dataset[0].timesteps - step_size - 1, dtype=int)
    for i in range(len(test_dataset)):
        predict_test_data(model, test_dataset[i], test_dataset_idx,
                          step_size, target_idx, model_name, writer, test_dataset_names[i],
                          adj_train=adj_train,
                          plot=plot, plot_variance=plot_variance, only_rope=only_rope)


def train_model(model, likelihood, X, Y, writer, adj_train=None):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 150  # 300

    for i in range(training_iter):

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X, adj_list_1=adj_train)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        writer.add_scalar("Training MLL", loss.item(), i + 1)
        optimizer.step()


def train_eGGP(input_dim, X, Y, adj_train, train_dataset, target_idx, args, phi_node=None):

    # Create a tensorboard
    model_type = "e-GGP"
    global main_folder
    model_name, writer_name = get_model_and_writer_name(args, main_folder, model_type,
                                                        args.root_kernel, args.leaf_kernel)
    writer = SummaryWriter(writer_name)

    # Get the kernel
    kernel = get_graph_kernel(input_dim, phi_node, args.root_kernel, args.leaf_kernel)
    # Transform the input to tensor form
    X = torch.from_numpy(X).double()
    Y = torch.from_numpy(Y).double().squeeze()

    # Build the e-GGP
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    e_ggp = eGGP(train_x=X, train_y=Y, likelihood=likelihood, kernel=kernel).double()

    # Train the model
    if args.load_eggp_model == "":
        e_ggp.train()
        likelihood.train()
        train_model(e_ggp, likelihood, X, Y, writer, adj_train=adj_train)
        # Save the model
        target_name = get_target_name(train_dataset, target_idx)
        save_torch_model(model=e_ggp, folder=writer_name, extra_name=target_name)
    else:
        state_dict = torch.load(args.load_eggp_model)
        e_ggp.load_state_dict(state_dict)

    # Move to evaluation mode
    e_ggp.eval()
    likelihood.eval()

    return e_ggp, model_name, writer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Output argument
    parser.add_argument("-o", "--output_folder", help="Folder where to save the models and results.",
                        default="runs/")

    # Data Arguments
    parser.add_argument("-df", "--data_file", help="Data file to load")
    parser.add_argument("-dt", "--data_test_folder", help="Folder with the data for testing")
    parser.add_argument("-n", "--new_dataset", default=False, action="store_true",
                        help="Whether we are using a new dataset or not")
    parser.add_argument("--shuffle", default=False, action="store_true",
                        help="Whether to shuffle the training data or not")
    parser.add_argument("--full_data", default=False, action="store_true",
                        help="Whether to use the full data-set for training or not")
    parser.add_argument("--data_limit", type=int, default=10,
                        help="If not using the full data-set, size limit of the training data")
    parser.add_argument("--no_pkl", action="store_true",
                        help="Flag to avoid saving the data set into a pkl")

    # Graph arguments
    parser.add_argument("--priors", default=False, action="store_true",
                        help="Whether to use priors on the dataset or not")
    parser.add_argument("--fixed_adj", default=False, action="store_true",
                        help="Whether to keep the adjacency fixed or not")
    parser.add_argument("--only_rope", default=False, action="store_true",
                        help="Whether to test only the rope prediction or not")
    parser.add_argument("--conn_r", type=float, default=0.015,
                        help="Connectivity radius used for the isolated subgraphs dataset")
    parser.add_argument("--k_nn", type=int, default=10,
                        help="K-neigbhours to look at.")

    # Load saved model argument
    parser.add_argument("--load_eggp_model", default="", help="Full path of e-GGP model to load")

    # e-GGP Arguments
    parser.add_argument("--root_kernel", default="RBF", choices=["RBF", "Matern32", "Matern52"])
    parser.add_argument("--leaf_kernel", default="RBF", choices=["RBF", "Matern32", "Matern52"])
    parser.add_argument("--phi_node", default="eye", choices=["eye", "pos", "vel"])

    # Plotting arguments
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--plot_variance", default=False, action="store_true")
    parser.add_argument("--graph_plot", default=False, action="store_true")
    parser.add_argument("--vel_plot", default=False, action="store_true")

    # Test arguments
    parser.add_argument("--targets", type=int, nargs="+",
                        help="Isolated subgraph dataset Vx-2, Vy-3; Graph interaction dataset Vx-3, Vz-5")

    args = parser.parse_args()

    main(args)

