# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import numpy as np
import torch
import gpytorch

from e_ggp.kernels import SumANK


def phi_node_remove_vel(x):
    # For the 3-D Cloth we want only to use the position, pos_dims = [0, 1, 2]
    x_pos = x[:, :3]
    return x_pos


def phi_node_remove_pos(x):
    # For the 3-D Cloth we want only to use the velocity vel_dims = [3, 4, 5]
    x_vel = x[:, 3:6]
    return x_vel


def phi_identity(x):
    return x


def get_phi_mappings(args):

    if args.phi_node == "eye":
        phi_node = phi_identity
    elif args.phi_node == "pos":
        phi_node = phi_node_remove_vel
    elif args.phi_node == "vel":
        phi_node = phi_node_remove_pos
    else:
        phi_node = phi_identity

    return phi_node


def get_kernel(kernel_name, variance, lengthscales, batch_shape=torch.Size([])):
    ard_dim = len(lengthscales)
    # Create one prior per dimension
    if "RBF" in kernel_name:
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim),
                                              batch_shape=batch_shape, ard_num_dims=None)

    elif "Matern32" in kernel_name:
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_dim, nu=3/2),
                                              batch_shape=batch_shape, ard_num_dims=None)
    elif "Matern52" in kernel_name:
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_dim, nu=5/2),
                                              batch_shape=batch_shape, ard_num_dims=None)
    else:
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim),
                                              batch_shape=batch_shape, ard_num_dims=None)

    # Get the constraint
    len_constraint = kernel.base_kernel.raw_lengthscale_constraint
    # Set the lengthscale
    kernel.base_kernel.raw_lengthscale.data.fill_(len_constraint.inverse_transform(torch.tensor(lengthscales[0])))
    # Get the variance constraint
    var_constraint = kernel.raw_outputscale_constraint
    # Set the variance
    kernel.raw_outputscale.data.fill_(var_constraint.inverse_transform(torch.tensor(variance)))

    return kernel


def get_node_edge_dims(input_dim, phi_node_fn):

    # Create a dummy dataset to get the dimensions
    size = 10
    x = np.random.rand(size, input_dim)
    nodes_size = phi_node_fn(torch.from_numpy(x)).shape[1]

    return nodes_size


def get_graph_kernel(input_dim, phi_node_fn, root_kernel, leaf_kernel):
    # Get graph node dime and edge dim based on phi functions
    graph_node_dim = get_node_edge_dims(input_dim, phi_node_fn)
    # Set the prior values and lengthscales
    ggp_var = 0.01

    ggp_lenscale = [1. for i in range(0, graph_node_dim)]

    k_b = get_kernel(root_kernel, ggp_var, ggp_lenscale)
    k_c = get_kernel(leaf_kernel, ggp_var, ggp_lenscale)

    kernel = SumANK(k_b=k_b, k_c=k_c)

    return kernel


