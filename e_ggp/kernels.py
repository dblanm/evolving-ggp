# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import torch
import numpy as np

from gpytorch.kernels import Kernel
from gpytorch.functions import add_jitter


class SumANK(Kernel):
    """ Attributed Sub-tree kernel class.
    """

    def __init__(self, k_b, k_c):
        super(SumANK, self).__init__(active_dims=None)
        self.k_b = k_b
        self.k_c = k_c

    @property
    def kernel_params(self):
        k_b_len = self.k_b.base_kernel.lengthscale.detach().numpy()
        k_b_var = self.k_b.outputscale.detach().numpy()
        k_c_len = self.k_c.base_kernel.lengthscale.detach().numpy()
        k_c_var = self.k_c.outputscale.detach().numpy()

        params_dict = {'k_b_len': k_b_len.tolist(), 'k_b_var': k_b_var.tolist(),
                       'k_c_len': k_c_len.tolist(), 'k_c_var': k_c_var.tolist()}
        return params_dict

    @staticmethod
    def __extend_adj_list(adj_1, adj_2, n1):
        """ Extend the values in and adjacency list given a new adjacency list
        :param adj_1: adjacency list to extend
        :param adj_2: adjacency list to concatenate
        :param n1: length of adj_1
        :return: extended adjacency list
        """
        adj_new_values = [[x + n1 for x in x_adj] for x_adj in adj_2]
        a1_list = adj_1.tolist()
        a1_list.extend(adj_new_values)

        adj_extended = np.array(a1_list, dtype=object)

        return adj_extended

    def __check_adjacency_dims(self, x1, x2, adj_list_1, adj_list_2):
        if adj_list_2 is None:
            adj_list_2 = adj_list_1
        else:
            n1 = len(adj_list_1)
            n2 = len(adj_list_2)
            if x1.shape == x2.shape:  # Same input
                if x1.shape[0] == n1:  # Use the correct adjacency list
                    adj_list_2 = adj_list_1
                else:
                    adj_list_1 = adj_list_2
            else:
                if x1.shape[0] == n1 and x2.shape[0] == n2:
                    pass  # All inputs are correct
                elif x2.shape[0] == n1 and x1.shape[0] == n2:
                    # Correct the input
                    adj_temp_1 = adj_list_2
                    adj_temp_2 = adj_list_1
                    adj_list_1 = adj_temp_1
                    adj_list_2 = adj_temp_2
                else:  # One of the inputs is the sum of the others
                    if x2.shape[0] == (n1 + n2):
                        # We need to extend the adjacency list
                        adj_list_2 = self.__extend_adj_list(adj_list_2, adj_list_1, n2)
                    elif x1.shape[0] == (n1 + n2):
                        adj_list_1 = self.__extend_adj_list(adj_list_1, adj_list_2, n1)
                    else:
                        raise ValueError("This case should not happen")
        return adj_list_1, adj_list_2

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, adj_list_1=None,
                adj_list_2=None, **params):
        # When computing test-train data (posterior evaluation)
        # the adj_list_1 must refer to the test data adjacency list
        # and adj_list_2 to the training adjacency list
        # This is performed on gpytorch/exact_prediction_strategies.py line 309 - test_covar.evaluate()
        adj_list_1, adj_list_2 = self.__check_adjacency_dims(x1, x2, adj_list_1, adj_list_2)

        with torch.enable_grad():
            K_b = self.k_b.forward(x1, x2)
            K_c = self.k_c.forward(x1, x2)

            K_nn = torch.zeros((x1.shape[0], x2.shape[0]))# , requires_grad=True)
            if x1.shape[0] == x2.shape[0]:
                for i in range(0, len(adj_list_1)):
                    for j in range(i, len(adj_list_2)):
                        N = len(adj_list_1[i]) * len(adj_list_2[j])
                        if N == 0:
                            K_nn[i, j] = 0
                            K_nn[j, i] = 0
                        else:
                            K_nn[i, j] = torch.sum(K_c[adj_list_1[i], :][:, adj_list_2[j]])
                            K_nn[i, j] /= N
                            K_nn[j, i] = K_nn[i, j]
                covar_x = K_b + K_nn
                covar_res = add_jitter(covar_x)
            else:
                for i in range(0, len(adj_list_1)):
                    for j in range(0, len(adj_list_2)):
                        N = len(adj_list_1[i]) * len(adj_list_2[j])
                        if N == 0:
                            K_nn[i, j] = 0
                        else:
                            K_nn[i, j] = torch.sum(K_c[adj_list_1[i], :][:, adj_list_2[j]])
                            K_nn[i, j] /= N
                covar_res = K_b + K_nn

            if diag:
                covar_res = torch.diag(covar_res)
        return covar_res
