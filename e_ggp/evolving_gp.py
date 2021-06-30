# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#


from gpytorch import means
from gpytorch.models import ExactGP

from gpytorch.distributions import MultivariateNormal


class eGGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(eGGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernel
        # self.k_b = k_b
        # self.k_c = k_c

    def forward(self, x, adj_list_1, adj_list_2=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, adj_list_1=adj_list_1, adj_list_2=adj_list_2)

        return MultivariateNormal(mean_x, covar_x)
