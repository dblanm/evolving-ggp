# Training using e-GGP
## Graph-Interaction environment
The first time you want to train with a new dataset you must use the flag `--new_dataset`.
This will create `.pkl` files to avoid creating new data sets every time the code is run.

```bash
python3 train_eggp.py  -df datasets/graph_interaction_origin.npy
-dt datasets/graph_interaction_test/ --new_dataset
```
Required arguments:
- `--data_limit 15`: set the limit of training points.
- `--targets 3 5`: train and test for the targets Vx and Vy.
- `--priors`: flag to add prior information on the connectivity of the rope and the sphere.


## Isolated evolving sub-graphs environment
Use the following command:
```bash
python3 train_eggp.py  -df datasets/isolated_subgraphs_44_particle.npy
-dt datasets/isolated_subgraphs_test/ --new_dataset
```
Required arguments:
- `--data_limit 15`: set the limit of training points.
- `--targets 2 3`: train and test for the targets Vx and Vy.
- `--conn_r`: Connectivity radius used for the isolated subgraphs dataset
- `--k_nn`: k-neighbours to look for.

**Kernel parameters:**
- `--root_kernel`: root kernel to use (RBF/Matern52/Matern32)
- `--leaf_kernel`: leaf kernel to use (RBF/Matern52/Matern32)

**Data parameters:**
- `--fixed_adj`: whether the adjacency matrix is fixed for all the graph sequences or not.
- `--new_dataset`: flag to use when using a data set for the first time.
- `--shuffle`: whether to shuffle the training data or not
- `--output_folder`: where to save the Tensorboard, figures and results CSV file.
- `--load_eggp_model`: load a single model for a target e.g. Vx or Vy.

**Plot parameters**
- `--graph_plot`: plot the graph connectivity for the data set.
- `--plot`: plot the predictive mean and variance of the test data.
- `--vel_plot`: plot the targets and derivatives of the targets.

