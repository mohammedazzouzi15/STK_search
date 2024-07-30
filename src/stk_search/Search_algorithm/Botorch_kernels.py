from botorch.models import SingleTaskGP
from gpytorch import kernels
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from stk_search.Search_algorithm.tanimoto_kernel import TanimotoKernel


# We define our custom GP surrogate model using the Tanimoto kernel
class TanimotoGP(SingleTaskGP):
    """This class is to define the surrogate model using the Tanimoto kernel.
    Here the Surrogate model is defined using the Tanimoto kernel and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y)
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MaternKernel(SingleTaskGP):
    """This class is to define the surrogate model using the Matern kernel.
    Here the Surrogate model is defined using the Matern kernel from GPYtorch and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y)
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            base_kernel=kernels.MaternKernel(ard_num_dims=train_X.shape[-1])
        )
        self.to(train_X)

    def change_kernel(self, kernel):
        self.covar_module = ScaleKernel(base_kernel=kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFKernel(SingleTaskGP):
    """This class is to define the surrogate model using the RBF kernel.
    
    Here the Surrogate model is defined using the RBF kernel from GPYtorch and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y)
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            base_kernel=kernels.RBFKernel(ard_num_dims=train_X.shape[-1])
        )
        self.to(train_X)

    def change_kernel(self, kernel):
        self.covar_module = ScaleKernel(base_kernel=kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
