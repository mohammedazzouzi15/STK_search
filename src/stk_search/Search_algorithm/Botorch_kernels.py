"""Definition of the different kernels used in stk_search.\

for the moment the kernels are defined as subclasses of SingleTaskGP from botorch.models.\
The kernels are defined using the gpytorch library.\
The kernels are defined as follows:
    - TanimotoKernel: The Tanimoto kernel is a custom kernel defined in the file tanimoto_kernel.py.\
    - MaternKernel: The Matern kernel is a kernel from the gpytorch library.\
    - RBFKernel: The RBF kernel is a kernel from the gpytorch library.
"""

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch import kernels
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean

from stk_search.Search_algorithm.tanimoto_kernel import TanimotoKernel


# We define our custom GP surrogate model using the Tanimoto kernel
class TanimotoGP(SingleTaskGP):
    """Define the surrogate model using the Tanimoto kernel.

    Here the Surrogate model is defined using the Tanimoto kernel and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_x, train_y):
        """Initialize the TanimotoGP class."""
        super().__init__(
            train_x, train_y, input_transform=Normalize(train_x.shape[-1])
        )
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(base_kernel=TanimotoKernel())

        self.to(train_x)

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MaternKernel(SingleTaskGP):
    """Define the surrogate model using the Matern kernel.

    Here the Surrogate model is defined using the Matern kernel from GPYtorch and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_x, train_y) -> None:
        """Initialize the MaternKernel class."""
        super().__init__(
            train_x, train_y, input_transform=Normalize(train_x.shape[-1])
        )
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            base_kernel=kernels.MaternKernel(ard_num_dims=train_x.shape[-1])
        )

        self.to(train_x)

    def change_kernel(self, kernel):
        """Change the kernel of the model.

        changes the covar_modul of the singleTaskGP model to the kernel passed as argument.

        Args:
        ----
            kernel (gpytorch.kernels): the kernel to be used in the model.

        """
        self.covar_module = ScaleKernel(base_kernel=kernel)

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFKernel(SingleTaskGP):
    """Define the surrogate model using the RBF kernel.

    Here the Surrogate model is defined using the RBF kernel from GPYtorch and is a subclass of SingleTaskGP.

    Args:
    ----
        train_X (torch.tensor): training input
        train_Y (torch.tensor): training output

    """

    def __init__(self, train_x, train_y):
        """Initialize the RBFKernel class.

        here the RBFKernel is initialized as a subclass of SingleTaskGP.
        and uses the ard_num_dims parameter to define the number of dimensions of the input.
        """
        super().__init__(
            train_x,
            train_y,
            input_transform=Normalize(d=train_x.shape[1]),
            outcome_transform=Standardize(m=1),
        )
        self.mean_module = ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            base_kernel=kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )
        self.to(train_x)

    def change_kernel(self, kernel):
        """Change the kernel of the model.

        changes the covar_modul of the singleTaskGP model to the kernel passed as argument.

        Args:
        ----
            kernel (gpytorch.kernels): the kernel to be used in the model.

        """
        self.covar_module = ScaleKernel(base_kernel=kernel)

    def forward(self, x):
        """Forward pass of the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
