import torch
import torch.nn as nn
from torch import Tensor
from typing import Any


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, bias: bool=True) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

class Priornet(nn.Module):
    """
        Prior network for epinet. This network contains an ensemble of randomly initialized models which are held fixed during training.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int,
                 z_dim: int, bias: bool=True, generator: torch.Generator=None) -> None:
        super().__init__()
        self.models = []
        self.generator = generator
        for _ in range(z_dim):
            model = MLP(input_dim, hidden_dims, output_dim, bias=bias)
            model.apply(self.init_xavier_uniform)
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

        # model specification for vmap operations:
        self.base_model: nn.Module = MLP(input_dim, hidden_dims, output_dim, bias=bias)
        self.base_model.apply(self.init_xavier_uniform)
        self.params: dict[str, Any]
        self.buffers: dict[str, Any]
        self.generate_params_buffers()

    def generate_params_buffers(self) -> None:
        """
            Generate parameters and buffers for priornet parallelization
        """
        self.params, self.buffers = torch.func.stack_module_state(self.models)

    def init_xavier_uniform(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, generator=self.generator)
            m.bias.data.fill_(0.01)

    def call_single_model(self, params: dict[str, Any], buffers: dict[str, Any], data: Tensor) -> Tensor:
        return torch.func.functional_call(self.base_model, (params, buffers), (data,))

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
            Perform forward pass on the priornet ensemble and weigh by epistemic index z
            x and z are assumed to already be formatted.
            Input:
                x: tensor consistsing of concatenated input and epistemic index
                z: tensor consisting of epistemic index
            Output:
                ensemble output of x weighed by epistemic index vector z
        """
        outputs = torch.vmap(self.call_single_model, (0, 0, None))(self.params, self.buffers, x)
        # outputs of shape (ensemble_size x batch_size x output_dim)
        return torch.einsum('ijk,ji,jk', outputs, z)

class Epinet(nn.Module):

    def __init__(self, index_dim: int, input_dim: int, output_dim: int, num_indices: int,
                 epi_hiddens: list[int], prior_hiddens: list[int], prior_scale: float,
                 bias: bool =True, generator: torch.Generator=None) -> None:
        super().__init__()
        
        self.index_dim = index_dim
        self.num_indices = num_indices
        self.epinet_input_dim = input_dim + self.index_dim

        # Trainable Epinet
        self.epinet: nn.Module = MLP(self.epinet_input_dim, epi_hiddens, self.index_dim * output_dim, bias=bias)
        self.epinet.apply(self.init_xavier_uniform)

        # Priornet
        self.priornet: nn.Module = Priornet(input_dim, prior_hiddens, output_dim, self.index_dim, bias, generator)

    def init_xavier_uniform(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, generator=self.generator)
            m.bias.data.fill_(0.01)

    def format_xz(self, x: Tensor, z: Tensor):
        """
        Take cartesian product of x and z and concatenate for forward pass.
        Input:
            x: Feature vectors containing item and user embeddings and interactions
            z: Epinet epistemic indices
        Output:
            xz: Concatenated cartesian product of x and z
        """
        batch_size, d = x.shape
        num_indices, _ = z.shape
        x_expanded = x.unsqueeze(1).expand(batch_size, num_indices, d)
        z_expanded = z.unsqueeze(0).expand(batch_size, num_indices, self.index_dim)
        xz = torch.cat([x_expanded, z_expanded], dim=-1)
        return xz.view(batch_size * num_indices, d + self.index_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Input:
            x: Feature vector containing item and user embeddings and interactions
            z: Matrix containing . Epinet epistemic indices
        Output:
            posterior samples corresponding to z
        """
        xz = self.format_xz(x, z)
        x_cartesian, z_cartesian = xz[:, : -self.index_dim], xz[:, -self.index_dim :]
        batch_size, _ = xz.shape
        epinet_out = self.epinet(xz.detach()).view(
            batch_size, self.output_dim, self.index_dim
        )
        epinet_out = torch.einsum("ijk,ik->ij", epinet_out, z_cartesian)
        with torch.no_grad():
            priornet_out = self.prior_scale * self.priornet(x_cartesian, z_cartesian)
        return epinet_out + priornet_out