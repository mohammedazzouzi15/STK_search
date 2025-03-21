
import ase
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class SchNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        node_class=None,
        readout="mean",
        dipole=False,
        mean=None,
        std=None,
        atomref=None,
        gamma=None,
    ):
        super().__init__()

        assert readout in ["add", "sum", "mean"]

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = "add" if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer("atomic_mass", atomic_mass)

        self.embedding = Embedding(node_class, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians, gamma=gamma)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # TODO: double-check hidden size
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.register_buffer("initial_atomref", atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None, edge_index=None, return_latent=False):
        if z.dim() == 1:
            assert z.dim() == 1
            assert z.dtype == torch.long
            h = self.embedding(z)
        else:  # When the input z is one-hot
            assert z.dim() == 2
            h = torch.matmul(z, self.embedding.weight)
        batch = torch.zeros_like(z) if batch is None else batch

        if edge_index is None:
            #pass
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        if return_latent:
            return out, h
        return out

    def forward_with_gathered_index(self, gathered_z, pos, batch, edge_index, gathered_batch, periodic_index_mapping, return_latent=False):
        gathered_h = self.embedding(gathered_z)
        batch = torch.zeros_like(gathered_z) if batch is None else batch

        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        gathered_row = periodic_index_mapping[row]
        gathered_col = periodic_index_mapping[col]
        gathered_edge_index = torch.stack([gathered_row, gathered_col])

        for interaction in self.interactions:
            gathered_h = gathered_h + interaction(gathered_h, gathered_edge_index, edge_weight, edge_attr)

        gathered_h = self.lin1(gathered_h)
        gathered_h = self.act(gathered_h)
        gathered_h = self.lin2(gathered_h)

        out = scatter(gathered_h, gathered_batch, dim=0, reduce=self.readout)

        if return_latent:
            return out, gathered_h
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        return self.lin(x)


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * 3.14 / self.cutoff) + 1.0)

        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        # propagate_type: ( x: Tensor, W: Tensor )
        x = self.propagate(edge_index, x=x, W=W)
        return self.lin2(x)

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, gamma=None):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if gamma is None:
            self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        else:
            self.coeff = -gamma
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
