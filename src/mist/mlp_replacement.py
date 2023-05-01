import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, to_networkx


# TODO not useable
class GCN(torch.nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super(GCN, self).__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout

        self.mlp = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.ReLU(),
        )
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(spectra_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(spectra_dropout),
        )

    def forward(self, form_vec):
        B, N, D = form_vec.shape

        # Reduce dimensions of the last 5 features to 1 using MLP - this is the head node
        nodes = self.mlp(form_vec[:, :, 16:]).squeeze(-1)

        # Concatenate the first 16 nodes and the 17th head node
        x = torch.cat([form_vec[:, :, :16], nodes.unsqueeze(-1)], dim=-1)

        # Define the edge_index outside the loop
        edge_index = torch.tensor(
            [[i for i in range(17)], [17] * 17], dtype=torch.long
        ).to(form_vec.device)

        # Create a list of Data objects
        data_list = []
        for i in range(B):
            data = Data(x=x[i], edge_index=edge_index)
            data_list.append(data)

        # Create a batch using Batch.from_data_list
        batch = Batch.from_data_list(data_list)

        # Perform 2 layers of message passing
        x = self.conv1(data.x, batch.edge_index)
        x = F.relu(x)
        x = self.conv2(x, batch.edge_index)
        x = F.relu(x)

        # Apply the formula encoder MLP
        x = self.lin(x)

        return x


class MeanPermInvariantMLP(nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super().__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.formula_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )

    def forward(self, x):
        x_mean = x.mean(
            dim=-1, keepdim=True
        )  # Average along the last dimension, shape: [batch_size, num_points, 1] or [batch_size, num_points, num_points, 1]
        x = x_mean.repeat(
            1, *([1] * (len(x.shape) - 2)), self.formula_dim
        )  # Repeat the averaged representation for each feature, shape: [batch_size, num_points, formula_dim] or [batch_size, num_points, num_points, formula_dim]

        x = self.dense_encoder(
            x
        )  # Apply dense_encoder to each point, shape: [batch_size, num_points, hidden_size] or [batch_size, num_points, num_points, hidden_size]

        return x


class SumPermInvariantMLP(nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super().__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.formula_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )

    def forward(self, x):
        x_sum = x.sum(
            dim=-1, keepdim=True
        )  # Average along the last dimension, shape: [batch_size, num_points, 1] or [batch_size, num_points, num_points, 1]
        x = x_sum.repeat(
            1, *([1] * (len(x.shape) - 2)), self.formula_dim
        )  # Repeat the averaged representation for each feature, shape: [batch_size, num_points, formula_dim] or [batch_size, num_points, num_points, formula_dim]

        x = self.dense_encoder(
            x
        )  # Apply dense_encoder to each point, shape: [batch_size, num_points, hidden_size] or [batch_size, num_points, num_points, hidden_size]

        return x


class MeanAggPermInvariantMLP(nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super().__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.formula_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )
        self.form_encoder = nn.Sequential(
            nn.Linear(16, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.formula_dim - 16, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )

    def forward(self, x):
        # formula embedding
        form_part = x[:, :, :16]  # torch.Size([128, 60, 16])
        form_part = self.form_encoder(form_part)  # torch.Size([128, 60, 256])
        # precursor_mass & peak type
        global_part = x[:, :, 16:]  # torch.Size([128, 60, 5])
        global_part = self.global_encoder(global_part)  # torch.Size([128, 60, 256])
        x = form_part + global_part
        x = self.dense_encoder(x)
        x_mean = x.sum(
            dim=-1, keepdim=True
        )  # Average along the last dimension, shape: [batch_size, num_points, 1] or [batch_size, num_points, num_points, 1]
        x = x_mean.repeat(
            1, *([1] * (len(x.shape) - 2)), self.formula_dim
        )  # Repeat the averaged representation for each feature, shape: [batch_size, num_points, formula_dim] or [batch_size, num_points, num_points, formula_dim]

        x = self.dense_encoder(
            x
        )  # Apply dense_encoder to each point, shape: [batch_size, num_points, hidden_size] or [batch_size, num_points, num_points, hidden_size]

        return x


class SumAggPermInvariantMLP(nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super().__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.formula_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )

    def forward(self, x):
        x_mean = x.mean(
            dim=-1, keepdim=True
        )  # Average along the last dimension, shape: [batch_size, num_points, 1] or [batch_size, num_points, num_points, 1]
        x = x_mean.repeat(
            1, *([1] * (len(x.shape) - 2)), self.formula_dim
        )  # Repeat the averaged representation for each feature, shape: [batch_size, num_points, formula_dim] or [batch_size, num_points, num_points, formula_dim]

        x = self.dense_encoder(
            x
        )  # Apply dense_encoder to each point, shape: [batch_size, num_points, hidden_size] or [batch_size, num_points, num_points, hidden_size]

        return x


# Used in original MIST paper
class MLP(nn.Module):
    def __init__(
        self,
        formula_dim: int = 21,
        hidden_size: int = 256,
        spectra_dropout: float = 0.5,
    ):
        super(MLP, self).__init__()
        self.formula_dim = formula_dim
        self.hidden_size = hidden_size
        self.spectra_dropout = spectra_dropout
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.formula_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.spectra_dropout),
        )

    def forward(self, x):
        x = self.dense_encoder(x)
        return x


mlp_replace_register = {
    "mlp": MLP,
    "mean-perm-inv-mlp": MeanPermInvariantMLP,
    "sum-perm-inv-mlp": SumPermInvariantMLP,
    "meanagg-perm-inv-mlp": MeanAggPermInvariantMLP,
    "sumagg-perm-inv-mlp": SumAggPermInvariantMLP,
    "gcn": GCN,
}


if __name__ == "__main__":
    pass
