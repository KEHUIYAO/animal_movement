import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.layers import PositionalEncoding
from tsl.nn.blocks.encoders.transformer import TransformerLayer
from tsl.nn.blocks.encoders.mlp import MLP

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=64, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table






class CsdiModel(nn.Module):
    def __init__(self,
                 input_dim=1,
                 hidden_dim=64,
                 diffusion_embedding_dim=128,
                 covariate_dim=0,
                 nheads=1
                 ):


        super().__init__()

        self.hidden_dim = hidden_dim


        self.pe = PositionalEncoding(hidden_dim)
        self.cond_projection = MLP(covariate_dim, hidden_dim, n_layers=2)
        self.input_projection = MLP(input_dim * 2, hidden_dim, n_layers=2)
        self.output_projection = MLP(input_size=hidden_dim,
                                    hidden_size=hidden_dim,
                                    output_size=input_dim,
                                    n_layers=2)

        self.st_transformer_layer = TransformerLayer(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            ff_size=hidden_dim,
            n_heads=nheads,
            activation='relu',
            causal=False,
            dropout=0.)

        self.mask_token = StaticGraphEmbedding(1, hidden_dim)

        self.diffusion_embedding =  DiffusionEmbedding(
            num_steps=50,
            embedding_dim=diffusion_embedding_dim,
        )

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, hidden_dim)




    def forward(self, x, mask, noisy_data, diffusion_step, u=None, **kwargs):
        # x: [batches steps nodes features]
        side_info = u
        x = x * mask + noisy_data * (1-mask)
        x = torch.cat([x, mask], dim=-1)
        # convert x to double
        x = x.float()
        x = self.input_projection(x)

        if side_info is not None:
            side_info = self.cond_projection(side_info)
            x = x + side_info
        x = self.pe(x)


        # diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)
        x = x + diffusion_emb.unsqueeze(1).unsqueeze(2)


        x = self.st_transformer_layer(x)
        x = self.output_projection(x)

        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--covariate_dim', type=int, default=0)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--diffusion_embedding_dim', type=int, default=128)

        return parser