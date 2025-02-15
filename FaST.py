import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x):
        # x:b,n,d
        x, g = torch.chunk(self.linear(x), chunks=2, dim=-1)
        return x * F.sigmoid(g)


class ParallelMoEWithGLU(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        router_fea_dim,
        num_experts,
        day_of_step,
        week_of_step,
        num_nodes,
    ):
        super(ParallelMoEWithGLU, self).__init__()
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.day_of_step = day_of_step
        self.week_of_step = week_of_step

        self.router_logit_layer = nn.Linear(router_fea_dim, self.num_experts)

        self.adaptive_router_day = nn.Parameter(
            torch.empty(day_of_step, self.num_experts)
        )
        nn.init.xavier_uniform_(self.adaptive_router_day)

        self.adaptive_router_week = nn.Parameter(
            torch.empty(week_of_step, self.num_experts)
        )
        nn.init.xavier_uniform_(self.adaptive_router_week)

        self.adaptive_router_node = nn.Parameter(
            torch.empty(num_nodes, self.num_experts)
        )
        nn.init.xavier_uniform_(self.adaptive_router_node)

        self.experts = nn.Linear(in_dim, self.num_experts * self.out_dim * 2)

    def forward(self, x, history_data):
        """x:b,n,d"""
        b, n, d = x.shape

        # router logit
        router = self.router_logit_layer(
            history_data[:, :, :, 0].transpose(2, 1).contiguous()
        )
        # +adaptive_router_day
        router += self.adaptive_router_day[
            (history_data[:, -1, :, 1] * self.day_of_step).type(torch.LongTensor)
        ].contiguous()
        # +adaptive_router_week
        router += self.adaptive_router_week[
            (history_data[:, -1, :, 2] * self.week_of_step).type(torch.LongTensor)
        ].contiguous()
        # +adaptive_router_node
        router += self.adaptive_router_node
        # Probabilistic
        router = F.softmax(router, dim=-1)
        # Parallel MoE With GLU
        x = self.experts(x).view(-1, n, self.num_experts, 2 * self.out_dim)
        x, gate = torch.chunk(x, chunks=2, dim=-1)
        # GLU
        x = x * F.sigmoid(gate)
        x = torch.einsum("bne,bned->bnd", router, x)

        return x


class AAGA(nn.Module):
    """Adaptive Graph Agent Attention (AGAA)"""

    def __init__(self, dim):
        super(AAGA, self).__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.agent = nn.Linear(dim, dim * 2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, agent, x):
        # agent: (k, d)
        # x: (b, n, d)

        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        q_agent, k_agent = torch.chunk(self.agent(agent), chunks=2, dim=-1)

        # Graph-to-Agent Attention
        attn = torch.einsum("kd,bnd->bkn", (q_agent, k))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc1(v)

        # Agent-to-Graph Attention
        attn = torch.einsum("bnd,kd->bnk", (q, k_agent))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc2(v)

        return v + x  # Residual


class mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        # x:b,n,d
        return self.layers(x)


class FaST(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_len=96,
        output_len=48,
        layers=3,
        num_experts=8,
        day_of_step=96,
        week_of_step=7,
        hidden_dim=64,
        num_agent=32,
    ):
        super(FaST, self).__init__()
        self.L = input_len
        self.layers = layers
        self.day_of_step = day_of_step

        self.input_layer = ParallelMoEWithGLU(
            self.L,
            hidden_dim,
            self.L,
            num_experts,
            day_of_step,
            week_of_step,
            num_nodes,
        )
        self.input_norm = nn.LayerNorm(normalized_shape=[hidden_dim])

        self.AAGA = nn.ModuleList()
        self.MoE = nn.ModuleList()
        for _ in range(layers):
            self.AAGA.append(AAGA(hidden_dim))
            self.MoE.append(
                ParallelMoEWithGLU(
                    hidden_dim,
                    hidden_dim,
                    self.L,
                    num_experts,
                    day_of_step,
                    week_of_step,
                    num_nodes,
                )
            )

        self.output_layer = mlp(hidden_dim * (layers + 1), output_len)

        # adaptive agent
        self.agent = nn.Parameter(torch.empty(num_agent, hidden_dim))
        nn.init.xavier_uniform_(self.agent)
        # time of day
        self.tape_day = nn.Parameter(torch.empty(day_of_step, hidden_dim))
        nn.init.xavier_uniform_(self.tape_day)
        # day of week
        self.tape_week = nn.Parameter(torch.empty(week_of_step, hidden_dim))
        nn.init.xavier_uniform_(self.tape_week)
        # node embedding
        self.node_emb = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (torch.Tensor): shape (b, l, n, 3)
            - 0: data
            - 1: time of day
            - 2: day of week

        Returns:
            torch.Tensor: (b, p, n, 1)
        """
        B, L, N, C = history_data.shape

        x = history_data[:, :, :, 0].transpose(2, 1).contiguous()

        x = self.input_layer(x, history_data)
        # + day embedding
        x += self.tape_day[
            (history_data[:, -1, :, 1] * self.day_of_step).type(torch.LongTensor)
        ].contiguous()
        # + week embedding
        x += self.tape_week[
            (history_data[:, -1, :, 2] * 7).type(torch.LongTensor)
        ].contiguous()
        # + node embedding
        x += self.node_emb
        x = self.input_norm(x)

        skip = [x]
        for i in range(self.layers):
            x = self.AAGA[i](self.agent, x)
            res = x
            x = self.MoE[i](x, history_data)
            skip.append(x)
            x = x + res

        x = torch.cat(skip, dim=-1)
        x = self.output_layer(x).unsqueeze(-1).transpose(2, 1).contiguous()

        return x  # prediction:[b, p, n, 1]


if __name__ == "__main__":
    model = FaST(716, 96, 48)
    summary(model, [64, 96, 716, 3])
