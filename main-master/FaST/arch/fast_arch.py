import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from easytorch.device import to_device

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
            Zhang B, Sennrich R. Root mean square layer normalization. Advances in neural information processing systems, 2019, 32.
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class TrainableParameterLayer(nn.Module):
    def __init__(self, shape):  # shape = [num_embeddings, dim]
        super(TrainableParameterLayer, self).__init__()
        self.parameter = nn.Parameter(torch.empty(*shape))
        nn.init.xavier_uniform_(self.parameter)

    def forward(self, indices):
        return self.parameter[indices]


class HARoutingLayer(nn.Module):
    def __init__(self, router_fea_dim, num_experts, daily_steps, weekly_days, num_nodes):
        super(HARoutingLayer, self).__init__()
        self.daily_steps = daily_steps
        self.weekly_days = weekly_days
        self.num_nodes = num_nodes
        
        self.router_logit_layer = nn.Linear(router_fea_dim, num_experts)
        self.adaptive_router_day  = TrainableParameterLayer([daily_steps, num_experts])
        self.adaptive_router_week = TrainableParameterLayer([weekly_days, num_experts])
        self.adaptive_router_node = TrainableParameterLayer([num_nodes, num_experts])

    def forward(self, x, day_idx, week_idx, node_idx):
        # router logit
        router = self.router_logit_layer(x)
        # +adaptive_router_day bias
        router += self.adaptive_router_day(day_idx)
        # +adaptive_router_week bias
        router += self.adaptive_router_week(week_idx)
        # +adaptive_router_node bias
        router += self.adaptive_router_node(node_idx)
        # Probabilistic
        router = F.softmax(router, dim=-1)
        return router

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim=-1):
        super(GLU, self).__init__()
        if out_dim<0: out_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim*2)

    def forward(self, x):
        # x:b,n,d
        x, g = torch.chunk(self.linear(x), chunks=2, dim=-1)
        return x * F.sigmoid(g)

class ParallelMoEWithGLU(nn.Module):
    def __init__(self,in_dim, out_dim, num_experts, num_nodes, res_flag=True):
        super(ParallelMoEWithGLU, self).__init__()
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_nodes = num_nodes
        self.res_flag = res_flag
        self.GLU_Experts = GLU(in_dim, num_experts * out_dim)
        if res_flag:
            self.norm = RMSNorm(d=out_dim)

    def forward(self, x, router):
        """x:b,n,d"""

        res = x
        # reshape: b,n,ed->b,n,e,d
        x = self.GLU_Experts(x).view(-1, self.num_nodes, self.num_experts, self.out_dim)
        x = torch.einsum("bne,bned->bnd", router, x)
        if self.res_flag:
            return self.norm(x + res),x
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
        self.norm = RMSNorm(d=dim)

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

        return self.norm(v + x)

class mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mlp, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),nn.Linear(in_dim, out_dim))

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
        daily_steps=96,
        weekly_days=7,
        hidden_dim=64,
        num_agent=32,
    ):
        super(FaST, self).__init__()
        self.L = input_len
        self.layers = layers
        self.daily_steps = daily_steps
        self.weekly_days = weekly_days
        self.num_nodes = num_nodes


        self.node_idx = to_device(torch.arange(self.num_nodes)).unsqueeze(0)  # (N,)->(1, N)

        self.input_layer = nn.ModuleList([
            HARoutingLayer(input_len, num_experts, daily_steps, weekly_days, num_nodes),
            ParallelMoEWithGLU(input_len, hidden_dim, num_experts, num_nodes, res_flag=False)]
            )

        self.AAGA = nn.ModuleList()
        self.Router = nn.ModuleList()
        self.MoE = nn.ModuleList()
        for _ in range(layers):
            self.AAGA.append(AAGA(hidden_dim))
            self.Router.append(HARoutingLayer(input_len, num_experts, daily_steps, weekly_days, num_nodes))
            self.MoE.append(ParallelMoEWithGLU(hidden_dim, hidden_dim, num_experts, num_nodes))

        self.output_layer = mlp(hidden_dim * (layers + 1), output_len)


        # adaptive agent
        self.agent = nn.Parameter(torch.empty([num_agent, hidden_dim]))
        nn.init.xavier_uniform_(self.agent)
        # time of day
        self.tod_emb = TrainableParameterLayer([daily_steps, hidden_dim])
        # day of week
        self.dow_emb = TrainableParameterLayer([weekly_days, hidden_dim])
        # node embedding
        self.node_emb = TrainableParameterLayer([num_nodes, hidden_dim])

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (torch.Tensor): shape (b, l, n, 3)
            - 0: data
            - 1: index for time of day
            - 2: index for day of week

        Returns:
            torch.Tensor: (b, p, n, 1)
        """
        B, L, N, C = history_data.shape

        raw = history_data[:, :, :, 0].transpose(2, 1).contiguous()

        # instance norm
        seq_mean = torch.mean(raw, dim=-1, keepdim=True)
        seq_var = torch.var(raw, dim=-1, keepdim=True) + 1e-5
        raw = (raw - seq_mean) / torch.sqrt(seq_var)

        day_idx  = (history_data[:, -1, :, 1] * self.daily_steps).long()   # (B, N)
        week_idx = (history_data[:, -1, :, 2] * self.weekly_days).long()   # (B, N)

        router = self.input_layer[0](raw, day_idx, week_idx, self.node_idx)
        x = self.input_layer[1](raw, router)

        # + time of day embedding
        x += self.tod_emb(day_idx).contiguous()
        # + day of week embedding
        x += self.dow_emb(week_idx).contiguous()
        # + node embedding
        x += self.node_emb(self.node_idx).contiguous()

        skip = [x]
        for i in range(self.layers):
            x = self.AAGA[i](self.agent, x)
            router = self.Router[i](raw, day_idx, week_idx, self.node_idx)
            x,s = self.MoE[i](x, router)
            skip.append(s)
        x = torch.cat(skip, dim=-1)
        x = self.output_layer(x) # b,n,p
        
        # instance denorm
        x = x * torch.sqrt(seq_var) + seq_mean
        return x.unsqueeze(-1).transpose(2, 1).contiguous()  # prediction:[b, p, n, 1]


if __name__ == "__main__":
    model = FaST(716, 96, 720)
    summary(model, [64, 96, 716, 3])
