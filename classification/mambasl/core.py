""" """

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from pypots.nn.modules import ModelCore
from pypots.nn.modules.loss import Criterion


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe = torch.zeros(max_len, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding_cls(nn.Module):
    # original TokenEmbedding in tslib uses fixed d_kernel(=3).
    # Keep the class name and behavior used by the original MambaSL codebase.
    def __init__(self, c_in: int, d_model: int, d_kernel: int = 3):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=d_kernel,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding_cls(nn.Module):
    # original DataEmbedding in tslib uses fixed max_len(=5000).
    # MambaSL sets max_len=max(5000, seq_len) to avoid long-sequence warnings.
    def __init__(self, c_in: int, d_model: int, seq_len: int, dropout: float, d_kernel: int):
        super().__init__()
        self.value_embedding = TokenEmbedding_cls(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class Mamba_TimeVariant(nn.Module):
    """Mamba block with time-variant switches for dt/B/C.

    This follows the naming and init/forward structure used in the original
    MambaSL repository.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        tv_dt: bool,
        tv_B: bool,
        tv_C: bool,
        use_D: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.tv_dt = tv_dt
        self.tv_B = tv_B
        self.tv_C = tv_C

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.dw_conv = (
            nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias=True,
            )
            if d_conv > 0
            else nn.Identity()
        )
        self.act = nn.SiLU()
        self.activation = "silu"

        self.tv_proj_dim = [0, 0, 0]
        if tv_dt:
            self.tv_proj_dim[0] = self.dt_rank
        if tv_B:
            self.tv_proj_dim[1] = d_state
        if tv_C:
            self.tv_proj_dim[2] = d_state
        tv_proj_dim = sum(self.tv_proj_dim)
        self.x_proj = nn.Linear(self.d_inner, tv_proj_dim, bias=False) if tv_proj_dim > 0 else None

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(
            min=1e-4
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        if not tv_B:
            self.B = nn.Parameter(torch.rand(self.d_inner, d_state))
            self.B._no_weight_decay = True
        if not tv_C:
            self.C = nn.Parameter(torch.rand(self.d_inner, d_state))
            self.C._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner)) if use_D else None
        if self.D is not None:
            self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _masked_split(self, x_dbl: torch.Tensor) -> tuple:
        dt = B = C = None
        start = 0
        if self.tv_dt:
            end = start + self.dt_rank
            dt = x_dbl[..., start:end]
            start = end
        if self.tv_B:
            end = start + self.d_state
            B = x_dbl[..., start:end]
            start = end
        if self.tv_C:
            C = x_dbl[..., start : start + self.d_state]
        return dt, B, C

    def _forward_fallback(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        A: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        # Fallback path when selective_scan_fn is unavailable.
        # This mirrors the recurrence equations used by Mamba's step path,
        # but does not replicate the fused CUDA kernel implementation details.
        batch_size, _, seq_len = x.shape
        state = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            if self.tv_dt:
                dt_t = F.softplus(dt[:, :, t] + self.dt_proj.bias.to(dtype=dt.dtype))
            else:
                dt_t = F.softplus(self.dt_proj.bias.to(dtype=x.dtype)).unsqueeze(0).expand(batch_size, -1)

            dA = torch.exp(torch.einsum("bd,dn->bdn", dt_t, A))
            if self.tv_B:
                dB = torch.einsum("bd,bn->bdn", dt_t, B[:, :, t])
            else:
                dB = torch.einsum("bd,dn->bdn", dt_t, self.B)

            state = state * dA + x[:, :, t].unsqueeze(-1) * dB

            if self.tv_C:
                y_t = torch.einsum("bdn,bn->bd", state, C[:, :, t])
            else:
                y_t = torch.einsum("bdn,dn->bd", state, self.C)

            if self.D is not None:
                y_t = y_t + self.D.to(dtype=x.dtype) * x[:, :, t]

            y_t = y_t * self.act(z[:, :, t])
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Keep the original MambaBlock layout: BLD -> D(BL) matmul -> BDL
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seq_len,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, z = xz.chunk(2, dim=1)

        if self.d_conv == 0:
            x = self.act(x)
        elif (causal_conv1d_fn is None) or (self.d_conv not in [2, 3, 4]):
            x = self.act(self.dw_conv(x)[..., :seq_len])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.dw_conv.weight, "d 1 w -> d w"),
                bias=self.dw_conv.bias,
                activation=self.activation,
            )

        dt_raw = B_raw = C_raw = None
        if self.x_proj is not None:
            x_dbl = self.x_proj(x.permute(0, 2, 1).reshape(batch_size * seq_len, self.d_inner))
            dt_raw, B_raw, C_raw = self._masked_split(x_dbl)

        if self.tv_dt:
            dt = F.linear(dt_raw, self.dt_proj.weight).view(batch_size, seq_len, self.d_inner).permute(0, 2, 1)
        else:
            dt = torch.zeros(
                batch_size,
                self.d_inner,
                seq_len,
                device=self.dt_proj.bias.device,
                dtype=self.dt_proj.bias.dtype,
            )

        if self.tv_B:
            B_seq = B_raw.view(batch_size, seq_len, self.d_state).permute(0, 2, 1)
        else:
            B_seq = self.B

        if self.tv_C:
            C_seq = C_raw.view(batch_size, seq_len, self.d_state).permute(0, 2, 1)
        else:
            C_seq = self.C

        A = -torch.exp(self.A_log.float())

        if selective_scan_fn is not None:
            y = selective_scan_fn(
                x,
                dt,
                A,
                B_seq,
                C_seq,
                self.D,
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = rearrange(y, "b d l -> b l d")
        else:
            y = self._forward_fallback(x, z, A, dt, B_seq, C_seq)

        return self.out_proj(y)


class _MambaSL(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        n_kernels: int,
        projection_type: str,
        n_heads: int,
        tv_dt: bool,
        tv_B: bool,
        tv_C: bool,
        use_D: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_classes = n_classes
        self.projection_type = projection_type
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # Unlike MASCIT, the original MambaSL embeds only the value channels.
        self.embedding = DataEmbedding_cls(
            c_in=n_features,
            d_model=d_model,
            seq_len=n_steps,
            dropout=dropout,
            d_kernel=n_kernels,
        )
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                tv_dt=tv_dt,
                tv_B=tv_B,
                tv_C=tv_C,
                use_D=use_D,
            ),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        output_features = d_model * n_steps if projection_type == "full" else d_model
        self.out_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_features, n_classes, bias=False),
        )

        if projection_type == "gating":
            self.attn_weight = nn.Linear(d_model, n_heads, bias=True)
            nn.init.zeros_(self.attn_weight.weight)
            if self.attn_weight.bias is not None:
                self.attn_weight.bias.data.fill_(1.0)
            self.gating_values = []

        for module in self.out_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        valid_step_mask = (missing_mask.sum(dim=2) > 0).to(X.dtype)
        mamba_in = self.embedding(X)
        mamba_out = self.mamba(mamba_in)

        if self.projection_type == "full":
            logits = self.out_layer(mamba_out.reshape(mamba_out.size(0), -1))
        elif self.projection_type == "last":
            logits = self.out_layer(mamba_out[:, -1, :])
        elif self.projection_type == "avg":
            step_logits = self.out_layer(mamba_out) * valid_step_mask.unsqueeze(2)
            logits = step_logits.mean(dim=1)
        elif self.projection_type == "max":
            step_logits = self.out_layer(mamba_out) * valid_step_mask.unsqueeze(2)
            logits = step_logits.max(dim=1).values
        elif self.projection_type == "gating":
            step_logits = self.out_layer(mamba_out) * valid_step_mask.unsqueeze(2)
            gate_scores = self.attn_weight(mamba_out).amax(dim=2)
            weights = torch.softmax(gate_scores, dim=1).unsqueeze(2)
            logits = (step_logits * weights).sum(dim=1)
            gini = weights.detach().squeeze(-1).pow(2).sum(-1).tolist()
            if isinstance(gini, float):
                gini = [gini]
            self.gating_values.extend(gini)
        else:
            raise ValueError(
                "projection_type must be one of ['full', 'last', 'avg', 'max', 'gating'], "
                f"but got {self.projection_type}"
            )

        classification_proba = torch.softmax(logits, dim=1)
        results = {
            "classification_proba": classification_proba,
            "logits": logits,
        }

        if calc_criterion:
            if self.training:
                results["loss"] = self.training_loss(logits, inputs["y"])
            else:
                results["metric"] = self.validation_metric(logits, inputs["y"])

        return results
