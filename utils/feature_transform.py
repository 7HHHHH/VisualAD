"""
Feature transformation module
Provides simple linear layers, MLP layers, residual MLP, and adapter residual layers for feature transformation
"""

import torch
import torch.nn as nn


class SimpleLinearTransform(nn.Module):
    """Simple linear transformation layer for quick testing"""
    def __init__(self, input_dim, output_dim=None, dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.transform(x)


class FeatureTransformMLP(nn.Module):
    """Complete MLP module for more complex feature transformations"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPTransform(nn.Module):
    """Standard MLP with residual branch to both enhance expressiveness and preserve CLIP original features"""

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 2)
        if output_dim is None:
            output_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        self.residual_proj = None
        if output_dim != input_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x


class ResidualAdapterTransform(nn.Module):
    """LoRA-style residual adapter for fine-tuning features while preserving CLIP semantics"""

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1, init_scale=1e-3):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if output_dim != input_dim:
            raise ValueError("ResidualAdapterTransform requires output_dim to match input_dim for residual addition")
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 4)

        self.pre_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, input_dim, bias=False)

        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1) * init_scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        for module in self.context_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x)
        adapter = self.down_proj(x)
        gate = self.context_gate(adapter)
        adapter = self.act(adapter)
        adapter = adapter * gate
        adapter = self.up_proj(adapter)
        adapter = self.dropout(adapter)
        return residual + self.res_scale * adapter


class LeakyReLUTransform(nn.Module):
    """Linear+LeakyReLU+Linear+LeakyReLU transformation layer"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1, negative_slope=0.01):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = input_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.transform(x)


def create_feature_transform(transform_type="linear", input_dim=1024, hidden_dim=None, output_dim=None, dropout=0.1, negative_slope=0.01):
    """
    Factory function to create feature transformation modules

    Args:
        transform_type: "linear", "mlp", "mlp_residual", "adapter" or "leakyrelu"
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (used by MLP/Adapter/LeakyReLU)
        output_dim: Output feature dimension
        dropout: Dropout rate
        negative_slope: Negative slope for LeakyReLU (only used by LeakyReLU)

    Returns:
        Feature transformation module
    """
    if transform_type == "linear":
        return SimpleLinearTransform(input_dim, output_dim, dropout)
    elif transform_type == "mlp":
        return FeatureTransformMLP(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "mlp_residual":
        return ResidualMLPTransform(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "adapter":
        return ResidualAdapterTransform(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "leakyrelu":
        return LeakyReLUTransform(input_dim, hidden_dim, output_dim, dropout, negative_slope)
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}. Use 'linear', 'mlp', 'mlp_residual', 'adapter', or 'leakyrelu'")


# For backward compatibility, keep original class names
LinearTransform = SimpleLinearTransform
MLPTransform = FeatureTransformMLP
