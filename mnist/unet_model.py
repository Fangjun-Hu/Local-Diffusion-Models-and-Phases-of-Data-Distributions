import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""UNet for diffusion models with scalar sinusoidal conditioning.

This version supports three **scalar** conditioning signals per sample:
  • `t`      – diffusion timestep
  • `pos_x`  – x‑coordinate (any scalar)
  • `pos_y`  – y‑coordinate (any scalar)

The three embeddings are added together and injected into every residual
block, exactly like the timestep embedding in the original DDPM paper.

Usage
-----
>>> model = UNet(in_channels=1, out_channels=1)   # MNIST
>>> logits = model(img, t, pos_x, pos_y)
All conditioning tensors must be shape `(B,)` or broadcastable to it.
"""

# -----------------------------------------------------------------------------
# sinusoidal + MLP embedding helpers
# -----------------------------------------------------------------------------
def build_xy_coordinates(H: int, W: int, device=None):
    """Return (1, 1, H, W) tensors with absolute x and y coords in [0, 1]."""
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
    ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
    return xs, ys
def sinusoidal_2d(x_map: torch.Tensor, y_map: torch.Tensor, dim: int = 64):
    """
    Create a (B, 2*dim, H, W) positional encoding from normalized x/y maps.

    dim must be divisible by 2.  Half the channels are x-freqs, half y-freqs.
    """
    assert dim % 2 == 0, "dim must be even"
    B, _, H, W = x_map.shape
    device = x_map.device

    half = dim // 2
    # Frequencies: 1, 2, 4, …, 2^{half-1}
    freq = torch.arange(half, device=device).float()
    freq = 2 ** freq  # (half,)

    # Reshape for broadcasting
    freq = freq.view(1, half, 1, 1)

    # Encode x
    x_enc = x_map * freq * math.pi
    x_sin = torch.sin(x_enc)
    x_cos = torch.cos(x_enc)

    # Encode y
    y_enc = y_map * freq * math.pi
    y_sin = torch.sin(y_enc)
    y_cos = torch.cos(y_enc)

    # Concatenate to (B, 2*dim, H, W)
    pos = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=1)
    return pos
def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings of shape (B, dim) for scalar inputs.
    `values` is a 1‑D tensor of shape (B,) containing scalar floats/ints.
    """
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=values.device) * -(math.log(10000) / (half_dim - 1))
    )
    angles = values[:, None] * freqs[None, :]
    emb = torch.cat((angles.sin(), angles.cos()), dim=1)
    return emb  # (B, dim)


class ScalarEmbedding(nn.Module):
    """Sinusoidal embedding followed by two SiLU‑MLP layers (Dim → 4×Dim → Dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        emb = sinusoidal_embedding(x.float(), self.linear1.in_features)
        emb = F.silu(self.linear1(emb))
        return self.linear2(emb)  # (B, dim)


# -----------------------------------------------------------------------------
# Core building blocks
# -----------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """(Conv → GN → SiLU) × 2 with added embedding bias. If emb is None, it means x is concat(x, emb).  """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int = 8):
        super().__init__()

        # Ensure `num_groups` divides `out_channels`.
        if out_channels % num_groups != 0:
            num_groups = math.gcd(out_channels, num_groups)

        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, emb) -> torch.Tensor:
        h = self.block1(x)
        if emb is not None:
            gamma, beta = torch.chunk(self.emb_proj(emb), 2, dim=1)
            # print(gamma.shape, beta.shape)
            h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        return self.block2(h)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, emb):
        x = self.conv(x, emb)
        return self.pool(x), x  # pooled, residual


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, emb)


# -----------------------------------------------------------------------------
# Full U‑Net
# -----------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, emb_dim: int = 256, use_pos: bool = False, num_classes = None):
        super().__init__()
        # Embeddings for t, pos_x, pos_y
        self.time_emb = ScalarEmbedding(emb_dim)
        self.use_pos = use_pos
        if use_pos:
            self.posx_emb = ScalarEmbedding(emb_dim)
            self.posy_emb = ScalarEmbedding(emb_dim)
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, emb_dim)

        self.emb_combine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.down1 = Down(in_channels, 64, emb_dim)
        self.down2 = Down(64, 128, emb_dim)
        self.down3 = Down(128, 256, emb_dim)

        self.mid = ConvBlock(256, 512, emb_dim)

        self.up1 = Up(512, 256, emb_dim)
        self.up2 = Up(256, 128, emb_dim)
        self.up3 = Up(128, 64, emb_dim)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,          # (B, C, H, W)
        t: torch.Tensor,          # (B,) diffusion step
        pos_x: torch.Tensor = None,      # (B, 1, H, W) scalar x‑position (any units)
        pos_y: torch.Tensor = None,      # (B, 1, H, W) scalar y‑position
        cond: torch.Tensor = None, # (B,)
    ) -> torch.Tensor:
        # Combined embedding
        emb = self.time_emb(t)
        if self.use_pos:
            emb += self.posx_emb(pos_x) + self.posy_emb(pos_y)  # (B, emb_dim)
        if cond is not None:
            emb_parts = torch.cat([
                self.time_emb(t),
                self.class_emb(cond),
            ], dim=1)
            emb = self.emb_combine(emb_parts)

        # Encoder
        x1, res1 = self.down1(x, emb)
        x2, res2 = self.down2(x1, emb)
        x3, res3 = self.down3(x2, emb)

        # Bottleneck
        x_mid = self.mid(x3, emb)
        
        # Decoder
        x = self.up1(x_mid, res3, emb)
        x = self.up2(x, res2, emb)
        x = self.up3(x, res1, emb)

        return self.out_conv(x)


if __name__ == "__main__":

    xs, ys = build_xy_coordinates(28,28)
    pos = sinusoidal_2d(xs, ys, dim=16)
    pos_exp = pos.expand(8, -1, -1, -1) 

    img = torch.randn(8, 1, 28, 28)
    img = torch.cat([img, pos_exp], dim=1)

    print(img.shape)
    t = torch.randint(0, 1000, (8,))
    labels = torch.randint(0, 10, (8,))
    # MNIST example (grayscale)
    model = UNet(in_channels=1+16*2, num_classes=10)

    out = model(img, t, labels)
    print(out.shape)  # (8, 1, 28, 28)

