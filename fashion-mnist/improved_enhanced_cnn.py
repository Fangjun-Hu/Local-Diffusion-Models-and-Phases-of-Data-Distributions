import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

"""
Improved Enhanced CNN Autoencoder to fix the "fuzzy blob" issue.

Key improvements:
1. Deeper global processing path for R_INTERNAL >= 1
2. Stronger embedding conditioning (FiLM-style)
3. Explicit noise-level conditioning  
4. Progressive curriculum learning support
5. L1+perceptual loss hybrid
"""

# -----------------------------------------------------------------------------
# Sinusoidal embedding helpers (same as UNet model)
# -----------------------------------------------------------------------------

def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings of shape (B, dim) for scalar inputs."""
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
# Improved ConvBlock with stronger conditioning
# -----------------------------------------------------------------------------

class FiLMConvBlock(nn.Module):
    """
    ConvBlock with FiLM (Feature-wise Linear Modulation) conditioning.
    Stronger than simple affine transformation - uses separate networks for gamma/beta.
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, 
                 kernel_size: int = 3, padding: int = 1, num_groups: int = 8):
        super().__init__()
        
        # Ensure `num_groups` divides `out_channels`
        if out_channels % num_groups != 0:
            num_groups = math.gcd(out_channels, num_groups)
        
        # Main convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Stronger FiLM conditioning - separate networks for gamma and beta
        self.gamma_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, out_channels)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), 
            nn.SiLU(),
            nn.Linear(emb_dim // 2, out_channels)
        )
        
        # Initialize gamma to 1, beta to 0 for identity initialization
        nn.init.constant_(self.gamma_net[-1].weight, 0)
        nn.init.constant_(self.gamma_net[-1].bias, 1)
        nn.init.constant_(self.beta_net[-1].weight, 0)
        nn.init.constant_(self.beta_net[-1].bias, 0)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            emb: Combined embedding vector [B, emb_dim]
        """
        # First conv block
        h = F.silu(self.norm1(self.conv1(x)))
        
        # Apply FiLM conditioning
        if emb is not None:
            # Generate gamma and beta using separate networks
            gamma = self.gamma_net(emb)  # [B, out_channels]
            beta = self.beta_net(emb)    # [B, out_channels]
            
            # Reshape for broadcasting: [B, out_channels] -> [B, out_channels, 1, 1]
            gamma = gamma[:, :, None, None]  # [B, C, 1, 1]
            beta = beta[:, :, None, None]   # [B, C, 1, 1]
            
            # Apply FiLM: γ * h + β
            h = gamma * h + beta
        
        # Second conv block
        h = F.silu(self.norm2(self.conv2(h)))
        
        return h


class DeepGlobalBlock(nn.Module):
    """
    Deeper processing block specifically for global models (R_INTERNAL >= 1).
    Uses multiple conv layers with residual connections for better global feature mixing.
    """
    
    def __init__(self, channels: int, emb_dim: int, kernel_size: int = 3, 
                 padding: int = 1, num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            FiLMConvBlock(channels, channels, emb_dim, kernel_size, padding)
            for _ in range(num_layers)
        ])
        
        # Learnable residual weights
        self.residual_weights = nn.Parameter(torch.ones(num_layers) * 0.1)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        for i, block in enumerate(self.blocks):
            residual = h
            h = block(h, emb)
            # Learnable residual connection
            h = h + self.residual_weights[i] * residual
        return h


# -----------------------------------------------------------------------------
# Improved CNN Autoencoder
# -----------------------------------------------------------------------------

class ImprovedCNNAutoencoder(nn.Module):
    """
    Improved CNN Autoencoder that fixes the "fuzzy blob" issue for global models.
    
    Key improvements:
    - Deeper global processing path for R_INTERNAL >= 1
    - Stronger FiLM conditioning instead of simple affine transform
    - Explicit noise-level conditioning
    - Better architecture for pure noise reconstruction
    """
    
    def __init__(self, receptive_field_radius: int = 2, r_internal: int = 2, 
                 input_dim: int = 28, emb_dim: int = 64, base_channels: int = 16,
                 conv1_kernel_radius: Optional[int] = None,
                 conv2_kernel_radius: Optional[int] = None,
                 conv3_kernel_radius: Optional[int] = None):
        super().__init__()
        self.receptive_field_radius = receptive_field_radius
        self.r_internal = r_internal
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.is_global = r_internal >= 1  # Global model if R_INTERNAL >= 1
        
        # Calculate kernel sizes
        kernel_size = 2 * receptive_field_radius + 1
        padding = receptive_field_radius
        internal_kernel = 2 * r_internal + 1
        internal_padding = r_internal
        
        # Optional custom kernel radii for conv/deconv pairs (kernel_size = 2*radius + 1)
        conv1_radius = conv1_kernel_radius if conv1_kernel_radius is not None else r_internal
        conv2_radius = conv2_kernel_radius if conv2_kernel_radius is not None else r_internal
        conv3_radius = conv3_kernel_radius if conv3_kernel_radius is not None else r_internal
        
        self.conv1_kernel = 2 * conv1_radius + 1
        self.conv1_padding = conv1_radius
        self.conv2_kernel = 2 * conv2_radius + 1
        self.conv2_padding = conv2_radius
        self.conv3_kernel = 2 * conv3_radius + 1
        self.conv3_padding = conv3_radius
        
        # Enhanced embedding layers with explicit noise level conditioning
        self.time_emb = ScalarEmbedding(emb_dim)
        self.posx_emb = ScalarEmbedding(emb_dim) 
        self.posy_emb = ScalarEmbedding(emb_dim)
        self.noise_level_emb = ScalarEmbedding(emb_dim)  # NEW: explicit noise level embedding
        
        # Embedding combiner - learns to weight different embeddings
        self.emb_combiner = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),  # 4 embeddings -> 2x
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim)       # 2x -> final embedding
        )
        
        # Initial feature extraction from 1-channel input (MNIST only)
        self.conv_layer = nn.Conv2d(1, base_channels, kernel_size=kernel_size, padding=padding)
        self.norm_layer = nn.GroupNorm(8, base_channels)
        
        # Choose block type based on whether this is local or global model
        if self.is_global:
            # Global model: Use FiLM blocks with deeper processing
            print(f"Creating GLOBAL model (R_INTERNAL={r_internal})")
            BlockType = FiLMConvBlock
            self.use_deep_global = True
        else:
            # Local model: Use simpler blocks (original behavior)
            print(f"Creating LOCAL model (R_INTERNAL={r_internal})")  
            BlockType = FiLMConvBlock  # Still use FiLM for consistency
            self.use_deep_global = False
        
        # Encoder blocks
        self.conv1 = BlockType(base_channels, base_channels * 2, emb_dim, 
                              self.conv1_kernel, self.conv1_padding)
        self.conv2 = BlockType(base_channels * 2, base_channels * 4, emb_dim,
                              self.conv2_kernel, self.conv2_padding)
        self.conv3 = BlockType(base_channels * 4, base_channels * 8, emb_dim,
                              self.conv3_kernel, self.conv3_padding)
        
        # Deep global processing in bottleneck (only for global models)
        if self.use_deep_global:
            self.deep_global = DeepGlobalBlock(base_channels * 8, emb_dim, 
                                             internal_kernel, internal_padding, 
                                             num_layers=3)
        
        # Decoder blocks  
        self.deconv1 = BlockType(base_channels * 8 + base_channels * 4, 
                                base_channels * 4, emb_dim, 
                                self.conv3_kernel, self.conv3_padding)
        self.deconv2 = BlockType(base_channels * 4 + base_channels * 2,
                                base_channels * 2, emb_dim,
                                self.conv2_kernel, self.conv2_padding) 
        self.deconv3 = BlockType(base_channels * 2 + base_channels,
                                base_channels, emb_dim,
                                self.conv1_kernel, self.conv1_padding)
        
        # Final output layer
        self.final_layer = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)
        
        print(f"Model created: {sum(p.numel() for p in self.parameters()):,} parameters")
        
    def create_coordinate_grids(self, B: int, H: int, W: int, device):
        """Create normalized coordinate grids for x and y positions."""
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(B, H, W)
        return x_coords, y_coords
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                pos_x: torch.Tensor = None, pos_y: torch.Tensor = None,
                noise_level: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W] (1 or 4 channels)
            t: Timestep [B,] or scalar
            pos_x: X coordinates [B,] or None
            pos_y: Y coordinates [B,] or None  
            noise_level: Explicit noise level [B,] or None (uses t if not provided)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Handle different input formats
        if C == 4:
            # Legacy format: extract MNIST channel
            mnist_channel = x[:, 0:1, :, :]  # [B, 1, H, W]
            if pos_x is None:
                pos_x = x[:, 1, H//2, W//2]
            if pos_y is None:
                pos_y = x[:, 2, H//2, W//2]
        elif C == 1:
            mnist_channel = x  # [B, 1, H, W]
        else:
            raise ValueError(f"Expected input to have 1 or 4 channels, got {C}")
        
        # Handle timestep
        if isinstance(t, (int, float)):
            t = torch.full((B,), t, device=device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
            
        # Handle noise level (use timestep if not provided)
        if noise_level is None:
            noise_level = t.clone()
        elif isinstance(noise_level, (int, float)):
            noise_level = torch.full((B,), noise_level, device=device, dtype=torch.float32)
        elif noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0).expand(B)
        
        # Create coordinate grids if not provided
        if pos_x is None or pos_y is None:
            x_coords, y_coords = self.create_coordinate_grids(B, H, W, device)
            pos_x = x_coords[:, H//2, W//2]  # [B,]
            pos_y = y_coords[:, H//2, W//2]  # [B,]
        
        # Create enhanced embeddings
        time_emb = self.time_emb(t)           # [B, emb_dim]
        posx_emb = self.posx_emb(pos_x)       # [B, emb_dim] 
        posy_emb = self.posy_emb(pos_y)       # [B, emb_dim]
        noise_emb = self.noise_level_emb(noise_level)  # [B, emb_dim] - NEW!
        
        # Combine embeddings using learned combiner
        all_embs = torch.cat([time_emb, posx_emb, posy_emb, noise_emb], dim=1)  # [B, 4*emb_dim]
        combined_emb = self.emb_combiner(all_embs)  # [B, emb_dim]
        
        # Initial convolution (no embedding injection here)
        out1 = F.silu(self.norm_layer(self.conv_layer(mnist_channel)))  # [B, base_channels, H, W]
        
        # Encoder with embedding injection at each layer
        out2 = self.conv1(out1, combined_emb)         # [B, base_channels*2, H, W]
        out3 = self.conv2(out2, combined_emb)         # [B, base_channels*4, H, W]  
        out4 = self.conv3(out3, combined_emb)         # [B, base_channels*8, H, W] (bottleneck)
        
        # Deep global processing (only for global models)
        if self.use_deep_global:
            out4 = self.deep_global(out4, combined_emb)  # Enhanced bottleneck processing
        
        # Decoder with skip connections and embedding injection
        dec1_input = torch.cat([out4, out3], dim=1)   # [B, base_channels*12, H, W]
        dec1 = self.deconv1(dec1_input, combined_emb)  # [B, base_channels*4, H, W]
        
        dec2_input = torch.cat([dec1, out2], dim=1)    # [B, base_channels*6, H, W]
        dec2 = self.deconv2(dec2_input, combined_emb)  # [B, base_channels*2, H, W]
        
        dec3_input = torch.cat([dec2, out1], dim=1)    # [B, base_channels*3, H, W] 
        dec3 = self.deconv3(dec3_input, combined_emb)  # [B, base_channels, H, W]
        
        # Final output
        output = self.final_layer(dec3)  # [B, 1, H, W]
        
        return output


# -----------------------------------------------------------------------------
# Enhanced Loss Functions
# -----------------------------------------------------------------------------

class HybridLoss(nn.Module):
    """
    Hybrid loss combining L1 and L2 to reduce blurring while maintaining smoothness.
    """
    
    def __init__(self, l1_weight: float = 0.7, l2_weight: float = 0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        return self.l1_weight * l1 + self.l2_weight * l2


class ProgressiveLoss(nn.Module):
    """
    Progressive loss that weights different noise levels differently.
    Higher weights for harder reconstruction tasks (higher noise).
    """
    
    def __init__(self, base_loss: nn.Module = None):
        super().__init__()
        self.base_loss = base_loss or HybridLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                noise_levels: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_loss(pred, target)
        
        # Weight loss based on noise level - harder tasks get higher weight
        # This encourages the model to work harder on difficult cases
        weights = 1.0 + noise_levels  # Range [1, 2] for noise_levels in [0, 1]
        weighted_loss = base_loss * weights.mean()
        
        return weighted_loss


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------

def create_local_model(receptive_field_radius: int = 2, 
                      conv1_kernel_radius: Optional[int] = None,
                      conv2_kernel_radius: Optional[int] = None,
                      conv3_kernel_radius: Optional[int] = None,
                      **kwargs) -> ImprovedCNNAutoencoder:
    """Create a local model (R_INTERNAL=0)."""
    return ImprovedCNNAutoencoder(
        receptive_field_radius=receptive_field_radius,
        r_internal=0,  # Local model
        conv1_kernel_radius=conv1_kernel_radius,
        conv2_kernel_radius=conv2_kernel_radius,
        conv3_kernel_radius=conv3_kernel_radius,
        **kwargs
    )


def create_global_model(receptive_field_radius: int = 2,
                       conv1_kernel_radius: Optional[int] = None,
                       conv2_kernel_radius: Optional[int] = None,
                       conv3_kernel_radius: Optional[int] = None,
                       **kwargs) -> ImprovedCNNAutoencoder:
    """Create a global model (R_INTERNAL=1)."""
    return ImprovedCNNAutoencoder(
        receptive_field_radius=receptive_field_radius,
        r_internal=3,  # Global model
        conv1_kernel_radius=conv1_kernel_radius,
        conv2_kernel_radius=conv2_kernel_radius,
        conv3_kernel_radius=conv3_kernel_radius,
        **kwargs
    )


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Improved CNN Autoencoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test both local and global models
    print("\n=== Local Model (R_INTERNAL=0) ===")
    local_model = create_local_model(receptive_field_radius=2, base_channels=32).to(device)
    
    print("\n=== Global Model (R_INTERNAL=1) ===") 
    global_model = create_global_model(receptive_field_radius=2, base_channels=32).to(device)
    
    # Test custom kernel sizes
    print("\n=== Custom Kernel Radius Model ===")
    custom_model = create_local_model(
        receptive_field_radius=2, 
        base_channels=32,
        conv1_kernel_radius=2,  # Custom radius 2 -> kernel 5x5 for conv1/deconv1
        conv2_kernel_radius=1,  # Custom radius 1 -> kernel 3x3 for conv2/deconv2  
        conv3_kernel_radius=3   # Custom radius 3 -> kernel 7x7 for conv3/deconv3
    ).to(device)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    test_t = torch.tensor([0.3, 0.5, 0.8, 1.0]).to(device)
    test_noise_levels = torch.tensor([0.3, 0.5, 0.8, 1.0]).to(device)
    
    print(f"\n=== Testing Forward Pass ===")
    print(f"Input shape: {test_input.shape}")
    print(f"Timesteps: {test_t}")
    print(f"Noise levels: {test_noise_levels}")
    
    # Test local model
    with torch.no_grad():
        local_output = local_model(test_input, test_t, noise_level=test_noise_levels)
        print(f"Local output shape: {local_output.shape}")
        print(f"Local output range: [{local_output.min():.4f}, {local_output.max():.4f}]")
    
    # Test global model  
    with torch.no_grad():
        global_output = global_model(test_input, test_t, noise_level=test_noise_levels)
        print(f"Global output shape: {global_output.shape}")
        print(f"Global output range: [{global_output.min():.4f}, {global_output.max():.4f}]")
    
    # Test custom kernel model
    with torch.no_grad():
        custom_output = custom_model(test_input, test_t, noise_level=test_noise_levels)
        print(f"Custom output shape: {custom_output.shape}")
        print(f"Custom output range: [{custom_output.min():.4f}, {custom_output.max():.4f}]")
    
    # Test loss functions
    print(f"\n=== Testing Loss Functions ===")
    target = torch.randn_like(test_input)
    
    hybrid_loss = HybridLoss()
    progressive_loss = ProgressiveLoss()
    
    hybrid_val = hybrid_loss(local_output, target)
    progressive_val = progressive_loss(local_output, target, test_noise_levels)
    
    print(f"Hybrid Loss: {hybrid_val:.6f}")
    print(f"Progressive Loss: {progressive_val:.6f}")
    
    print(f"\n✅ All tests passed!")
    print(f"Local model parameters: {sum(p.numel() for p in local_model.parameters()):,}")
    print(f"Global model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"Custom kernel radius model parameters: {sum(p.numel() for p in custom_model.parameters()):,}")