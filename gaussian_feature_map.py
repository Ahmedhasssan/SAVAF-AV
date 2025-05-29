import numpy as np
import torch
import matplotlib.pyplot as plt

class GaussianFeatureMap:
    def __init__(self, resolution=(270, 480), feature_dim=32):
        """
        Convert 3D Gaussian parameters to 2D visual feature maps
        
        Args:
            resolution: Tuple of (height, width) for the feature map
            feature_dim: Dimension of the output feature at each spatial location (default: 32)
        """
        self.resolution = resolution
        self.feature_dim = feature_dim
        
    def project_gaussians(self, positions, scales, rotations, opacities, camera_matrix=None):
        """
        Project 3D Gaussians to 2D space
        
        Args:
            positions: Tensor of shape [N, 3] for xyz positions
            scales: Tensor of shape [N, 3] for xyz scales
            rotations: Tensor of shape [N, 4] for quaternion rotations or [N, 3, 3] for rotation matrices
            opacities: Tensor of shape [N, 1] for opacity values
            camera_matrix: Optional camera projection matrix of shape [3, 4]
            
        Returns:
            Dictionary of projected 2D Gaussian parameters
        """
        if camera_matrix is None:
            # Default simple orthographic projection
            camera_matrix = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.0]
            ], dtype=positions.dtype, device=positions.device)
        
        # Apply camera transformation to get 2D positions
        homogeneous_positions = torch.cat(
            [positions, torch.ones(positions.shape[0], 1, device=positions.device)], 
            dim=1
        )
        projected_positions = torch.matmul(homogeneous_positions, camera_matrix.t())
        
        # Normalize by depth for perspective projection
        projected_positions = projected_positions[:, :2] / projected_positions[:, 2:3]
        
        # Get depth values for z-ordering
        depth_values = positions[:, 2]
        
        # Simplify scale projection (more accurate would involve full covariance)
        projected_scales = scales[:, :2]
        
        # Pack into dictionary
        projected_params = {
            'positions': projected_positions,  # [N, 2]
            'scales': projected_scales,        # [N, 2]
            'rotations': rotations,            # [N, 4] or [N, 3, 3]
            'opacities': opacities,            # [N, 1]
            'depth': depth_values              # [N]
        }
        
        return projected_params
    
    def create_feature_map(self, projected_params, feature_encoding='default'):
        """
        Create a 2D feature map from projected Gaussian parameters
        
        Args:
            projected_params: Dictionary of projected parameters from project_gaussians
            feature_encoding: Method to encode features ('default', 'positional', 'fourier')
            
        Returns:
            Feature map tensor of shape [H, W, feature_dim]
        """
        H, W = self.resolution
        
        # Initialize empty feature map
        feature_map = torch.zeros(H, W, self.feature_dim, 
                                  dtype=projected_params['positions'].dtype,
                                  device=projected_params['positions'].device)
        
        # Extract parameters
        positions = projected_params['positions']  # [N, 2]
        scales = projected_params['scales']        # [N, 2]
        opacities = projected_params['opacities']  # [N, 1]
        depths = projected_params['depth']         # [N]
        
        # Sort gaussians by depth for correct layering (back to front)
        sorted_indices = torch.argsort(depths)
        positions = positions[sorted_indices]
        scales = scales[sorted_indices]
        opacities = opacities[sorted_indices]
        
        # Scale positions to image coordinates
        positions_img = positions.clone()
        positions_img[:, 0] = (positions_img[:, 0] + 1) * W / 2
        positions_img[:, 1] = (positions_img[:, 1] + 1) * H / 2
        
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=positions.dtype, device=positions.device),
            torch.arange(W, dtype=positions.dtype, device=positions.device),
            indexing='ij'
        )
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        
        # Process each Gaussian
        for i in range(positions.shape[0]):
            pos = positions_img[i]
            scale = scales[i] * min(H, W) / 4  # Scale appropriately for image size
            opacity = opacities[i]
            
            # Compute Gaussian influence at each pixel
            # ||grid - pos||^2 / scale^2
            dist_sq = torch.sum(((grid - pos) / scale) ** 2, dim=-1)
            influence = torch.exp(-0.5 * dist_sq) * opacity
            
            # Only process pixels with significant influence for efficiency
            significant = influence > 0.01
            if not torch.any(significant):
                continue
                
            # Create feature for this Gaussian
            if feature_encoding == 'positional':
                # Positional encoding (like in NeRF)
                feature = self._positional_encoding(pos, scale, opacity, i, self.feature_dim)
            elif feature_encoding == 'fourier':
                # Fourier features
                feature = self._fourier_features(pos, scale, opacity, self.feature_dim)
            else:
                # Default: Simple feature based on properties
                # Concatenate normalized position, scale, depth, opacity, and identity
                normalized_pos = positions[i]  # Keep in [-1, 1] range
                normalized_scale = scale / (min(H, W) / 2)
                feature_id = torch.zeros(self.feature_dim, device=positions.device)
                feature_id[i % self.feature_dim] = 1.0  # One-hot encoding by index (cyclic)
                
                # Fill feature vector (adapt to your feature_dim)
                feature = torch.zeros(self.feature_dim, device=positions.device)
                # Handle values individually to avoid shape issues
                if self.feature_dim > 0:
                    feature[0] = normalized_pos[0].item()
                if self.feature_dim > 1:
                    feature[1] = normalized_pos[1].item()
                if self.feature_dim > 2:
                    feature[2] = normalized_scale[0].item()
                if self.feature_dim > 3:
                    feature[3] = normalized_scale[1].item()
                if self.feature_dim > 4:
                    feature[4] = depths[i].item()
                if self.feature_dim > 5:
                    feature[5] = opacity.item()
                
                # Make sure we don't exceed feature_dim when copying feature_id
                if self.feature_dim > 6:
                    max_copy = min(self.feature_dim - 6, feature_id.shape[0])
                    feature[6:6+max_copy] = feature_id[:max_copy]
            
            # Blend feature into the feature map based on influence
            # Use broadcasting for efficient computation
            influence_expanded = influence.unsqueeze(-1)
            current_values = feature_map[significant]
            new_values = (1 - influence_expanded[significant]) * current_values + \
                        influence_expanded[significant] * feature
            feature_map[significant] = new_values
        
        return feature_map
    
    def _positional_encoding(self, pos, scale, opacity, idx, dim):
        """Generate positional encoding features similar to NeRF"""
        freq_bands = 2.0 ** torch.linspace(0, dim//4-1, dim//4, device=pos.device)
        
        # Create input vector with position, scale, opacity
        x = torch.cat([pos, scale, opacity], dim=-1)
        
        # Apply sin/cos to each component with different frequencies
        encoding = []
        for freq in freq_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))
        
        # Concat all encodings
        encoding = torch.cat(encoding, dim=-1)
        
        # If needed, pad or trim to match feature_dim
        if encoding.shape[-1] < dim:
            padding = torch.zeros(dim - encoding.shape[-1], device=pos.device)
            encoding = torch.cat([encoding, padding], dim=-1)
        else:
            encoding = encoding[:dim]
            
        return encoding
    
    def _fourier_features(self, pos, scale, opacity, dim):
        """Generate random Fourier features"""
        # Create fixed random projection matrix
        torch.manual_seed(0)  # For reproducibility
        B = torch.randn(5, dim//2, device=pos.device)
        
        # Create input vector with position, scale, opacity
        x = torch.cat([pos, scale, opacity], dim=-1)
        
        # Project and apply sin/cos
        projection = torch.matmul(x, B)
        features = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        
        return features
    
    def visualize_feature_map(self, feature_map, method='pca'):
        """
        Visualize the feature map for inspection
        
        Args:
            feature_map: Feature map tensor of shape [H, W, feature_dim]
            method: Visualization method ('pca', 'rgb', 'channels')
            
        Returns:
            Matplotlib figure
        """
        feature_map_np = feature_map.detach().cpu().numpy()
        H, W, C = feature_map_np.shape
        
        if method == 'pca':
            # PCA visualization (reduce to 3 dimensions for RGB)
            from sklearn.decomposition import PCA
            
            # Reshape to [H*W, C]
            features_flat = feature_map_np.reshape(-1, C)
            
            # Fit PCA
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features_flat)
            
            # Normalize to [0, 1] for RGB
            features_pca = (features_pca - features_pca.min(axis=0)) / \
                          (features_pca.max(axis=0) - features_pca.min(axis=0) + 1e-8)
            
            # Reshape back to image
            rgb_image = features_pca.reshape(H, W, 3)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_image)
            ax.set_title('PCA Visualization of Feature Map')
            ax.axis('off')
            
        elif method == 'rgb':
            # Use first 3 channels as RGB
            rgb_channels = min(3, C)
            rgb_image = feature_map_np[:, :, :rgb_channels]
            
            # Normalize each channel
            for i in range(rgb_channels):
                channel = rgb_image[:, :, i]
                rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            # Pad if needed
            if rgb_channels < 3:
                padding = np.zeros((H, W, 3 - rgb_channels))
                rgb_image = np.concatenate([rgb_image, padding], axis=-1)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_image)
            ax.set_title('First 3 Channels as RGB')
            ax.axis('off')
            
        else:  # 'channels'
            # Show individual channels
            num_channels = min(16, C)  # Show up to 16 channels
            rows, cols = int(np.ceil(np.sqrt(num_channels))), int(np.ceil(np.sqrt(num_channels)))
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            axes = axes.flatten()
            
            for i in range(num_channels):
                channel = feature_map_np[:, :, i]
                normalized = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                
                axes[i].imshow(normalized, cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_channels, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
        
        return fig

# Example usage
def create_gaussian_feature_map(
    positions,
    scales,
    rotations,
    opacities,
    camera_matrix,
    n_gaussians=100,
    resolution=(170, 240),
    feature_dim=1,
    feature_encoding='default',
    viz_method='pca',
):
    # # Create sample Gaussian parameters
    # positions = torch.randn(n_gaussians, 3)  # xyz positions
    # scales = torch.abs(torch.randn(n_gaussians, 3)) + 0.5  # positive xyz scales
    # rotations = torch.randn(n_gaussians, 4)  # quaternion rotations
    rotations = rotations / rotations.norm(dim=1, keepdim=True)  # normalize quaternions
    # opacities = torch.sigmoid(torch.randn(n_gaussians, 1))  # opacity values in [0, 1]
    
    # Create the converter
    converter = GaussianFeatureMap(resolution=resolution, feature_dim=feature_dim)
    
    # Project Gaussians to 2D
    projected_params = converter.project_gaussians(positions, scales, rotations, opacities, camera_matrix=camera_matrix)
    
    # Create feature map
    feature_map = converter.create_feature_map(projected_params, feature_encoding=feature_encoding)
    
    # # Visualize feature map
    # fig = converter.visualize_feature_map(feature_map, method=viz_method)
    
    return {
        'projected_params': projected_params,
        'feature_map': feature_map.permute(2,0,1)
        # 'visualization': fig
    }

# Run with different settings
if __name__ == "__main__":
    # Default settings
    results = create_gaussian_feature_map(
        n_gaussians=100,
        resolution=(256, 256),
        feature_dim=32,
        feature_encoding='default',
        viz_method='pca'
    )
    plt.show()
    plt.savefig('feature_map.jpg')


def grid_based_features(positions, scales, rotations, opacities, camera_matrix, 
                       resolution=(270, 480), feature_dim=32, grid_size=16):
    """Much faster grid-based feature extraction"""
    H, W = resolution
    
    # Check shapes and fix if needed
    print(f"Positions shape: {positions.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Opacities shape: {opacities.shape}")
    
    # Ensure positions is [N, 3]
    if len(positions.shape) == 1:
        positions = positions.unsqueeze(0)  # Add batch dimension if missing
    
    # Project to 2D
    homogeneous = torch.cat([positions, torch.ones(positions.shape[0], 1, device=positions.device)], dim=1)
    clip_space = torch.matmul(homogeneous, camera_matrix.t())
    
    # Safe division
    w = clip_space[:, 2:3].clamp(min=1e-8)
    screen_pos = clip_space[:, :2] / w
    depths = positions[:, 2]
    
    # Scale to [0, grid_size-1]
    grid_x = ((screen_pos[:, 0] + 1) / 2 * grid_size).clamp(0, grid_size-1).long()
    grid_y = ((screen_pos[:, 1] + 1) / 2 * grid_size).clamp(0, grid_size-1).long()
    
    # Initialize grid features
    grid_features = torch.zeros(feature_dim, grid_size, grid_size, device=positions.device)
    grid_counts = torch.zeros(grid_size, grid_size, device=positions.device)
    
    # Accumulate features in grid cells
    for i in range(positions.shape[0]):
        x, y = grid_x[i], grid_y[i]
        
        # Create feature for this Gaussian
        feature = torch.zeros(feature_dim, device=positions.device)
        
        # Safely access tensor elements
        if feature_dim > 0:
            feature[0] = screen_pos[i, 0].item() if screen_pos.size(1) > 0 else 0.0
        if feature_dim > 1:
            feature[1] = screen_pos[i, 1].item() if screen_pos.size(1) > 1 else 0.0
        if feature_dim > 2:
            feature[2] = scales[i, 0].item() if scales.size(1) > 0 else 1.0
        if feature_dim > 3:
            feature[3] = scales[i, 1].item() if scales.size(1) > 1 else 1.0
        if feature_dim > 4:
            feature[4] = depths[i].item() if i < depths.shape[0] else 0.0
        if feature_dim > 5:
            feature[5] = opacities[i].item() if i < opacities.shape[0] else 1.0
        
        # Add to grid with opacity weighting
        weight = opacities[i].item() if i < opacities.shape[0] else 1.0
        grid_features[:, y, x] += feature * weight
        grid_counts[y, x] += weight
    
    # Normalize by counts
    valid_cells = grid_counts > 0
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_counts[i, j] > 0:
                grid_features[:, i, j] = grid_features[:, i, j] / grid_counts[i, j]
    
    # Upsample to full resolution using bilinear interpolation
    import torch.nn.functional as F
    feature_map = F.interpolate(grid_features.unsqueeze(0), size=(H, W), 
                               mode='bilinear', align_corners=False)[0]
    
    return feature_map