import torch
import torch.nn as nn
import utils
import torch.nn.functional as F

from utils import trunc_normal_

class ImprovedClsHead(nn.Module):
    '''
    Improved classification head with better regularization and architecture
    for handling class imbalance
    '''
    def __init__(self, embed_dim, num_classes, layers=3, dropout=0.3, use_batch_norm=True):
        super(ImprovedClsHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build classifier with better architecture
        channels = [self.embed_dim]
        for i in range(layers):
            channels.append(self.embed_dim // (2**i))
        channels.append(self.num_classes)
        
        layers_list = []
        for i in range(layers):
            # Linear layer
            layers_list.append(nn.Linear(channels[i], channels[i+1]))
            
            # Normalization layer
            if self.use_batch_norm:
                layers_list.append(nn.BatchNorm1d(channels[i+1]))
            else:
                layers_list.append(nn.LayerNorm(channels[i+1]))
            
            # Activation
            layers_list.append(nn.GELU())
            
            # Dropout for regularization
            layers_list.append(nn.Dropout(self.dropout_rate))
        
        self.classifier = nn.Sequential(*layers_list)
        
        # Additional regularization
        self.channel_bn = nn.BatchNorm1d(
            self.embed_dim,
            eps=1e-6,
            momentum=0.99,
        )
        
        # Label smoothing for better generalization
        self.label_smoothing = 0.1
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        
        # Apply batch normalization to input
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        
        # Forward through classifier
        logits = self.classifier(x)
        
        # Apply label smoothing during training
        if self.training:
            # Label smoothing is applied in the loss function, not here
            pass
        
        return logits
    
    def get_smoothed_labels(self, labels):
        """Apply label smoothing to prevent overconfidence"""
        n_classes = logits.size(-1)
        # Convert to one-hot
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1.0)
        # Apply smoothing
        smoothed = one_hot * (1 - self.label_smoothing) + (self.label_smoothing / n_classes)
        return smoothed


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss using weighted cross entropy and focal loss
    """
    def __init__(self, class_weights=None, alpha=1.0, gamma=2.0, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        # Weighted cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')
        
        # Focal loss
        focal_loss = self.focal_loss(inputs, targets)
        
        # Combine both losses
        total_loss = (1 - self.focal_weight) * ce_loss + self.focal_weight * focal_loss
        
        return total_loss


class AttentionPooling(nn.Module):
    """
    Attention pooling to better aggregate patch tokens
    """
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.Tanh(),
            nn.Linear(embed_dim // 8, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: [B, N, D] where N is number of patches
        weights = self.attention(x)  # [B, N, 1]
        weighted_features = torch.sum(x * weights, dim=1)  # [B, D]
        return weighted_features


class MultiScaleFeatures(nn.Module):
    """
    Multi-scale feature extraction for better representation
    """
    def __init__(self, embed_dim):
        super(MultiScaleFeatures, self).__init__()
        self.embed_dim = embed_dim
        
        # Different pooling strategies
        self.attention_pool = AttentionPooling(embed_dim)
        self.max_pool = nn.AdaptiveAvgPool1d((1,))  # Will be replaced by max
        self.avg_pool = nn.AdaptiveAvgPool1d((1,))
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, features):
        # features: [B, N, D] where N is number of features (patches)
        
        # Different pooling strategies
        att_feat = self.attention_pool(features)  # [B, D]
        max_feat, _ = torch.max(features, dim=1)  # [B, D]
        avg_feat = torch.mean(features, dim=1)  # [B, D]
        
        # Concatenate and fuse
        fused = torch.cat([att_feat, max_feat, avg_feat], dim=-1)  # [B, 3*D]
        output = self.fusion(fused)  # [B, D]
        
        return output