"""Vision Transformer (ViT) built from scratch in PyTorch.

Steps:
1. Split the image into patches and project them (patch embedding)
2. Prepend a [CLS] token and add positional embeddings
3. Pass the sequence through a stack of transformer encoder blocks
4. Use the [CLS] output for classification
"""
import math

import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]  # H, W
        self.patch_size = config["patch_size"]  # P
        self.num_channels = config["num_channels"]  # C
        self.hidden_size = config["hidden_size"]  # D
        self.num_patches = (self.image_size // self.patch_size) ** 2  # HW / P^2
        # kernel_size == stride == patch_size -> non-overlapping patches
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embedding(nn.Module):
    """Patch embeddings + [CLS] token + positional embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.rand(1, 1, config["hidden_size"]))
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        # (1, 1, D) -> (B, 1, D)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """A single self-attention head (Q, K, V all come from the same input)."""

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs


class MultiHeadAttention(nn.Module):
    """Concatenate several attention heads and project back to hidden_size."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    self.hidden_size,
                    self.attention_head_size,
                    config["attention_probs_dropout_prob"],
                    self.qkv_bias,
                )
                for _ in range(self.num_attention_heads)
            ]
        )
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([out for out, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return attention_output, None
        attention_probs = torch.stack([probs for _, probs in attention_outputs], dim=1)
        return attention_output, attention_probs


class GELUActivation(nn.Module):
    """GELU (tanh approximation)."""

    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: LN -> MHA -> residual, LN -> MLP -> residual."""

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return x, None
        return x, attention_probs


class Encoder(nn.Module):
    """Stack of transformer blocks."""

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return x, None
        return x, all_attentions


class ViT4Classification(nn.Module):
    """Full ViT model for image classification."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        # Classify using the [CLS] token only
        logits = self.classifier(encoder_output[:, 0])
        if not output_attentions:
            return logits, None
        return logits, all_attentions
