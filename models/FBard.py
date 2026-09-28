import copy
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex
import math

from models.scale_block import ScaleBlock

from utils import to_per_pixel_logits_semantic, logit_to_fg_prob

class FBard(nn.Module):
    def __init__(
            self,
            encoder,
            device,

            # INP-Former parameters
            bottleneck,
            aggregation,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=True,  # Whether to drop 5 extra tokens for INP consistency loss (default True)
            encoder_require_grad_layer=[],  # Block indices that receive gradients; empty list freezes all
            prototype_token=None,

            # EoMT parameters
            num_classes=2,                      # Number of segmentation classes (foreground and background)
            num_q=2,                            # Number of queries
            num_blocks=3,                       # Number of blocks using masked attention
            masked_attn_enabled=True,           # Enable masked attention
            attn_mask_annealing_enabled=True,   # Enable mask annealing schedule
            mask_annealing_poly_power=0.9,      # Polynomial power for mask annealing curve
            ignore_idx=255,                     # Label index to ignore

            # Additional parameters
            ema_decay = 0.99,                   # EMA decay for background prototype updates
            update_bg_prototypes=False,         # Enable background feature prototype updates
            use_segmentation_branch=True,       # Enable segmentation branch
            use_bg_feature_aggregation=False,   # Enable background feature aggregation (False in train, True in test)
            bg_feature_aggregation_binary=False,  # True: binary mask aggregation; False: probability-weighted aggregation
            fg_map_dilate_ksize = 5,            # Foreground map dilation kernel size (0 disables dilation)
            post_fusion: bool = False           # If True, skip bg_prob_map-based encoder feature processing

    ) -> None:
        super(FBard, self).__init__()  # Call nn.Module constructor
        self.encoder = encoder

        # INP-Former attributes
        self.bottleneck = bottleneck
        self.aggregation = aggregation  # INP Extractor (a standard ViT block)
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        # EoMT attributes
        self.num_classes = num_classes                                      # Segmentation classes (foreground and background)
        self.num_q = num_q                                                  # Number of queries
        self.num_blocks = num_blocks                                        # Blocks using masked attention
        self.masked_attn_enabled = masked_attn_enabled                      # bool: enable masked attention
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled      # bool: enable mask annealing
        self.poly_power = mask_annealing_poly_power                         # Mask annealing polynomial power
        self.device = device

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))  # Mask enable probability per block
        self.q = nn.Embedding(num_q, self.encoder.embed_dim)  # Query(2, 768)  # Query(2, 768)
        self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)   # Class head (768, 2+1)  # FIXME: should output dim be 2?
        self.mask_head = nn.Sequential(                                        # Mask head, three linear layers (768, 768)
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        )
        patch_size = encoder.patch_size                       # patch_size=14
        num_upscale = max(1, int(math.log2(patch_size)) - 1)  # Number of 2x upsamples; patch_size=14 -> 2 upsamples
        self.upscale = nn.Sequential(                         # Upsampling module
            *[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],
        )
        self.init_metrics_semantic(ignore_idx, self.num_blocks + 1 if self.masked_attn_enabled else 1)  # Initialize evaluation metrics

        # Additional attributes
        self.seg_blocks = nn.ModuleList(                       # Deep-copy last num_blocks encoder blocks for segmentation
            [copy.deepcopy(blk) for blk in self.encoder.blocks[-self.num_blocks:]]
        )

        self.ema_decay = ema_decay   # EMA decay coefficient
        self.update_bg_prototypes = update_bg_prototypes
        self.use_segmentation_branch = use_segmentation_branch  # Enable segmentation branch
        self.use_bg_feature_aggregation = use_bg_feature_aggregation  # Enable background feature aggregation
        self.bg_feature_aggregation_binary = bg_feature_aggregation_binary
        self.fg_map_dilate_ksize = fg_map_dilate_ksize  # Foreground map grayscale dilation kernel size
        self.post_fusion = post_fusion
        self.segmentation_requires_grad = True  # Can disable seg-branch backward pass in anomaly detection to save memory

        if use_segmentation_branch:
            prototype_token_num = len(self.target_layers)
            self.register_buffer('background_prototypes',  # Background prototypes (no grad; updated via EMA)
                                 torch.randn(prototype_token_num, 1, self.encoder.embed_dim))

    # Average all feature maps in feat_list
    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)  # 8 or 4 x (1, 37*37, 768) -> (1, 37*37, 768)



    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)  # Cosine distance = 1 - cosine similarity
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)  # Min cosine distance from each query to keys
        gather_loss = self.distance.mean()  # Mean of nearest distances
        return gather_loss


    # Initialize evaluation metrics
    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )


    # Predict segmentation masks
    def predict_mask(self, x: torch.Tensor, feat_hw):  # x(B, token_nums, dim)=(16, 1369+1+4+2, 768)
        q = x[:, : self.num_q, :]  # x(B, q_nums, dim): query tokens only (queries precede cls token)

        class_logits = self.class_head(q)  # Queries through class head (linear) -> class logits

        # (16, 1376, 768)->(16, 1369, 768)   # INP-Former input here is (16, 789, 768)
        x = x[:, self.num_q + self.encoder.num_prefix_tokens:, :]  # num_prefix_tokens = prefix count; x(B, patch_token_nums, dim): all patch tokens
        x = x.transpose(1, 2).reshape(  # (B, num_patch_tokens, dim) -> (B, dim, num_patch_tokens)
            x.shape[0], -1, feat_hw, feat_hw)  # -> (B, dim, H_patches, W_patches)=(16, 768, 37, 37))

        # Upsample features; MLP(q) dotted with each feature vector -> mask logits
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits


    # Build attention mask
    def _attn_mask(self, x, mask_logits, i, feat_hw):
        attn_mask = torch.ones(  # Build attention mask (B, num_tokens, num_tokens)=(16, 1369+4+1+2, 1369+4+1+2)
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(  # Downsample mask_logits (16, 2, 37*4, 37*4) to feature size (16, 2, 37, 37)
            mask_logits,
            size=(feat_hw, feat_hw),
            mode="bilinear",
        )
        interpolated = interpolated.view(  # (16, 2, 37, 37) -> (16, 2, 1369)
            interpolated.size(0), interpolated.size(1), -1)
        attn_mask[  # attn_mask=(B, 1376, 1376): Q->patch region uses predicted mask; set positions where prediction > 0
        :,
        : self.num_q,
        self.num_q + self.encoder.num_prefix_tokens:,
        ] = (
                interpolated > 0
        )
        attn_mask = self._disable_attn_mask(  # Randomly disable part of mask during training; attn_mask_probs set in training loop
            attn_mask,
            self.attn_mask_probs[i - (len(self.encoder.blocks) - self.num_blocks)]
        )
        return attn_mask


    # Mask annealing schedule; returns per-layer mask probability
    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)


    # Masked attention
    def mask_attn(self, module, x, mask, rope):  # module: blk.attn
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)  # qkv(): nn.Linear(dim, dim * 3, bias=qkv_bias) (B, N, C)->(B, N, 3C)->(B, N, 3C, HN, HD)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, N, 3C, HN, HD) -> (3, B, HN, N, HD) -> three (B, H, HN, HD)
        q, k = module.q_norm(q), module.k_norm(k)  # Normalize q and k

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)  # (B, N, N) -> (B, HN, N, N)

        dropout_p = module.attn_drop.p if self.training else 0.0  # Dropout probability

        x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)  # Masked attention

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))  # Linear proj; output x=(B, N, C)

        return x


    # Block name suffix, returns _block_xxx
    def block_postfix(self, block_idx):
        if not self.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.num_blocks
            else ""
        )


    def forward(self, x):

        # ===== Initialization
        # ---------- Initial setup
        x = self.encoder.prepare_tokens(x)  # Patchify, insert cls/reg tokens, add pos embed x=(B, 1369+1+4, 768)
        B, L, _ = x.shape  # B=batch_size, L=num_tokens(1+4+(518/14)^2)
        feat_hw = int((L - self.encoder.num_prefix_tokens) ** 0.5)  # Feature map side length
        seg_grad_enabled = self.training and self.segmentation_requires_grad  # Disable seg-branch grads in anomaly detection to save memory

        # ---------- INP-Former initialization
        en_list = []  # Per-layer features; each entry is one target layer output

        # ---------- EoMT initialization
        attn_mask = None  # Mask for masked attention
        mask_logits_per_layer, class_logits_per_layer = [], []  # Per-layer intermediate predictions
        rope = self.encoder.rope_embeddings(x) if hasattr(self.encoder, "rope_embeddings") else None  # Rotary position embeddings

        # ===== Encoding
        for i, blk in enumerate(self.encoder.blocks):

            # ---------- Entering segmentation branch
            if i == len(self.encoder.blocks) - self.num_blocks:
                x_seg = torch.cat((self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x.clone()),
                                  dim=1)  # Clone features for seg branch; prepend segmentation queries

            # ---------- Shared-feature forward pass in early blocks
            if i < len(self.encoder.blocks) - self.num_blocks or i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:  # Use gradients for this block
                    x = blk(x)  # Forward pass
                else:
                    with torch.no_grad():
                        x = blk(x)

                if i in self.target_layers:  # Save current layer output to feature list
                    en_list.append(x)  # x=(B, N, C)

            # ---------- Segmentation branch
            if self.use_segmentation_branch:
                if i >= len(self.encoder.blocks) - self.num_blocks:
                    seg_blk = self.seg_blocks[i - (len(self.encoder.blocks) - self.num_blocks)]
                    with torch.set_grad_enabled(seg_grad_enabled):
                        if self.masked_attn_enabled:  # Build attention mask when masked attention is enabled
                            mask_logits, class_logits = self.predict_mask(self.encoder.norm(x_seg), feat_hw)  # Predict intermediate masks
                            mask_logits_per_layer.append(mask_logits)  # [(16, 2, 37*4, 37*4)]  # Record intermediate result
                            class_logits_per_layer.append(class_logits)  # [(16, 2, 3)]

                            attn_mask = self._attn_mask(x_seg, mask_logits, i, feat_hw)  # Build attention mask

                        attn = seg_blk.attn if hasattr(seg_blk, "attn") else seg_blk.attention  # Forward pass
                        attn_out = self.mask_attn(attn, seg_blk.norm1(x_seg), attn_mask, rope=rope)
                        if hasattr(seg_blk, "ls1"):
                            x_seg = x_seg + seg_blk.ls1(attn_out)
                        elif hasattr(seg_blk, "layer_scale1"):
                            x_seg = x_seg + seg_blk.layer_scale1(attn_out)

                        mlp_out = seg_blk.mlp(seg_blk.norm2(x_seg))
                        if hasattr(seg_blk, "ls2"):
                            x_seg = x_seg + seg_blk.ls2(mlp_out)
                        elif hasattr(seg_blk, "layer_scale2"):
                            x_seg = x_seg + seg_blk.layer_scale2(mlp_out)

        # ===== Encoding complete
        if self.use_segmentation_branch:
            with torch.set_grad_enabled(seg_grad_enabled):
                mask_logits, class_logits = self.predict_mask(self.encoder.norm(x_seg), feat_hw)  # Final EoMT segmentation prediction
            mask_logits_per_layer.append(mask_logits)  # (B, 2, 148, 148)
            class_logits_per_layer.append(class_logits)  # (B, 2, 2); (B, i, j) = prob query i matches class j

            # --------- Compute background probability map from segmentation
            logit = to_per_pixel_logits_semantic(mask_logits, class_logits)  # logit=(B, 2, 148, 148): per-pixel class probabilities
            fg_prob_map = logit_to_fg_prob(logit)  # fg_prob_map=(B, 148, 148): foreground probability per pixel

            # ---------- Grayscale morphological dilation on foreground map for small-object robustness
            if self.fg_map_dilate_ksize > 0:
                yy, xx = torch.meshgrid(torch.arange(self.fg_map_dilate_ksize), torch.arange(self.fg_map_dilate_ksize), indexing='ij')
                kernel = ((xx - 4) ** 2 + (yy - 4) ** 2 <= 4 ** 2).float().to(fg_prob_map.device)
                # Grayscale morphological dilation
                fg_prob_map = kornia.morphology.dilation(fg_prob_map.unsqueeze(1), kernel).squeeze(1)

            bg_prob_map = 1.0 - fg_prob_map  # bg_prob_map=(B, 148, 148): background probability per pixel
            bg_prob_map_featsize = F.interpolate(
                bg_prob_map.unsqueeze(1),  # interpolate expects (B, C, h, w)
                size=(feat_hw, feat_hw),
                mode='bilinear',
                align_corners=False).squeeze(1)  # Downscale background prob map to feature size (B, 37, 37)
            bg_prob_flat = bg_prob_map_featsize.view(x.shape[0], -1)  # Flatten to (B, 37*37)

        if self.remove_class_token:  # Drop 5 extra tokens for INP consistency loss (default True)
            en_list = [e[:, self.encoder.num_prefix_tokens:, :] for e in en_list]


        # ===== Update background feature prototypes (training only)
        if (not self.post_fusion) and self.use_segmentation_branch and self.update_bg_prototypes:

            for i, feat in enumerate(en_list):  # EMA update over layers 2~9

                batch_bg_feat_avg = cal_batch_bg_avg_feat(bg_prob_flat, feat)  # Batch mean background feature (C,)

                if batch_bg_feat_avg is not None:
                    bg_prototype = self.background_prototypes[i]  # Background prototype for this layer (1, C)
                    updated_proto = bg_prototype * self.ema_decay + batch_bg_feat_avg * (1 - self.ema_decay)  # EMA update
                    self.background_prototypes[i].copy_(updated_proto.detach())  # Write EMA update to buffer; detach clears grad
                else:  # Skip update if batch has too few background pixels
                    break


        # ===== Background feature aggregation
        if (not self.post_fusion) and self.use_segmentation_branch and self.use_bg_feature_aggregation:
            bg_feature_aggregated_en_list = []  # Features after background aggregation (layers 2~9)
            fg_binary_flat = None
            if self.bg_feature_aggregation_binary:
                # Downscale foreground prob map to patch grid; threshold at 0.5 (white=keep features, black=replace with prototype)
                fg_prob_map_featsize = F.interpolate(
                    fg_prob_map.unsqueeze(1),
                    size=(feat_hw, feat_hw),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
                fg_binary_flat = (fg_prob_map_featsize >= 0.5).to(en_list[0].dtype).reshape(x.shape[0], -1)
            for i, feat in enumerate(en_list):  # Iterate layers 2~9
                bg_proto = self.background_prototypes[i]  # Background prototype for this layer (1, C)

                if self.bg_feature_aggregation_binary:
                    adjusted_feat = torch.einsum('bnc,bn->bnc', feat, fg_binary_flat) + \
                                    torch.einsum('bc,bn->bnc', bg_proto, 1.0 - fg_binary_flat)
                else:
                    # Weight original features and background prototype via einsum; feat=(B, N, C), bg_prob_flat=(B, N), bg_proto=(B, C)
                    adjusted_feat = torch.einsum('bnc,bn->bnc', feat, 1 - bg_prob_flat) + \
                                    torch.einsum('bc,bn->bnc', bg_proto, bg_prob_flat)
                bg_feature_aggregated_en_list.append(adjusted_feat)  # (B, N, C)
            en_list = bg_feature_aggregated_en_list  # Replace with background-aggregated features


        x = self.fuse_feature(en_list)  # Average all feature maps in en_list (layers 2~9)

        agg_prototype = self.prototype_token  # Learnable prototype tokens
        for i, blk in enumerate(self.aggregation):  # Aggregation blocks
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(query=x, keys=agg_prototype)  # INP consistency loss

        # ===== INP-Former decoding
        # ---------- bottleneck
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # ---------- decoder
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        # ---------- Fuse encoder and decoder features
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]  # [mean of layers 3~6, mean of layers 7~10]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]  # [mean of layers 3~6, mean of layers 7~10]

        if not self.remove_class_token:  # Remove 5 extra tokens if not already removed for INP loss
            en = [e[:, self.encoder.num_prefix_tokens:, :] for e in en]
            de = [d[:, self.encoder.num_prefix_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, feat_hw, feat_hw]).contiguous() for e in en]  # (1, 1369, 768) -> (1, 768, 37, 37)
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, feat_hw, feat_hw]).contiguous() for d in de]

        if self.use_segmentation_branch:
            return en, de, g_loss, mask_logits_per_layer, class_logits_per_layer

        else:
            return en, de, g_loss

    def set_segmentation_requires_grad(self, enabled: bool):
        """Enable/disable backward pass through the segmentation branch to save memory during anomaly detection."""
        self.segmentation_requires_grad = enabled


'''
Compute per-batch mean background feature across all images.

Args:
    bg_prob_flat: Per-element background probability for the batch (B, N), N = feat_hw ** 2
    feat: Feature sequence for the batch (B, N, C)
    min_prob_sum: Minimum tolerated sum of probabilities (fraction; ~0.1% background pixels per image on average)
'''

def cal_batch_bg_avg_feat(bg_prob_flat, feat, min_prob_sum=0.001):
    B, N = feat.shape[0], feat.shape[1]  # batch_size, token sequence length
    sum_of_prob = torch.sum(bg_prob_flat)  # Sum of background probs (denominator) (B, N) -> ()
    if sum_of_prob < B * N * min_prob_sum:  # Too few background pixels
        return None
    else:
        weighted_sum = torch.sum(bg_prob_flat.unsqueeze(-1) * feat, dim=(0, 1))  # Weighted sum of features (numerator): (B, N, 1) * (B, N, C) -> (C,)
        bg_feat_avg = weighted_sum / sum_of_prob  # Weighted average: (C,) / scalar -> (C,)
        return bg_feat_avg.unsqueeze(0)  # (1, C)
