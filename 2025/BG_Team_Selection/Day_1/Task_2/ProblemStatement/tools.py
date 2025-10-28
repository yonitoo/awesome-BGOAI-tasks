import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm.auto import tqdm
from torch import optim

def contrastive_loss(
    pred:      torch.Tensor,
    en_emb:    torch.Tensor,
    mask:      torch.Tensor,
    temp:      float = 0.07,
    eps:       float = 1e-8
) -> torch.Tensor:
    """
    pred:   [B, T, D]  — output of BG2CLIPTransformerDecoder, raw embeddings
    en_emb: [B, T, D]  — target CLIP token embeddings
    mask:   [B, T]     — bool or 0/1 mask (1=real token, 0=pad)
    temp:   scalar     — temperature for scaling logits
    eps:    scalar     — epsilon for numeric stability

    Returns:
      scalar InfoNCE loss over all real tokens in the batch.
    """
    B, T, D = pred.shape
    assert en_emb.shape == (B, T, D), "en_emb must match pred shape"
    assert mask.shape == (B, T),         "mask must be [B, T]"

    # 1) L2-normalize both sets of embeddings (zeroing out pads first)
    m = mask.unsqueeze(-1).float()  # → [B, T, 1]
    pred_n = F.normalize(pred * m,   p=2, dim=-1, eps=eps)  # [B, T, D]
    en_n   = F.normalize(en_emb * m, p=2, dim=-1, eps=eps)  # [B, T, D]

    # 2) Flatten to [B*T, D], keep only real tokens
    mask_flat = mask.view(-1).bool()        # [B*T]
    if mask_flat.sum() == 0:
        # no valid tokens? return zero loss (or raise)
        return torch.tensor(0., device=pred.device, requires_grad=True)

    pred_flat = pred_n.view(B*T, D)[mask_flat]  # [N, D]
    en_flat   = en_n.view(B*T, D)[mask_flat]    # [N, D]

    # 3) Compute full pairwise cosine similarities → [N, N], scaled by 1/temp
    logits = torch.matmul(pred_flat, en_flat.t()) / temp  # [N, N]

    # 4) InfoNCE: each row i should match column i
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels) 

    return loss

def matrix_cosine_mse_loss(pred: torch.Tensor,
                           en_emb: torch.Tensor,
                           mask: torch.Tensor):
    """
    pred:   [B, T, D]   — model outputs
    en_emb: [B, T, D]   — target embeddings
    mask:   [B, T]      — 1 for real tokens, 0 for pad

    Returns:
        scalar MSE between pred vs. en_emb cosine‐similarity matrices,
        averaged over valid token‐pairs and over the batch.
    """
    # 1) normalize
    pred_n = F.normalize(pred, dim=-1)   # [B, T, D]
    en_n   = F.normalize(en_emb, dim=-1) # [B, T, D]

    # 2) pairwise cosine matrices
    #    sim_pred[b,i,j] = <pred_n[b,i], pred_n[b,j]>
    sim_pred = torch.matmul(pred_n, pred_n.transpose(1, 2))  # [B, T, T]
    sim_en   = torch.matmul(en_n,   en_n.transpose(1, 2))    # [B, T, T]

    # 3) build a [B, T, T] mask of valid token‐pairs
    #    only positions where both i and j are real tokens
    m = mask.bool()
    pair_mask = m.unsqueeze(1) & m.unsqueeze(2)              # [B, T, T]

    # 4) compute MSE over valid entries
    diff = sim_pred - sim_en
    # flatten only the valid differences
    valid_diffs = diff[pair_mask]                            # [num_valid_pairs]
    loss = valid_diffs.pow(2).mean()

    return loss

def validate(
    decoder,
    loader,
    bg_encoder,
    clip_text_encoder,
    device,
):
    """
    Validation loop computing only the mean-pooled cosine similarity between
    decoder predictions (from Bulgarian encoder outputs) and CLIP text encoder embeddings.

    Args:
        decoder:           Model mapping bg_encoder outputs to predictions.
        loader:            DataLoader yielding batches of dicts (bg_in, en_in).
        bg_encoder:        Background encoder (Bulgarian) producing hidden states.
        clip_text_encoder: Pretrained CLIP text encoder for target embeddings.
        device:            Torch device (e.g., 'cuda').

    Returns:
        avg_cosine_sim (float): Average cosine similarity over all samples.
    """
    # Switch to eval
    decoder.eval()

    total_cosine = 0.0
    steps = 0

    with torch.no_grad():
        for bg_in, en_in in loader:
            # Move inputs to device
            bg_in = {k: v.to(device) for k, v in bg_in.items()}
            en_in = {k: v.to(device) for k, v in en_in.items()}

            # Encode text embeddings via CLIP
            en_out = clip_text_encoder(**en_in)
            en_emb = en_out.last_hidden_state  # [B, T, D]

            # Encode Bulgarian inputs
            bg_out = bg_encoder(**bg_in).last_hidden_state  # [B, T, D]

            # Decoder prediction
            pred = decoder(bg_out, bg_in['attention_mask'])  # [B, T, D]
            mask = en_in['attention_mask'].unsqueeze(-1).bool().expand(-1, -1, pred.size(-1))
            en_emb_masked = en_emb[mask]
            mse = F.mse_loss(pred[:, :en_emb.shape[1], :][mask], en_emb_masked)
            cos_sim = matrix_cosine_mse_loss(pred[:, :en_emb.shape[1], :], en_emb, en_in["attention_mask"]) + mse
            total_cosine += cos_sim.item()
            steps += 1

    # Restore train mode
    decoder.train()
    return total_cosine / steps if steps > 0 else 0.0

def train(decoder, bg_encoder, clip_text_encoder, train_loader, val_loader, opt, EPOCHS, DEVICE):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    for epoch in range(EPOCHS):
        decoder.train()
        total_sim = 0.0
        total_mse = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, (bg_in, en_in) in enumerate(loop):
            bg_in = {k: v.to(DEVICE) for k, v in bg_in.items()}
            en_in = {k: v.to(DEVICE) for k, v in en_in.items()}
            with torch.no_grad():
                en_emb = clip_text_encoder(**en_in).last_hidden_state
                bg_out = bg_encoder(**bg_in).last_hidden_state
            r = random.randint(0, 1)
            if r == 0:
                drop_mask = torch.rand_like(bg_in["attention_mask"], dtype=torch.float) > 0.4
                bg_out = (bg_out * drop_mask.unsqueeze(-1))
                _, T, _ = bg_out.shape
                shift = torch.randint(0, T, (1,), device=DEVICE).item()
                # circularly roll along the token dimension
                bg_out = torch.roll(bg_out, shifts=shift, dims=1)
            pred = decoder(bg_out, bg_in['attention_mask'])

            mask = en_in['attention_mask'].unsqueeze(-1).bool().expand(-1, -1, pred.size(-1))
            en_emb_masked = en_emb[mask]
            loss_sim = matrix_cosine_mse_loss(pred[:, :en_emb.shape[1], :], en_emb, en_in["attention_mask"])
            loss = F.mse_loss(pred[:, :en_emb.shape[1], :][mask], en_emb_masked) 

            total_sim += loss_sim.item()
            total_mse += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            avg_sim = total_sim / (step + 1)
            avg_mse = total_mse / (step + 1)
            loop.set_postfix({'avg_sim': avg_sim, 'avg_mse': avg_mse})
        scheduler.step()
        if (epoch+1)%2==0:
            print(f"Validation: {validate(decoder, val_loader, bg_encoder, clip_text_encoder, DEVICE)}")
