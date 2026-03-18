# =========================================================
# FFA-Net: Feature Fusion Attention Network for Dehazing
# Fine-tuning on your custom dataset from ITS pretrained weights
# =========================================================
# SETUP (run in a Kaggle cell before this script):
#   !pip install gdown -q
#   !gdown --id 1-pgSXN6-NXLzmTp21L_qIg -O /kaggle/working/its_train_ffa_3_19.pk
#   OR manually upload its_train_ffa_3_19.pk as a Kaggle dataset
#   Pretrained weights (ITS indoor): Google Drive link from README:
#   https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5
# =========================================================

import os
import random
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim_sk

# =========================================================
# Paths  ✏️ edit
# =========================================================

TRAIN_DIR   = "/kaggle/input/datasets/ashwinvprasad/dehaze/train"
GT_DIR      = "/kaggle/input/datasets/ashwinvprasad/dehaze/gt"
SAVE_DIR    = "/kaggle/working/checkpoints_ffa"
os.makedirs(SAVE_DIR, exist_ok=True)

# Download pretrained ITS weights from Google Drive beforehand
# File: its_train_ffa_3_19.pk
PRETRAINED  = "/kaggle/input/ffa-pretrained/its_train_ffa_3_19.pk"  # ✏️ update path

# =========================================================
# Hyperparams
# =========================================================

GPS         = 3       # number of groups (paper uses 3)
BLOCKS      = 19      # blocks per group (paper uses 19)
PATCH_SIZE  = 256
REPEATS     = 50
BATCH_SIZE  = 4
EPOCHS      = 60
LR          = 5e-5    # lower LR for fine-tuning (was 1e-4 for training from scratch)
WARMUP      = 3

device    = "cuda"
multi_gpu = torch.cuda.device_count() > 1
print(f"GPUs: {torch.cuda.device_count()}")

# =========================================================
# FFA-Net Architecture (from paper + official repo)
# =========================================================

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, bias=bias)

class PALayer(nn.Module):
    """Pixel Attention Layer"""
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    """Basic Block = Local Residual + Feature Attention (CA + PA)"""
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Group(nn.Module):
    """Group of blocks with local residual"""
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFA(nn.Module):
    """
    Full FFA-Net:
    gps   = number of groups
    blocks = blocks per group
    """
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super().__init__()
        self.gps   = gps
        self.dim   = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3, "This implementation supports gps=3 only"

        self.g1 = Group(conv, self.dim, kernel_size, blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.pre  = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        # Attention-based feature fusion
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = (w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3)
        out = self.palayer(out)
        x = self.post(out)
        return x + x1   # global residual

# =========================================================
# Dataset  (same as your Restormer setup)
# =========================================================

class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, patch=256, repeats=50):
        self.patch   = patch
        self.repeats = repeats
        self.to_tensor = T.ToTensor()
        print("Loading dataset into RAM...")
        self.hazy_imgs, self.gt_imgs = [], []
        for name in sorted(os.listdir(hazy_dir)):
            h = self.to_tensor(Image.open(os.path.join(hazy_dir, name)).convert("RGB"))
            g = self.to_tensor(Image.open(os.path.join(gt_dir, name.replace("_NTHazy", "_GT"))).convert("RGB"))
            self.hazy_imgs.append(h)
            self.gt_imgs.append(g)
        print(f"Loaded {len(self.hazy_imgs)} pairs")

    def random_crop(self, h, g):
        _, H, W = h.shape
        x = random.randint(0, W - self.patch)
        y = random.randint(0, H - self.patch)
        return h[:, y:y+self.patch, x:x+self.patch], g[:, y:y+self.patch, x:x+self.patch]

    def augment(self, h, g):
        if random.random() > 0.5: h, g = torch.flip(h,[2]), torch.flip(g,[2])
        if random.random() > 0.5: h, g = torch.flip(h,[1]), torch.flip(g,[1])
        k = random.randint(0, 3)
        if k:
            h = torch.rot90(h, k, [1,2])
            g = torch.rot90(g, k, [1,2])
        return h, g

    def __len__(self): return len(self.hazy_imgs) * self.repeats

    def __getitem__(self, idx):
        i = idx % len(self.hazy_imgs)
        h, g = self.hazy_imgs[i].clone(), self.gt_imgs[i].clone()
        h, g = self.random_crop(h, g)
        h, g = self.augment(h, g)
        return h, g

dataset = DehazeDataset(TRAIN_DIR, GT_DIR, patch=PATCH_SIZE, repeats=REPEATS)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, pin_memory=True, persistent_workers=True)

# =========================================================
# Model
# =========================================================

model = FFA(gps=GPS, blocks=BLOCKS)

if multi_gpu:
    model = nn.DataParallel(model)

model = model.to(device)

# =========================================================
# Load pretrained weights (ITS indoor — closest to your data)
# =========================================================

if os.path.exists(PRETRAINED):
    print(f"Loading pretrained weights from {PRETRAINED}")
    ckpt = torch.load(PRETRAINED, map_location=device)

    # Handle DataParallel prefix if present
    raw = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]

    clean = {k.replace("module.", ""): v for k, v in raw.items()}

    # Load with strict=False to handle minor architecture diffs
    missing, unexpected = (model.module if multi_gpu else model).load_state_dict(clean, strict=False)
    print(f"Loaded pretrained ✓  |  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (will train from scratch): {missing[:5]}")
else:
    print(f"WARNING: Pretrained weights not found at {PRETRAINED}")
    print("Training from scratch — download its_train_ffa_3_19.pk from Google Drive")
    print("https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5")

# =========================================================
# Loss
# =========================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

criterion = CharbonnierLoss().to(device)

# =========================================================
# Optimizer + Scheduler
# =========================================================

optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP
    return 0.5 * (1 + np.cos((epoch - WARMUP) / (EPOCHS - WARMUP) * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = torch.amp.GradScaler("cuda")

# =========================================================
# Metrics
# =========================================================

def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(pred, gt):
    return ssim_sk(pred, gt, channel_axis=2, data_range=255)

def batch_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# =========================================================
# Validation on full images (not crops)
# =========================================================

val_files   = sorted(os.listdir(TRAIN_DIR))
to_tensor   = T.ToTensor()

def validate():
    m = model.module if multi_gpu else model
    m.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for name in val_files:
            hazy = to_tensor(Image.open(os.path.join(TRAIN_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
            gt   = to_tensor(Image.open(os.path.join(GT_DIR, name.replace("_NTHazy","_GT"))).convert("RGB")).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda"):
                pred = torch.clamp(m(hazy), 0, 1)
            pred_np = (pred[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            gt_np   = (gt[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            psnrs.append(compute_psnr(pred_np, gt_np))
            ssims.append(compute_ssim(pred_np, gt_np))
    return np.mean(psnrs), np.mean(ssims)

# =========================================================
# Visualization batch
# =========================================================

with torch.no_grad():
    vis_hazy, vis_gt = next(iter(loader))
    vis_hazy = vis_hazy.to(device)
    vis_gt   = vis_gt.to(device)

# =========================================================
# Training loop
# =========================================================

best_psnr = 0.0

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for hazy, gt in pbar:
        hazy = hazy.to(device, non_blocking=True)
        gt   = gt.to(device,   non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            pred = model(hazy)
            pred = torch.clamp(pred, 0, 1)
            loss = criterion(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_psnr += batch_psnr(pred.detach(), gt).item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step()

    avg_loss = running_loss / len(loader)
    avg_psnr = running_psnr / len(loader)

    # Full-image validation every 5 epochs
    if (epoch + 1) % 5 == 0:
        val_psnr, val_ssim = validate()
        print(f"\nEpoch {epoch+1} | Loss {avg_loss:.4f} | Train PSNR {avg_psnr:.2f}"
              f" | Val PSNR {val_psnr:.2f} | Val SSIM {val_ssim:.4f}"
              f" | LR {optimizer.param_groups[0]['lr']:.2e}")

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            torch.save((model.module if multi_gpu else model).state_dict(),
                       f"{SAVE_DIR}/ffa_best.pth")
            print(f"  ★ New best saved: {val_psnr:.2f} dB")
    else:
        print(f"\nEpoch {epoch+1} | Loss {avg_loss:.4f} | Train PSNR {avg_psnr:.2f}"
              f" | LR {optimizer.param_groups[0]['lr']:.2e}")

    # Save every epoch checkpoint as zip
    ckpt_path = f"{SAVE_DIR}/ffa_epoch_{epoch+1}.pth"
    zip_path  = f"{SAVE_DIR}/ffa_epoch_{epoch+1}.zip"
    torch.save((model.module if multi_gpu else model).state_dict(), ckpt_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ckpt_path, arcname=f"ffa_epoch_{epoch+1}.pth")
    print(f"  Saved: {zip_path}")

    # =====================================================
    # Visualization every 5 epochs: Hazy | GT | Pred
    # =====================================================

    if (epoch + 1) % 5 == 0:
        m = model.module if multi_gpu else model
        m.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                vis_pred = torch.clamp(m(vis_hazy), 0, 1)

        n = min(4, vis_hazy.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))

        for i in range(n):
            hazy_np = vis_hazy[i].permute(1,2,0).cpu().numpy()
            gt_np   = vis_gt[i].permute(1,2,0).cpu().numpy()
            pred_np = vis_pred[i].permute(1,2,0).cpu().numpy()

            p = compute_psnr((pred_np * 255).astype(np.uint8),
                             (gt_np   * 255).astype(np.uint8))
            s = compute_ssim( (pred_np * 255).astype(np.uint8),
                              (gt_np   * 255).astype(np.uint8))

            axes[i][0].imshow(hazy_np); axes[i][0].set_title("Hazy");         axes[i][0].axis("off")
            axes[i][1].imshow(gt_np);   axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
            axes[i][2].imshow(pred_np)
            axes[i][2].set_title(f"FFA Epoch {epoch+1}\nPSNR:{p:.2f} SSIM:{s:.4f}", fontsize=9)
            axes[i][2].axis("off")

        plt.suptitle(f"FFA-Net — Epoch {epoch+1}", fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close()

print(f"\n✅ Training complete. Best Val PSNR: {best_psnr:.2f} dB")
print(f"   Best model saved at: {SAVE_DIR}/ffa_best.pth")
