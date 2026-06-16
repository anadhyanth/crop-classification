# ── Core Imports ────────────────────────────────────────────────────────────
import os, random, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.utils import make_grid
import timm

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, top_k_accuracy_score)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG  — adjust paths and hyperparameters here
# ═══════════════════════════════════════════════════════════════════════════
class CFG:
    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_ROOT   = Path('/kaggle/input/datasets/snikhilrao/crop-disease-detection-dataset/Plant Village Dataset')
    TRAIN_DIR   = DATA_ROOT / 'Train'
    VAL_DIR     = DATA_ROOT / 'Val'
    TEST_DIR    = DATA_ROOT / 'Test'
    OUTPUT_DIR  = Path('/kaggle/working')

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_NAME  = 'tf_efficientnetv2_s'   # ~22M params, excellent accuracy/speed
    PRETRAINED  = True
    DROP_RATE   = 0.3
    DROP_PATH   = 0.2

    # ── Training ─────────────────────────────────────────────────────────────
    IMG_SIZE    = 224
    BATCH_SIZE  = 64          # P100 16GB can handle 64 easily
    NUM_EPOCHS  = 25
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    # ── Optimiser ────────────────────────────────────────────────────────────
    LR          = 3e-4
    MIN_LR      = 1e-6
    WEIGHT_DECAY= 1e-4
    WARMUP_EPOCHS = 3

    # ── Label Smoothing ──────────────────────────────────────────────────────
    LABEL_SMOOTH = 0.1

    # ── Mixed Precision ──────────────────────────────────────────────────────
    AMP         = True

    # ── Early Stopping ───────────────────────────────────────────────────────
    PATIENCE    = 7

    # ── Checkpoint ───────────────────────────────────────────────────────────
    CKPT_PATH   = OUTPUT_DIR / 'best_model.pth'

print('✅ Config loaded')
for k, v in vars(CFG).items():
    if not k.startswith('_'):
        print(f'   {k:<18} = {v}')
# ── Discover classes ─────────────────────────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in CFG.TRAIN_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX2CLASS   = {i: c for c, i in CLASS2IDX.items()}

print(f'📦 Total disease classes: {NUM_CLASSES}')
print('\nClasses:')
for i, c in enumerate(CLASS_NAMES):
    print(f'  [{i:2d}] {c}')
# ── Count images per class per split ────────────────────────────────────────
def count_images(root: Path):
    counts = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.JPG'))) \
              + len(list(cls_dir.glob('*.png'))) + len(list(cls_dir.glob('*.PNG')))
            counts[cls_dir.name] = n
    return counts

train_counts = count_images(CFG.TRAIN_DIR)
val_counts   = count_images(CFG.VAL_DIR)
test_counts  = count_images(CFG.TEST_DIR)

df_counts = pd.DataFrame({'Train': train_counts, 'Val': val_counts, 'Test': test_counts}).fillna(0).astype(int)
df_counts['Total'] = df_counts.sum(axis=1)
print(df_counts)
print(f'\n📊 Dataset totals: Train={df_counts.Train.sum():,}  Val={df_counts.Val.sum():,}  Test={df_counts.Test.sum():,}')
# ── Albumentations augmentation pipelines ────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32),
    A.RandomCrop(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        A.CLAHE(clip_limit=2.0, p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    min_holes=2, fill_value=0, p=0.4),  # CutOut
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

print('✅ Augmentation pipelines ready')
# ── Custom Dataset ───────────────────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for PlantVillage folder structure."""
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self, root: Path, class2idx: dict, transforms=None):
        self.transforms = transforms
        self.class2idx  = class2idx
        self.samples    = []   # (path, label)
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = class2idx.get(cls_dir.name)
            if label is None:
                continue
            for f in cls_dir.iterdir():
                if f.suffix in self.IMG_EXTS:
                    self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, label


# ── Build Datasets ────────────────────────────────────────────────────────────
train_ds = PlantDiseaseDataset(CFG.TRAIN_DIR, CLASS2IDX, train_transforms)
val_ds   = PlantDiseaseDataset(CFG.VAL_DIR,   CLASS2IDX, val_test_transforms)
test_ds  = PlantDiseaseDataset(CFG.TEST_DIR,  CLASS2IDX, val_test_transforms)

print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False,
                          num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)
test_loader  = DataLoader(test_ds,  batch_size=CFG.BATCH_SIZE, shuffle=False,
                          num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)

print(f'Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')
# ── EfficientNetV2-S + Custom Classification Head ────────────────────────────
class PlantDiseaseNet(nn.Module):
    """
    EfficientNetV2-S backbone with a multi-layer classification head.
    Head: GlobalAvgPool → BN → Dropout → FC(512) → GELU → BN → Dropout → FC(num_classes)
    """
    def __init__(self, num_classes: int, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2):
        super().__init__()
        # Load backbone via timm
        self.backbone = timm.create_model(
            CFG.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,          # Remove original classifier
            global_pool='avg',
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        in_features = self.backbone.num_features

        # Custom head
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes),
        )
        # Weight initialisation
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.xavier_uniform_(self.head[6].weight)

    def forward(self, x):
        feats  = self.backbone(x)   # (B, in_features)
        logits = self.head(feats)   # (B, num_classes)
        return logits

    def get_cam_target_layer(self):
        """Return last conv layer for Grad-CAM."""
        return self.backbone.blocks[-1][-1].conv_pwl


model = PlantDiseaseNet(NUM_CLASSES, pretrained=CFG.PRETRAINED,
                        drop_rate=CFG.DROP_RATE, drop_path_rate=CFG.DROP_PATH).to(DEVICE)

total_params   = sum(p.numel() for p in model.parameters())
trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'📐 Model: {CFG.MODEL_NAME}')
print(f'   Total params  : {total_params:,}')
print(f'   Trainable     : {trainable:,}')
print(f'   Classes       : {NUM_CLASSES}')
# ── Compute class weights to handle any imbalance ────────────────────────────
train_label_counts = np.array([train_counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float32)
class_weights = torch.tensor(
    (train_label_counts.sum() / (NUM_CLASSES * train_label_counts)),
    dtype=torch.float32
).to(DEVICE)

# ── Loss, Optimiser, Scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CFG.LABEL_SMOOTH)

optimizer = torch.optim.AdamW(
    [
        {'params': model.backbone.parameters(), 'lr': CFG.LR * 0.1},  # Lower LR for backbone
        {'params': model.head.parameters(),     'lr': CFG.LR},        # Higher LR for head
    ],
    weight_decay=CFG.WEIGHT_DECAY,
)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[CFG.LR * 0.1, CFG.LR],
    steps_per_epoch=len(train_loader),
    epochs=CFG.NUM_EPOCHS,
    pct_start=CFG.WARMUP_EPOCHS / CFG.NUM_EPOCHS,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)

# Mixed precision scaler
scaler = GradScaler(enabled=CFG.AMP)

print('✅ Loss, Optimiser, Scheduler ready')
# ── Train & Validation Step Functions ────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss, correct, total = 0., 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch:03d} [Train]', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{correct/total:.4f}')
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc='Evaluating', leave=False):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=CFG.AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


print('✅ Training functions defined')
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.
patience_ctr = 0
start_time   = time.time()

print('=' * 70)
print(f'  Starting training: {CFG.MODEL_NAME}  |  {NUM_CLASSES} classes  |  {CFG.NUM_EPOCHS} epochs')
print('=' * 70)

for epoch in range(1, CFG.NUM_EPOCHS + 1):
    ep_start = time.time()

    # ── Train ─────────────────────────────────────────────────────────────────
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, epoch)

    # ── Validate ──────────────────────────────────────────────────────────────
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

    # ── Log ───────────────────────────────────────────────────────────────────
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    ep_time = time.time() - ep_start
    lr_now  = optimizer.param_groups[-1]['lr']

    flag = ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ctr = 0
        torch.save({
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_acc':     val_acc,
            'class_names': CLASS_NAMES,
        }, CFG.CKPT_PATH)
        flag = '  ◀ BEST ✅'
    else:
        patience_ctr += 1

    print(f'Ep {epoch:03d}/{CFG.NUM_EPOCHS:03d} | '
          f'T-loss {train_loss:.4f} T-acc {train_acc:.4f} | '
          f'V-loss {val_loss:.4f} V-acc {val_acc:.4f} | '
          f'LR {lr_now:.2e} | {ep_time:.0f}s{flag}')

    if patience_ctr >= CFG.PATIENCE:
        print(f'\n⏹️  Early stopping triggered at epoch {epoch} (patience={CFG.PATIENCE})')
        break

total_time = time.time() - start_time
print(f'\n🏁 Training complete in {total_time/60:.1f} min')
print(f'   Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
with open(CFG.OUTPUT_DIR / 'history.json', 'w') as f:
    json.dump(history, f)
epochs_ran = len(history['train_loss'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
xs = range(1, epochs_ran + 1)

ax1.plot(xs, history['train_loss'], 'b-o', ms=4, label='Train Loss')
ax1.plot(xs, history['val_loss'],   'r-o', ms=4, label='Val Loss')
ax1.set_title('Loss Curve', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(xs, [a*100 for a in history['train_acc']], 'b-o', ms=4, label='Train Acc')
ax2.plot(xs, [a*100 for a in history['val_acc']],   'r-o', ms=4, label='Val Acc')
best_ep = np.argmax(history['val_acc']) + 1
ax2.axvline(best_ep, color='green', linestyle='--', label=f'Best Ep={best_ep}')
ax2.set_title('Accuracy Curve', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle(f'Training History — {CFG.MODEL_NAME}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(CFG.OUTPUT_DIR / 'training_curves.png', dpi=150)
plt.show()
# ── Load best checkpoint ──────────────────────────────────────────────────────
ckpt = torch.load(CFG.CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} (Val Acc={ckpt['val_acc']:.4f})")

# ── Evaluate on test set ──────────────────────────────────────────────────────
test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion)

# Top-5 accuracy
top5 = top_k_accuracy_score(test_labels, test_probs, k=5) if NUM_CLASSES >= 5 else None

print(f'\n📊 TEST RESULTS')
print(f'   Top-1 Accuracy : {test_acc*100:.2f}%')
if top5: print(f'   Top-5 Accuracy : {top5*100:.2f}%')
print(f'   Test Loss      : {test_loss:.4f}')
# ── Classification Report ─────────────────────────────────────────────────────
report = classification_report(
    test_labels, test_preds,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)
with open(CFG.OUTPUT_DIR / 'classification_report.txt', 'w') as f:
    f.write(report)
