import sys, os
import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from torch.utils.data import DataLoader, random_split

from model.train_tools.process import process_datas
from model.train_tools.RoboDataset import RoboDataset
from model.train_tools.RobopilotCNNLSTM import RobopilotCNNLSTM

BATCH_SIZE = 64
EPOCHS = 20
VAL_SPLIT = 0.2
FILES_PATH = "record_signs_*.npz"

SEQ_LEN = 11
DELTA_FRAMES = 1

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
STEP_SIZE = 10
GAMMA = 0.95

BASE_MODEL_PATH = "scripts/model/output/robopilot_cnn_best_9k_11SEQ.pth"          
FINETUNED_MODEL_PATH = "scripts/model/output/robopilot_cnn_signs.pth"   

def prepare_datas():
    features, labels = process_datas(FILES_PATH)

    if len(features) == 0:
        raise RuntimeError(f"No data found with pattern {FILES_PATH}")

    if len(features) != len(labels):
        raise Exception("Features and Labels have not the same size")

    dataset = RoboDataset(
        features,
        labels,
        seq_len=SEQ_LEN,
        delta_frames=DELTA_FRAMES,
    )

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"[FT] Train size: {train_size}, Val size: {val_size}")
    return train_dataset, val_dataset


def train_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FT] Using device: {device}")

    train_dataset, val_dataset = prepare_datas()

    NUM_WORKERS = 0
    USE_PINNED_MEMORY = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_PINNED_MEMORY,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_PINNED_MEMORY,
        persistent_workers=False
    )

    model = RobopilotCNNLSTM(
        in_channels=3,
        seq_len=SEQ_LEN,
        cnn_feature_dim=256,
        lstm_hidden_dim=128,
        lstm_num_layers=1,
        output_size=3,    
        dropout_rate=0.3,
    ).to(device)

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    ckpt = torch.load(BASE_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[FT] Loaded base model from {BASE_MODEL_PATH}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for seq_imgs, labels in train_loader:
            seq_imgs = seq_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seq_imgs)  # (B,3) logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for seq_imgs, labels in val_loader:
                seq_imgs = seq_imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(seq_imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[FT] Epoch {epoch+1:03d}/{EPOCHS} "
            f"- Train Loss: {avg_train_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f} "
            f"- LR: {current_lr:.6f}"
        )

        if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
            print(f"[FT]   -> New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("[FT] Early stopping triggered.")
                break

    if best_state_dict is None:
        best_state_dict = model.state_dict()

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input": "RGB seq (T x 128x128) normalized /255",
            "task": "multi-label driving control (forward, left, right) -- fine-tuned on signs",
        },
        FINETUNED_MODEL_PATH,
    )
    print(f"[FT] Saved fine-tuned model to {FINETUNED_MODEL_PATH}")


if __name__ == "__main__":
    train_finetune()
