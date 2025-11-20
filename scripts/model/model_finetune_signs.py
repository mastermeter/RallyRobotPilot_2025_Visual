import torch
import torch.optim as optim
import torch.nn as nn
import sys, os

from torch.utils.data import DataLoader, random_split


PRETRAINED_MODEL_PATH = "scripts/model/output/robopilot_cnn_best.pth"
FILES_PATH = "records_signs/record_signs_*.npz"
SAVE_PATH = "scripts/model/output/robopilot_cnn_finetuned_signs.pth"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(PARENT_DIR, "model")
sys.path.append(MODEL_DIR)

from train_tools.old_LR_only.process import process_datas
from train_tools.old_LR_only.RoboDataset import RoboDataset
from train_tools.old_LR_only.RobopilotCNN import RobopilotCNN

BATCH_SIZE = 32 
EPOCHS = 25    
VAL_SPLIT = 0.2
DELTA_FRAMES = 1

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

LEARNING_RATE = 0.0001 
WEIGHT_DECAY = 1e-4
STEP_SIZE = 10
GAMMA = 0.95


def prepare_datas():
    features, labels = process_datas(FILES_PATH, DELTA_FRAMES)

    features_len = len(features)
    labels_len = len(labels)

    if features_len != labels_len:
        raise Exception("Features and Labels have not the same size")

    dataset = RoboDataset(features, labels)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Fine-tuning data loaded: {len(dataset)} samples.")
    return train_dataset, val_dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset = prepare_datas()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model = RobopilotCNN(output_size=2).to(device)
    
    try:
        ckpt = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Successfully loaded pre-trained model from {PRETRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Starting training from scratch instead.")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    
    print("--- Starting Fine-Tuning ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)           
            labels = labels.to(device)       

            optimizer.zero_grad()

            outputs = model(imgs)            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)


        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]


        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} "
            f"- Train Loss: {avg_train_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f} "
            f"- LR: {current_lr:.6f}"
        )

        if best_val_loss - avg_val_loss > EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
            print(f"   -> New best fine-tuned model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input": "RGB images 128x128 normalized /255",
            "task": "multi-label driving control (2 outputs, finetuned on signs)",
        },
        SAVE_PATH
    )
    print(f"Saved fine-tuned model to {SAVE_PATH}")

if __name__ == "__main__":
    train()