import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DATA_DIR = 'data'
METADATA_FILE = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'HAM10000_images')
MODEL_SAVE_PATH = 'models/skin_lesion_model.pth'
LABELS_SAVE_PATH = 'models/class_labels.json'
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224

# Lesion Type Dictionary
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# ===== CUSTOM DATASET CLASS =====
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        
        # Create labels
        self.df['label'] = self.label_encoder.fit_transform(self.df['dx'])
        self.class_labels = list(self.label_encoder.classes_)
        
        # Verify all images exist
        self.valid_indices = []
        for idx in range(len(self.df)):
            img_path = self.df.iloc[idx]['path']
            if img_path and os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Image not found at {img_path}")
        
        print(f"✅ Found {len(self.valid_indices)} valid images out of {len(self.df)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_path = self.df.iloc[actual_idx]['path']
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.df.iloc[actual_idx]['label']
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return placeholder instead of crashing
            placeholder = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            return placeholder, -1  # Use -1 for invalid labels

# ===== MODEL CREATION =====
def create_model(num_classes):
    """Create and configure the MobileNetV2 model"""
    print("📱 Creating MobileNetV2 model...")
    
    # Load pre-trained model (using the most compatible method)
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except:
        # Fallback for different torch versions
        model = models.mobilenet_v2(pretrained=True)
    
    # Freeze early layers, fine-tune later ones
    for param in model.features[:10].parameters():
        param.requires_grad = False
        
    for param in model.features[10:].parameters():
        param.requires_grad = True
    
    # Modify classifier for our specific task
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    print(f"✅ Model created with {num_classes} output classes")
    return model

# ===== DATA PREPARATION =====
def prepare_data():
    """Load and prepare the dataset"""
    print("📊 Preparing data...")
    
    # Load metadata
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file not found at {METADATA_FILE}")
        print("Please make sure the HAM10000 dataset is in the '../data' folder")
        return None
    
    df = pd.read_csv(METADATA_FILE)
    print(f"✅ Loaded metadata with {len(df)} entries")
    
    # Add full disease names
    df['dx_full'] = df['dx'].map(lesion_type_dict)
    
    # Create image path mapping
    all_image_paths = {}
    if os.path.exists(IMAGE_DIR):
        for file in os.listdir(IMAGE_DIR):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_id = os.path.splitext(file)[0]
                all_image_paths[image_id] = os.path.join(IMAGE_DIR, file)
    else:
        # Try alternative directory structure
        print("⚠️  HAM10000_images directory not found, searching for images...")
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_id = os.path.splitext(file)[0]
                    all_image_paths[image_id] = os.path.join(root, file)
    
    df['path'] = df['image_id'].map(all_image_paths.get)
    
    # Remove rows with missing images
    initial_count = len(df)
    df = df.dropna(subset=['path'])
    df = df[df['path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
    
    print(f"✅ After filtering, using {len(df)} images out of {initial_count}")
    
    if len(df) == 0:
        print("❌ No valid images found!")
        print("Please check:")
        print(f"1. Dataset is downloaded to {DATA_DIR}")
        print(f"2. Images are in {IMAGE_DIR} or similar structure")
        print(f"3. File paths are correct")
        return None
    
    # Show class distribution
    print("\n📈 Class Distribution:")
    print(df['dx_full'].value_counts())
    
    return df

# ===== HANDLE CLASS IMBALANCE =====
def balance_data(df):
    """Balance the dataset using oversampling"""
    print("⚖️  Balancing data...")
    
    counts = df['dx'].value_counts()
    max_count = counts.max()
    df_balanced = pd.DataFrame()
    
    for class_index in counts.index:
        class_df = df[df['dx'] == class_index]
        if len(class_df) < max_count:
            # Oversample minority classes
            class_df_oversampled = class_df.sample(max_count, replace=True, random_state=42)
            df_balanced = pd.concat([df_balanced, class_df_oversampled])
            print(f"  ↪ Oversampled {class_index} from {len(class_df)} to {max_count}")
        else:
            df_balanced = pd.concat([df_balanced, class_df])
    
    print("✅ Data balanced successfully")
    return df_balanced

# ===== TRAINING FUNCTION ===
# ===== TRAINING FUNCTION =====
def train_model():
    """Main training function"""
    print("🚀 Starting Dermatology AI Model Training")
    print("=" * 50)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # Prepare data
    df = prepare_data()
    if df is None:
        return
    
    # Balance data
    df_balanced = balance_data(df)
    
    # Split data
    train_df, val_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_balanced['dx']
    )
    
    print(f"📁 Training samples: {len(train_df)}")
    print(f"📁 Validation samples: {len(val_df)}")
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_transform)
    
    # Get number of classes
    num_classes = len(train_dataset.class_labels)
    print(f"🎯 Number of classes: {num_classes}")
    
    # Save class labels
    os.makedirs('models', exist_ok=True)
    class_labels_map = {
        'idx_to_label': {i: label for i, label in enumerate(train_dataset.class_labels)},
        'label_to_idx': {label: i for i, label in enumerate(train_dataset.class_labels)},
        'label_names': lesion_type_dict
    }
    
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_labels_map, f, indent=2)
    print(f"💾 Class labels saved to {LABELS_SAVE_PATH}")
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training tracking
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    # Create data loaders with SINGLE WORKER (fixes Windows multiprocessing issue)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # CRITICAL: Use 0 instead of 2 for Windows
        pin_memory=False  # CRITICAL: False for Windows
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,  # CRITICAL: Use 0 instead of 2 for Windows
        pin_memory=False  # CRITICAL: False for Windows
    )
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            # Filter out invalid samples manually (since we removed collate_fn)
            valid_indices = labels != -1
            if valid_indices.sum() == 0:
                continue  # Skip batch if all samples are invalid
                
            inputs = inputs[valid_indices]
            labels = labels[valid_indices]
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_train / total_train:.2f}%'
                })
        
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
            for inputs, labels in val_pbar:
                # Filter out invalid samples manually
                valid_indices = labels != -1
                if valid_indices.sum() == 0:
                    continue
                    
                inputs = inputs[valid_indices]
                labels = labels[valid_indices]
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Acc': f'{100 * correct_val / total_val:.2f}%'
                })
        
        val_acc = 100 * correct_val / total_val
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 ✅ New best model saved! Validation accuracy: {val_acc:.2f}%")
    
    # Training completed
    print("=" * 60)
    print(f"🎉 Training completed!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Test the model
    test_trained_model(model, device, val_dataset, class_labels_map)
    
    return model
    """Main training function"""
    print("🚀 Starting Dermatology AI Model Training")
    print("=" * 50)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # Prepare data
    df = prepare_data()
    if df is None:
        return
    
    # Balance data
    df_balanced = balance_data(df)
    
    # Split data
    train_df, val_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_balanced['dx']
    )
    
    print(f"📁 Training samples: {len(train_df)}")
    print(f"📁 Validation samples: {len(val_df)}")
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_transform)
    
    # Get number of classes
    num_classes = len(train_dataset.class_labels)
    print(f"🎯 Number of classes: {num_classes}")
    
    # Create data loaders with error handling
    def collate_fn(batch):
        """Custom collate function to handle invalid samples"""
        batch = [(img, label) for img, label in batch if label != -1]
        if len(batch) == 0:
            # Return a minimal valid batch if all samples are invalid
            return torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), torch.tensor([0])
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # Save class labels
    os.makedirs('models', exist_ok=True)
    class_labels_map = {
        'idx_to_label': {i: label for i, label in enumerate(train_dataset.class_labels)},
        'label_to_idx': {label: i for i, label in enumerate(train_dataset.class_labels)},
        'label_names': lesion_type_dict
    }
    
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_labels_map, f, indent=2)
    print(f"💾 Class labels saved to {LABELS_SAVE_PATH}")
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training tracking
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_train / total_train:.2f}%'
                })
        
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Acc': f'{100 * correct_val / total_val:.2f}%'
                })
        
        val_acc = 100 * correct_val / total_val
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 ✅ New best model saved! Validation accuracy: {val_acc:.2f}%")
    
    # Training completed
    print("=" * 60)
    print(f"🎉 Training completed!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Test the model
    test_trained_model(model, device, val_dataset, class_labels_map)
    
    return model

# ===== VISUALIZATION =====
def plot_training_history(train_losses, val_accuracies):
    """Plot training loss and validation accuracy"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'g-', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===== MODEL TESTING =====
def test_trained_model(model, device, val_dataset, class_labels_map):
    """Test the trained model on sample images"""
    print("\n🧪 Testing model on sample images...")
    
    model.eval()
    
    # Test on 5 random samples
    test_indices = np.random.choice(len(val_dataset), min(5, len(val_dataset)), replace=False)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, idx in enumerate(test_indices):
        try:
            image, true_label = val_dataset[idx]
            if true_label == -1:
                continue
                
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_label = class_labels_map['idx_to_label'][str(predicted.item())]
            true_label_str = class_labels_map['idx_to_label'][str(true_label)]
            
            is_correct = (predicted_label == true_label_str)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"Sample {i+1}: {status}")
            print(f"  Predicted: {predicted_label} ({lesion_type_dict[predicted_label]})")
            print(f"  Actual: {true_label_str} ({lesion_type_dict[true_label_str]})")
            print(f"  Confidence: {confidence.item()*100:.2f}%")
            print()
            
        except Exception as e:
            print(f"Error testing sample {i+1}: {e}")
    
    if total_predictions > 0:
        test_accuracy = 100 * correct_predictions / total_predictions
        print(f"📊 Test Accuracy: {test_accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🧬 Dermatology AI Lesion Analyzer - Model Training")
    print("=" * 60)
    
    try:
        model = train_model()
        print("\n" + "=" * 60)
        print("🎊 All done! Your model is ready for use.")
        print("Next steps:")
        print("1. Run 'python main.py' to start the API server")
        print("2. Open frontend/index.html in your browser")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check if dataset is properly downloaded to '../data/'")
        print("2. Verify all required packages are installed")
        print("3. Check if you have sufficient disk space")
        print("4. Try reducing BATCH_SIZE if you have memory issues")
        import traceback
        traceback.print_exc()