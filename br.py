# -*- coding: utf-8 -*-
"""
Plant Disease Detection - Crash-Resistant Micro-Chunk Version
Optimized for HP 840 G3 (16GB RAM, no GPU)
Includes: Auto-resume, micro-batching, memory management
"""

import os
import sys
import json
import gc
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves memory)
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2  # Lighter than EfficientNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION FOR LOW-END HARDWARE
# ============================================================================

BASE_PATH = Path.cwd()
CHECKPOINT_DIR = BASE_PATH / 'checkpoints'
LOG_DIR = BASE_PATH / 'logs'
PROGRESS_FILE = LOG_DIR / 'training_progress.json'

for directory in [CHECKPOINT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# CRITICAL SETTINGS FOR HP 840 G3
IMG_SIZE = 108  # Reduced from 224 to save memory
BATCH_SIZE = 4   # Small batch for CPU training
MICRO_CHUNK_SIZE = 200  # Process 500 images at a time
TOTAL_EPOCHS = 20
SAMPLES_PER_CLASS = 60  # Limit training samples (remove this line to use full dataset)

print("=" * 70)
print("🖥️  PLANT DISEASE DETECTION - LAPTOP OPTIMIZED")
print("=" * 70)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (reduced for speed)")
print(f"Batch Size: {BATCH_SIZE} (optimized for CPU)")
print(f"Micro-Chunks: {MICRO_CHUNK_SIZE} images per checkpoint")
print(f"Memory Limit: 16GB RAM (no GPU)")
print("=" * 70)

# ============================================================================
# SYSTEM CHECK
# ============================================================================

print("\n📊 System Information:")
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {len(gpus)}")
else:
    print("⚠️  No GPU - Using CPU (expect slower training)")
    # Optimize CPU usage
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track training progress for crash recovery"""
    
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress or create new"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.current_epoch = data.get('current_epoch', 0)
                self.samples_processed = data.get('samples_processed', 0)
                self.best_val_acc = data.get('best_val_acc', 0.0)
                self.history = data.get('history', [])
                print(f"\n✅ Resuming from Epoch {self.current_epoch}")
                print(f"   Samples processed: {self.samples_processed}")
                print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
        else:
            self.current_epoch = 0
            self.samples_processed = 0
            self.best_val_acc = 0.0
            self.history = []
            print("\n🆕 Starting fresh training")
    
    def save_progress(self, epoch, samples, val_acc, epoch_metrics):
        """Save current progress"""
        self.current_epoch = epoch
        self.samples_processed = samples
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        
        self.history.append(epoch_metrics)
        
        with open(self.progress_file, 'w') as f:
            json.dump({
                'current_epoch': self.current_epoch,
                'samples_processed': self.samples_processed,
                'best_val_acc': self.best_val_acc,
                'history': self.history,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
    
    def should_continue(self):
        """Check if training should continue"""
        return self.current_epoch < TOTAL_EPOCHS

# ============================================================================
# DATASET HANDLING
# ============================================================================

# ...existing code...
def download_or_locate_dataset():
    """Locate dataset at known path or in project. No automatic download."""
    print("\n📥 Locating dataset...")
    
    # Prefer the user-provided path
    preferred = Path(r"G:\bik\New Plant Diseases Dataset(Augmented)")
    if preferred.exists():
        print(f"✅ Using dataset at: {preferred}")
        return preferred
    
    # Fallback local checks (project folders)
    local_paths = [
        BASE_PATH / 'dataset',
        BASE_PATH / 'plant-disease-dataset',
        BASE_PATH / 'PlantVillage'
    ]
    
    for path in local_paths:
        if path.exists():
            print(f"✅ Found local dataset: {path}")
            return path
    
    # If nothing is found, instruct user (no auto-download)
    print("\n❌ Dataset not found.")
    print("Please place the dataset at one of the following locations:")
    print(f" 1) {preferred}")
    print(f" 2) {BASE_PATH / 'dataset'}")
    print(f" 3) {BASE_PATH / 'plant-disease-dataset'}")
    print(f" 4) {BASE_PATH / 'PlantVillage'}")
    print("\nThen run this script again.")
    sys.exit(1)
# ...existing code...

def setup_data_directories(dataset_path):
    """Find train/val directories"""
    print("\n🔍 Setting up data directories...")
    
    train_dir = None
    val_dir = None
    
    for item in Path(dataset_path).rglob('*'):
        if item.is_dir():
            name = item.name.lower()
            if 'train' in name and train_dir is None:
                train_dir = item
            elif ('valid' in name or 'val' in name) and val_dir is None:
                val_dir = item
    
    if not train_dir or not val_dir:
        print("❌ Could not find train/validation folders")
        sys.exit(1)
    
    print(f"✅ Train: {train_dir}")
    print(f"✅ Valid: {val_dir}")
    
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"✅ Classes: {len(class_names)}")
    
    return train_dir, val_dir, class_names

# ============================================================================
# MEMORY-EFFICIENT DATA GENERATOR
# ============================================================================

def create_limited_generators(train_dir, val_dir, samples_per_class=None):
    """Create generators with optional sample limiting"""
    print(f"\n🔄 Creating data generators (limit: {samples_per_class} per class)...")
    
    # If limiting samples, create subset
    if samples_per_class:
        limited_train_dir = BASE_PATH / 'limited_train'
        limited_train_dir.mkdir(exist_ok=True)
        
        total_copied = 0
        for class_folder in train_dir.iterdir():
            if class_folder.is_dir():
                # Create class folder
                new_class_dir = limited_train_dir / class_folder.name
                new_class_dir.mkdir(exist_ok=True)
                
                # Copy limited samples
                images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.JPG'))
                selected = images[:samples_per_class]
                
                for img in selected:
                    import shutil
                    dest = new_class_dir / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        total_copied += 1
        
        print(f"✅ Created subset with {total_copied} images")
        train_dir = limited_train_dir
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Train samples: {train_gen.samples}")
    print(f"✅ Val samples: {val_gen.samples}")
    
    return train_gen, val_gen

# ============================================================================
# LIGHTWEIGHT MODEL
# ============================================================================

def build_lightweight_model(num_classes):
    """Build MobileNetV2 model (lighter than EfficientNet)"""
    print("\n🏗️  Building lightweight model...")
    
    base = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model ready!")
    print(f"   Total params: {model.count_params():,}")
    
    return model

# ============================================================================
# MICRO-CHUNK TRAINING WITH AUTO-SAVE
# ============================================================================

class MicroChunkCallback(Callback):
    """Save checkpoint every N samples"""
    
    def __init__(self, tracker, checkpoint_dir, chunk_size):
        super().__init__()
        self.tracker = tracker
        self.checkpoint_dir = checkpoint_dir
        self.chunk_size = chunk_size
        self.samples_in_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.samples_in_epoch = 0
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{TOTAL_EPOCHS}")
        print(f"{'='*70}")
    
    def on_batch_end(self, batch, logs=None):
        batch_size = logs.get('size', BATCH_SIZE)
        self.samples_in_epoch += batch_size
        self.tracker.samples_processed += batch_size
        
        # Save every chunk
        if self.samples_in_epoch % self.chunk_size == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{self.tracker.current_epoch}_samples{self.tracker.samples_processed}.keras'
            self.model.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved: {self.tracker.samples_processed} samples processed")
            
            # Clear memory
            gc.collect()
    
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        
        # Save progress
        self.tracker.save_progress(
            epoch=epoch + 1,
            samples=self.tracker.samples_processed,
            val_acc=val_acc,
            epoch_metrics={
                'epoch': epoch + 1,
                'loss': float(logs.get('loss', 0)),
                'accuracy': float(logs.get('accuracy', 0)),
                'val_loss': float(logs.get('val_loss', 0)),
                'val_accuracy': float(val_acc),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Save best model
        if val_acc > self.tracker.best_val_acc:
            best_path = self.checkpoint_dir / 'best_model.keras'
            self.model.save(best_path)
            print(f"⭐ New best model saved! Val Acc: {val_acc:.4f}")
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Loss: {logs.get('loss', 0):.4f} | Acc: {logs.get('accuracy', 0):.4f}")
        print(f"   Val Loss: {logs.get('val_loss', 0):.4f} | Val Acc: {val_acc:.4f}")
        print(f"   Best Val Acc: {self.tracker.best_val_acc:.4f}")
        
        # Memory cleanup
        gc.collect()
        tf.keras.backend.clear_session()

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_resume(model, train_gen, val_gen, tracker):
    """Train with automatic resume capability"""
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"Starting from epoch: {tracker.current_epoch}")
    print(f"Target epochs: {TOTAL_EPOCHS}")
    print(f"Samples processed: {tracker.samples_processed}")
    print("="*70)
    
    # Load checkpoint if resuming
    if tracker.current_epoch > 0:
        checkpoint_files = sorted(CHECKPOINT_DIR.glob('checkpoint_*.keras'))
        if checkpoint_files:
            latest = checkpoint_files[-1]
            print(f"\n📂 Loading checkpoint: {latest.name}")
            model = keras.models.load_model(latest)
            print("✅ Model loaded successfully")
    
    # Callbacks
    chunk_callback = MicroChunkCallback(tracker, CHECKPOINT_DIR, MICRO_CHUNK_SIZE)
    
    # Calculate epochs remaining
    initial_epoch = tracker.current_epoch
    epochs_remaining = TOTAL_EPOCHS - initial_epoch
    
    if epochs_remaining <= 0:
        print("\n✅ Training already complete!")
        return model
    
    # Train
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=TOTAL_EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=[chunk_callback],
            verbose=1
        )
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Progress has been saved. Run again to resume.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Progress has been saved. Fix the issue and run again to resume.")
    
    return model

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_progress(tracker):
    """Plot training progress from saved history"""
    if not tracker.history:
        print("No training history to plot")
        return
    
    df = pd.DataFrame(tracker.history)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(df['epoch'], df['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0].plot(df['epoch'], df['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(df['epoch'], df['loss'], 'b-', label='Train', linewidth=2)
    axes[1].plot(df['epoch'], df['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'training_progress.png', dpi=150)
    print(f"\n✅ Progress plot saved to: {LOG_DIR / 'training_progress.png'}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🌿 PLANT DISEASE DETECTION - LAPTOP EDITION")
    print("="*70)
    
    # Initialize progress tracker
    tracker = ProgressTracker(PROGRESS_FILE)
    
    # Get dataset
    dataset_path = download_or_locate_dataset()
    
    train_dir, val_dir, class_names = setup_data_directories(dataset_path)
    
    # Create generators (with sample limiting for quick testing)
    train_gen, val_gen = create_limited_generators(
        train_dir, 
        val_dir, 
        samples_per_class=SAMPLES_PER_CLASS  # Remove this parameter for full training
    )
    
    # Build model
    model = build_lightweight_model(len(class_names))
    
    # Train with auto-resume
    model = train_with_resume(model, train_gen, val_gen, tracker)
    
    # Plot results
    plot_progress(tracker)
    
    # Save final model
    final_path = CHECKPOINT_DIR / 'final_model.keras'
    model.save(final_path)
    print(f"\n✅ Final model saved: {final_path}")
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"Best validation accuracy: {tracker.best_val_acc:.4f}")
    print(f"Total samples processed: {tracker.samples_processed}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()