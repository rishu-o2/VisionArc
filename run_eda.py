"""
Week 1 EDA Pipeline - VisionArc
Dataset uses ImageFolder structure (class labels = folder names).
"""
import os
import glob
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ── 0. Directory Setup
for d in ['data/raw', 'data/labels', 'data/processed', 'notebooks', 'reports/figures']:
    os.makedirs(d, exist_ok=True)

# ── 1. Build DataFrame from folder structure
# Use dr_unified_v2 as the primary (non-augmented) dataset
BASE_DIR = 'data/raw/dr_unified_v2/dr_unified_v2'
if not os.path.isdir(BASE_DIR):
    # Try alternate path
    BASE_DIR = 'data/raw/dr_unified_v2'
    if not os.path.isdir(BASE_DIR):
        print("ERROR: Cannot find dr_unified_v2 directory!")
        # List what we have
        for item in os.listdir('data/raw'):
            print(f"  {item}")
        exit(1)

print(f"Using dataset: {BASE_DIR}")
print(f"Splits found: {os.listdir(BASE_DIR)}")

CLASS_NAMES = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative'
}

records = []
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(BASE_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"  Warning: {split_dir} not found, skipping")
        continue
    for class_folder in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        severity = int(class_folder)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                records.append({
                    'image_id': img_file,
                    'severity': severity,
                    'split': split,
                    'image_path': os.path.join(class_path, img_file)
                })

df = pd.DataFrame(records)
print(f"\nTotal images found: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Splits: {dict(df['split'].value_counts())}")
print(f"\nFirst 10 rows:\n{df.head(10)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── 2. Class Distribution
print("\n== Class Distribution ==")
counts = df['severity'].value_counts().sort_index()
percentages = (counts / len(df)) * 100
imbalance_ratio = counts.max() / counts.min()

for idx, count in counts.items():
    name = CLASS_NAMES.get(idx, f'Class {idx}')
    print(f"  Class {idx} ({name}): {count} images ({percentages[idx]:.2f}%)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

# ── 3. Image Statistics (sample 500)
print("\n== Image Statistics (sampling 500 images) ==")
np.random.seed(42)
all_paths = df['image_path'].tolist()
sample_size = min(500, len(all_paths))
sample_paths = np.random.choice(all_paths, sample_size, replace=False)

heights, widths, channels_list = [], [], []
corrupted = 0
for p in sample_paths:
    img = cv2.imread(p)
    if img is None:
        corrupted += 1
        continue
    h, w, c = img.shape
    heights.append(h)
    widths.append(w)
    channels_list.append(c)

valid = sample_size - corrupted
quality_pct = (valid / sample_size) * 100
print(f"  Sampled: {sample_size}")
print(f"  Valid: {valid} | Corrupted: {corrupted}")
print(f"  Mean dims: {np.mean(heights):.0f} x {np.mean(widths):.0f}")
print(f"  Min dims:  {np.min(heights)} x {np.min(widths)}")
print(f"  Max dims:  {np.max(heights)} x {np.max(widths)}")
print(f"  Channels:  {set(channels_list)}")
print(f"  Data quality: {quality_pct:.1f}%")

# ── 4. Data Quality Checks
print("\n== Data Quality Checks ==")
# Check severity labels
unique_labels = sorted(df['severity'].unique())
print(f"  Unique severity labels: {unique_labels}")
print(f"  Labels valid (0-4 only): {set(unique_labels).issubset({0,1,2,3,4})}")

# Check for duplicate filenames
dup_count = df['image_id'].duplicated().sum()
print(f"  Duplicate filenames: {dup_count}")

# ── 5. Visualizations
print("\n== Generating Visualizations ==")
sns.set_theme(style='whitegrid')

# Plot 1: Class distribution bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    [f"Class {i}\n({CLASS_NAMES.get(i,'')})" for i in counts.index],
    counts.values,
    color=sns.color_palette('viridis', len(counts))
)
for bar, pct in zip(bars, percentages.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.01,
            f'{pct:.1f}%', ha='center', fontsize=11)
ax.set_title('Diabetic Retinopathy - Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Images')
plt.tight_layout()
plt.savefig('reports/figures/01_class_distribution.png', dpi=150)
plt.close()
print("  [OK] 01_class_distribution.png")

# Plot 2: Sample images per class (5x5 gallery)
fig, axes = plt.subplots(5, 5, figsize=(18, 18))
fig.suptitle('Sample Images per DR Severity Class', fontsize=16, fontweight='bold')
for i in range(5):
    class_rows = df[df['severity'] == i]
    if len(class_rows) == 0:
        continue
    samples = class_rows.sample(min(5, len(class_rows)), random_state=42)
    for j, (_, row) in enumerate(samples.iterrows()):
        if j >= 5:
            break
        img = cv2.imread(row['image_path'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            axes[i, j].imshow(img)
        axes[i, j].set_title(f'Class {i} ({CLASS_NAMES.get(i,"")})', fontsize=9)
        axes[i, j].axis('off')
# Turn off any unused axes
for i in range(5):
    for j in range(5):
        if not axes[i, j].has_data():
            axes[i, j].axis('off')
plt.tight_layout()
plt.savefig('reports/figures/02_sample_images_per_class.png', dpi=150)
plt.close()
print("  [OK] 02_sample_images_per_class.png")

# Plot 3: Image dimensions distribution
fig, axes_dim = plt.subplots(1, 2, figsize=(14, 5))
axes_dim[0].hist(heights, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes_dim[0].set_title('Height Distribution')
axes_dim[0].set_xlabel('Pixels')
axes_dim[1].hist(widths, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes_dim[1].set_title('Width Distribution')
axes_dim[1].set_xlabel('Pixels')
fig.suptitle('Image Dimensions Distribution (n=500)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/03_image_dimensions.png', dpi=150)
plt.close()
print("  [OK] 03_image_dimensions.png")

# Plot 4: Class imbalance pie chart
fig, ax = plt.subplots(figsize=(8, 8))
labels = [f"Class {i}\n{CLASS_NAMES.get(i,'')}" for i in counts.index]
colors = sns.color_palette('viridis', len(counts))
ax.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors,
       startangle=90, textprops={'fontsize': 11})
ax.set_title('Class Imbalance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/04_class_imbalance.png', dpi=150)
plt.close()
print("  [OK] 04_class_imbalance.png")

# Plot 5: Data quality breakdown
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(['Valid Images', 'Corrupted'], [valid, corrupted],
       color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('Data Quality Breakdown (n=500 sample)', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate([valid, corrupted]):
    ax.text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/05_data_quality.png', dpi=150)
plt.close()
print("  [OK] 05_data_quality.png")

# ── 6. Stratified Splits (re-split from full dataset for labels/)
print("\n== Creating Stratified Splits (70/15/15) ==")
# The dataset already has train/val/test, but we re-create our own stratified split
# from the FULL dataset to ensure exact 70/15/15 proportions
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['severity'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['severity'], random_state=42)

for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    split_df.to_csv(f'data/labels/{split_name}.csv', index=False)
    print(f"  {split_name}.csv: {len(split_df)} rows")
    dist = dict(split_df['severity'].value_counts().sort_index())
    print(f"    Class distribution: {dist}")

# ── 7. Generate Report
report = """========================================
WEEK 1 DATA EXPLORATION - COMPLETE
========================================

Dataset: EyePACS + APTOS + Messidor (Combined DR)
Source:   https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
Format:   ImageFolder (class labels = folder names 0-4)
Base Dir: {base_dir}

-- Total Images: {total} --

-- Class Distribution --
{class_dist}
-- Imbalance Ratio: {imbalance:.2f}x --

-- Image Statistics (sample of {sample_n}) --
  Mean dimensions: {mean_h:.0f} x {mean_w:.0f}
  Min dimensions:  {min_h} x {min_w}
  Max dimensions:  {max_h} x {max_w}
  Format:          RGB ({ch} channels)
  Corrupted:       {corrupted} / {sample_n}

-- Data Quality Score: {quality:.1f}% --

-- Train/Val/Test Split (Stratified 70/15/15) --
  Train: {n_train:>6} images (70%)
  Val:   {n_val:>6} images (15%)
  Test:  {n_test:>6} images (15%)

-- Key Findings --
  1. Dataset contains {total} retinal images across 5 severity levels.
  2. Class imbalance ratio is {imbalance:.2f}x - will need oversampling/class weights.
  3. Image dimensions vary - resizing to uniform size needed for training.
  4. {quality:.1f}% of sampled images are valid and readable.
  5. Unique severity labels: {labels} (all valid 0-4).
  6. Duplicate filenames: {dups}

-- Next Steps (Week 2) --
  1. Resize all images to 224x224 or 512x512.
  2. Apply data augmentation (rotation, flips, color jitter).
  3. Implement transfer learning with EfficientNet/ResNet.
  4. Use class weights or oversampling to handle imbalance.

-- Files Created --
  data/labels/train.csv
  data/labels/val.csv
  data/labels/test.csv
  reports/figures/01_class_distribution.png
  reports/figures/02_sample_images_per_class.png
  reports/figures/03_image_dimensions.png
  reports/figures/04_class_imbalance.png
  reports/figures/05_data_quality.png
  reports/01_DATA_EXPLORATION_REPORT.txt

========================================
""".format(
    base_dir=BASE_DIR,
    total=len(df),
    class_dist='\n'.join(f"  Class {idx} ({CLASS_NAMES.get(idx,'')}): {count:>6} images ({percentages[idx]:>5.2f}%)" for idx, count in counts.items()),
    imbalance=imbalance_ratio,
    sample_n=sample_size,
    mean_h=np.mean(heights), mean_w=np.mean(widths),
    min_h=np.min(heights), min_w=np.min(widths),
    max_h=np.max(heights), max_w=np.max(widths),
    ch=set(channels_list),
    corrupted=corrupted,
    quality=quality_pct,
    n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
    labels=unique_labels,
    dups=dup_count,
)

with open('reports/01_DATA_EXPLORATION_REPORT.txt', 'w') as f:
    f.write(report)
print("  [OK] 01_DATA_EXPLORATION_REPORT.txt")

# ── 8. Final Summary
print(f"""
========================================
WEEK 1 DATA EXPLORATION - COMPLETE
========================================
Total Images: {len(df)}
Class Distribution: [{', '.join(f'{p:.1f}%' for p in percentages.values)}]
Imbalance Ratio: {imbalance_ratio:.2f}x
Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)} images
Data Quality: {quality_pct:.1f}%
Files Created: 9
Ready for Week 2: YES
========================================
""")
