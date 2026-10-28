import os
import json
import zipfile
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nbformat as nbf

# Define folders
folders = [
    'data/raw',
    'data/labels',
    'data/processed',
    'notebooks',
    'reports/figures'
]
for f in folders:
    os.makedirs(f, exist_ok=True)

# Read token from environment or ~/.kaggle/ credentials
if 'KAGGLE_API_TOKEN' not in os.environ:
    token_file = os.path.expanduser('~/.kaggle/access_token')
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            os.environ['KAGGLE_API_TOKEN'] = f.read().strip()

print("Downloading Kaggle dataset...")
res = os.system('kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy -p data/raw/')
if res != 0:
    print("Download failed!")
    exit(1)
print("Extracting zip files...")
zip_path = 'data/raw/eyepacs-aptos-messidor-diabetic-retinopathy.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw/')
    os.remove(zip_path)

nb = nbf.v4.new_notebook()

nb_cells = []
nb_cells.append(nbf.v4.new_markdown_cell("# Week 1 Data Exploration"))
nb_cells.append(nbf.v4.new_code_cell())

nb_cells.append(nbf.v4.new_markdown_cell())
nb_cells.append(nbf.v4.new_code_cell())

nb_cells.append(nbf.v4.new_markdown_cell())
nb_cells.append(nbf.v4.new_code_cell())

nb_cells.append(nbf.v4.new_markdown_cell("## C. Image Statistics (sample 500 images)"))
nb_cells.append(nbf.v4.new_code_cell())

nb_cells.append(nbf.v4.new_markdown_cell("## D. Data Quality Checks"))
nb_cells.append(nbf.v4.new_code_cell())

nb_cells.append(nbf.v4.new_markdown_cell("## E. Create Stratified Splits"))
nb_cells.append(nbf.v4.new_code_cell())

nb.cells.extend(nb_cells)
with open('notebooks/01_data_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Pipeline script completed. Notebook created.")
