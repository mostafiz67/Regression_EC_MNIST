"""
Author: Md Mostafizur Rahman
File: Configaration file
"""

import os
from pathlib import Path

# directory-related
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
SRC = ROOT / "src"
CHECKPOINT = ROOT / "checkpoints"
OUT = ROOT / "output"

if not OUT.exists():
    os.makedirs(OUT, exist_ok=True)


# Analysis Constants
NB_TRAIN_SAMPLES = 60000
NB_TEST_SAMPLES = 10000
NB_CLASS = 10

IMG_SIZE = 28
IMG_CHANNEL = 1
IMG_SHAPE = (IMG_SIZE,  IMG_SIZE,  IMG_SIZE)

LR = 0.01
NB_EPOCS = 50
K_Fold = 5
BATCH_SIZE = 256
NB_REP = 15

CHOICES = ['Train-Test', 'Compare']
