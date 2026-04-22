# ─────────────────────────────────────────────────────────────────────────────
# src/config.py  —  Proje genelinde geçerli sabitler ve hiperparametreler
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
# ─────────────────────────────────────────────────────────────────────────────

# ── Tekrarlanabilirlik ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Veri bölme oranları ───────────────────────────────────────────────────────
TEST_SIZE = 0.15   # %15 test
DEV_SIZE  = 0.15   # %15 dev (validasyon)
# Train = %70

# ── Model 1 — Lab modeli (1 gizli katman, dar) ────────────────────────────────
N_H_1   = 8      # Gizli nöron sayısı
LR_1    = 0.1    # SGD öğrenme hızı
STEPS_1 = 1000   # Epoch sayısı

# ── Model 2 — Geniş (1 gizli katman) ─────────────────────────────────────────
N_H_2 = 32

# ── Model 3 — Geniş + L2 ─────────────────────────────────────────────────────
N_H_3      = 32
L2_LAMBDA_3 = 0.01

# ── Model 4 — 2 Gizli Katman ─────────────────────────────────────────────────
N_H1_4 = 16
N_H2_4 = 8

# ── Model 5 — 2 Gizli Katman + L2 ────────────────────────────────────────────
N_H1_5     = 16
N_H2_5     = 8
L2_LAMBDA_5 = 0.01

# ── Ortak eğitim ayarları (tüm modeller) ─────────────────────────────────────
LEARNING_RATE    = 0.1
N_STEPS          = 1000
HIDDEN_ACTIVATION = 'tanh'   # gizli katman aktivasyonu
OUTPUT_ACTIVATION = 'sigmoid' # çıkış katmanı aktivasyonu
LOSS              = 'binary_cross_entropy'
OPTIMIZER         = 'SGD'    # momentum=0.0 (saf SGD)
W_INIT            = 'N(0, 0.01)'
B_INIT            = 'zeros'

# ── Grid search ayarları ──────────────────────────────────────────────────────
GRID_NH_VALUES   = [4, 6, 8, 10, 12, 16]
GRID_STEP_VALUES = [200, 400, 600, 800, 1000, 1500]
GRID_THRESHOLD   = 0.75  # dev_acc eşiği

# ── Veri miktarı deneyi oranları ─────────────────────────────────────────────
DATA_FRACTIONS = [0.50, 0.75, 1.00]
