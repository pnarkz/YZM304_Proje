# YZM304 Derin Öğrenme — I. Proje Ödevi
## Pima Indians Diabetes: MLP ile İkili Sınıflandırma

**Ders:** YZM304 Derin Öğrenme | Ankara Üniversitesi  
**Dönem:** 2025–2026 Bahar  
**Veri Seti:** Pima Indians Diabetes Database (NIDDK / Kaggle)

---

## Introduction (Giriş)

Diyabet, dünya genelinde yaygın ve kronik bir metabolik hastalıktır. Erken teşhis, komplikasyonların önlenmesi açısından kritik önem taşımaktadır. Bu çalışmada, Pima Kızılderili kadınlarından elde edilen klinik ölçümler kullanılarak diyabet varlığını tahmin eden çok katmanlı algılayıcı (MLP) modelleri geliştirilmiştir.

Çalışmanın amacı; 13.03.2026 tarihli laboratuvar uygulamasındaki 1 gizli katmanlı NumPy MLP'yi gerçek bir tıbbi veri seti üzerinde uygulamak, modelin overfitting/underfitting davranışını analiz etmek, L2 regularizasyon ve çok katmanlı mimarilerin etkisini gözlemlemek, veri miktarının model başarısına etkisini incelemek ve sonuçları Sklearn ile PyTorch implementasyonlarıyla karşılaştırmaktır.

---

## Methods (Yöntemler)

### Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | NIDDK / Kaggle (Pima Indians Diabetes Database) |
| Dosya | `data/diabetes.csv` |
| Gözlem sayısı | 768 |
| Özellik sayısı | 8 |
| Hedef | `Outcome` — 0: Diyabet yok, 1: Diyabet var |
| Sınıf dağılımı | %65.1 (0) / %34.9 (1) — dengesiz |

### Veri Ön İşleme

1. **Biyolojik sıfır imputation:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` sütunlarındaki 0 değerleri medyan ile değiştirildi.
2. **Karıştırma:** `random_state=42` ile shuffle.
3. **Bölme:** Train %70 (537) / Dev %15 (115) / Test %15 (116) — stratified split.
4. **Standardizasyon:** `StandardScaler` — yalnızca train seti üzerinde `fit`, dev ve test setlerine `transform`.

### Hiperparametreler ve Başlangıç Ayarları

> **Tüm sabitler `src/config.py` dosyasında tanımlıdır.**

```python
RANDOM_SEED = 42
TEST_SIZE   = 0.15
DEV_SIZE    = 0.15

# Model 1 (Lab modeli)
N_H_1   = 8      # gizli nöron sayısı
LR_1    = 0.1    # SGD öğrenme hızı
STEPS_1 = 1000   # epoch sayısı

# Model 2 — Geniş
N_H_2 = 32

# Model 3 — Geniş + L2
N_H_3 = 32  |  L2_LAMBDA_3 = 0.01

# Model 4 — 2 Gizli Katman
N_H1_4, N_H2_4 = 16, 8

# Model 5 — 2 Gizli Katman + L2
N_H1_5, N_H2_5 = 16, 8  |  L2_LAMBDA_5 = 0.01

# Grid search
GRID_NH_VALUES   = [4, 6, 8, 10, 12, 16]
GRID_STEP_VALUES = [200, 400, 600, 800, 1000, 1500]
GRID_THRESHOLD   = 0.75  # dev_acc eşiği

# Veri miktarı deneyi
DATA_FRACTIONS = [0.50, 0.75, 1.00]
```

### Model Seçim Kriteri

> Dev accuracy ≥ **%75** olan modeller arasından `n_steps` en düşük olanı seç.  
> **Test seti model seçiminde kullanılmaz** — yalnızca final raporlamada kullanılır.

### Model Mimarileri

| ID | Katmanlar | Regularizasyon | Framework |
|----|-----------|----------------|-----------|
| M1 | 8 → 8 → 1 | — | NumPy |
| M2 | 8 → 32 → 1 | — | NumPy |
| M3 | 8 → 32 → 1 | L2 (λ=0.01) | NumPy |
| M4 | 8 → 16 → 8 → 1 | — | NumPy |
| M5 | 8 → 16 → 8 → 1 | L2 (λ=0.01) | NumPy |
| SK1 | 8 → 8 → 1 | — | Sklearn |
| SK2 | 8 → 16 → 8 → 1 | — | Sklearn |
| PT1 | 8 → 8 → 1 | — | PyTorch |
| PT2 | 8 → 16 → 8 → 1 | — | PyTorch |

### Dosya Yapısı

```
proje/
├── data/
│   └── diabetes.csv
├── src/
│   ├── __init__.py
│   ├── config.py          # Tüm sabitler ve hiperparametreler
│   ├── data_loader.py     # Yükleme, EDA, ön işleme, bölme
│   ├── models_numpy.py    # _BaseMLPNumpy, MLP_1Hidden, MLP_2Hidden
│   ├── models_sklearn.py  # Sklearn MLPClassifier sarmalayıcısı
│   ├── models_pytorch.py  # PyTorch MLP sınıfları ve eğitim fonksiyonu
│   └── evaluate.py        # Metrikler, grafik araçları
├── YZM304_Proje1_Diabetes_MLP.ipynb
├── requirements.txt
└── README.md
```

### Sınıf Hiyerarşisi

```
_BaseMLPNumpy          ← ortak: _sigmoid, predict, predict_proba, score, plot_learning_curve
├── MLP_1Hidden        ← özgün: __init__, _initialize_parameters, _hidden_act,
│                         _hidden_act_deriv, _forward, _compute_cost, _backward, _update, fit
└── MLP_2Hidden        ← özgün: __init__, _initialize_parameters,
                          _forward, _compute_cost, _backward, _update, fit
```

---

## Results (Sonuçlar)

### Veri Analizi Bulguları

- `Insulin` (%48.7) ve `SkinThickness` (%29.6) sütunlarında yüksek oranda biyolojik sıfır bulunmaktadır.
- `Glucose`, `BMI` ve `Age` hedef değişkenle en güçlü korelasyonu göstermektedir.
- Sınıf dağılımı dengesizdir (%65/%35); bu nedenle precision, recall ve F1 de raporlanmıştır.

### NumPy Modelleri — Overfitting / Underfitting Analizi

| Model | Train Acc | Dev Acc | Fark (Train−Dev) | Yorum |
|-------|-----------|---------|-------------------|-------|
| M1 (n_h=8) | 0.7840 | 0.8261 | −0.0421 | ✅ Makul denge |
| M2 (n_h=32) | 0.7747 | 0.8087 | −0.0340 | ✅ Makul denge |
| M3 (n_h=32+L2) | 0.7747 | 0.8087 | −0.0340 | L2 fark yaratmadı |
| M4 (16→8) | 0.7616 | 0.8087 | −0.0471 | ✅ Makul denge |
| M5 (16→8+L2) | 0.7616 | 0.8087 | −0.0471 | L2 fark yaratmadı |

> Tüm modellerde Dev Acc > Train Acc olması, overfitting olmadığını göstermektedir.  
> Negatif fark, modellerin hafif underfitting eğiliminde olduğuna işaret eder; ancak bu, küçük ve gürültülü bir veri seti (768 örnek) için beklenen bir davranıştır.

### Test Seti Metrikleri

| Model | Framework | Test Acc | Precision | Recall | F1 Score |
|-------|-----------|----------|-----------|--------|----------|
| M1 (n_h=8) | NumPy | 0.7586 | 0.6857 | 0.5854 | 0.6316 |
| M2 (n_h=32) | NumPy | **0.7672** | 0.7059 | 0.5854 | 0.6400 |
| M3 (n_h=32+L2) | NumPy | **0.7672** | 0.7059 | 0.5854 | 0.6400 |
| M4 (16→8) | NumPy | 0.7500 | 0.6765 | 0.5610 | 0.6133 |
| M5 (16→8+L2) | NumPy | 0.7500 | 0.6765 | 0.5610 | 0.6133 |
| SK1 (1 gizli) | Sklearn | **0.7672** | 0.7059 | 0.5854 | 0.6400 |
| SK2 (2 gizli) | Sklearn | 0.7586 | 0.6857 | 0.5854 | 0.6316 |
| PT1 (1 gizli) | PyTorch | **0.7672** | 0.7059 | 0.5854 | 0.6400 |
| PT2 (2 gizli) | PyTorch | **0.7672** | 0.8500 | 0.4146 | 0.5574 |

### Grid Search Sonucu

Grid search (n_h × n_steps), yalnızca train ve dev seti üzerinde yapılmıştır:
- **Seçilen model:** n_h=4, n_steps=200 (Dev acc ≥ %75 eşiğini en az adımda geçen model)
- **Seçilen model metrikleri:** Train: 0.7616 | Dev: 0.8000

### Veri Miktarı Deneyi

Seçilen model (n_h=4, n_steps=200), train verisinin %50, %75 ve %100'ü ile eğitilmiştir:

| Oran | Train Acc | Dev Acc | Test Acc |
|------|-----------|---------|----------|
| %50  | 0.7537    | 0.8087  | 0.7586   |
| %75  | 0.7587    | 0.7913  | 0.7759   |
| %100 | 0.7616    | 0.8000  | 0.7672   |

> Veri miktarı arttıkça train accuracy yükselirken, dev/test skorları nispeten stabil kalmaktadır.

### ROC Eğrisi ve AUC Skorları

Sınıf dengesizliği (%65/%35) nedeniyle accuracy tek başına yanıltıcı olabilir. AUC, modellerin sınıfları ayırma gücünü daha doğru yansıtır:

| Model | AUC |
|-------|-----|
| NumPy M1 (n_h=8) | 0.8550 |
| NumPy M2 (n_h=32) | 0.8556 |
| NumPy M4 (16→8) | 0.8582 |
| Sklearn M1 (n_h=8) | **0.8624** |
| PyTorch M1 (n_h=8) | 0.8537 |

> En yüksek AUC Sklearn M1 modelinde elde edilmiştir (0.8624). Tüm modeller 0.85+ AUC ile iyi ayırıcı performans göstermektedir.

### Mini-batch SGD Karşılaştırması

Tam-batch SGD ile mini-batch SGD (batch_size=64) karşılaştırılmıştır:

| Yöntem | Train Acc | Dev Acc | Test Acc |
|--------|-----------|---------|----------|
| Full-batch | 0.7840 | 0.8261 | 0.7586 |
| Mini-batch (bs=64) | 0.8399 | 0.7652 | 0.7845 |

> Mini-batch SGD, train accuracy'yi belirgin şekilde artırmıştır (+0.056) ve test accuracy'de de iyileşme sağlamıştır (+0.026). Ancak dev accuracy düşmüştür (−0.061), bu da mini-batch'in stokastik doğasından kaynaklanan varyans artışına işaret etmektedir.

### Framework Karşılaştırması

Aynı mimari, aynı SGD optimizer ve aynı başlangıç ağırlıkları ile üç framework (NumPy, Sklearn, PyTorch) benzer sonuçlar üretmiştir. En iyi test accuracy **0.7672** olup birden fazla model (M2, M3, SK1, PT1, PT2) bu değere ulaşmıştır. Küçük farklılıklar float32/float64 hassasiyet farklarından kaynaklanmaktadır.

---

## Discussion (Tartışma ve Gelecek Çalışmalar)

### Yorumlar

- **Overfitting gözlemlenmedi:** Tüm modellerde Dev Acc ≥ Train Acc olduğundan L2 regularizasyon belirgin bir fark yaratmamıştır.
- **Veri kalitesi:** `Insulin` ve `SkinThickness` sütunlarındaki yüksek kayıp veri oranı model başarısını sınırlamaktadır.
- **Sınıf dengesizliği:** %65/%35 dağılım nedeniyle recall (diyabeti kaçırma oranı) özellikle önemlidir. Tüm modellerde recall ~0.56 civarındadır. AUC skoru (0.85+) sınıfları ayırma gücünün iyi olduğunu göstermektedir.
- **Mini-batch SGD:** Mini-batch (bs=64) train accuracy'yi artırmış (+5.6pp) ve test'te de iyileşme sağlamıştır. Ancak dev accuracy düşmesi, stokastik gradyanın varyans artışına neden olduğunu göstermektedir.
- **Veri miktarı:** %50 train verisiyle bile test accuracy 0.7586 elde edilmiştir; bu, modelin veri miktarına karşı görece dayanıklı olduğunu göstermektedir.
- **Model seçimi:** Grid search yalnızca dev seti üzerinden yapılmış; test seti model seçim sürecinden tamamen ayrı tutulmuştur.
- **PyTorch M2 dikkat:** Precision çok yüksek (0.85) ama recall çok düşük (0.41). Model muhafazakâr tahmin yapıyor.

### Gelecek Çalışmalar

1. Dropout regularizasyon
2. Batch Normalization
3. Adam / RMSProp optimizer karşılaştırması
4. SMOTE ile sınıf dengesizliğinin giderilmesi
5. KNN / MICE imputation
6. He / Xavier ağırlık başlatma karşılaştırması

---

## Kurulum ve Çalıştırma

```bash
# Gerekli kütüphaneler
pip install -r requirements.txt

# Notebook'u başlat
jupyter notebook YZM304_Proje1_Diabetes_MLP.ipynb
```

**Python:** 3.9+  
**Önemli:** `data/diabetes.csv` dosyası repo içinde bulunmalıdır.  
Notebook, `data/diabetes.csv` → `diabetes.csv` → `../data/diabetes.csv` sırasıyla dosyayı arar.
