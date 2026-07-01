# 🥭 Mango Disease Detection System
**KLE Technological University, Belagavi**
**Guide:** Dr. Prema Akkasaligar
**Team:** Pavan Badiger

---

## 📁 Project Structure
```
mango-disease-detection/
│
├── train_model.py          ← Step 1: Train the CNN model
├── app.py                  ← Step 2: Run the web app
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Farmer-friendly web UI
│
├── dataset/                ← Put Kaggle dataset here (see below)
│   ├── Anthracnose/
│   ├── Bacterial Canker/
│   ├── Cutting Weevil/
│   ├── Die Back/
│   ├── Gall Midge/
│   ├── Healthy/
│   ├── Powdery Mildew/
│   └── Sooty Mould/
│
├── mango_disease_model.h5  ← Generated after training
└── class_indices.json      ← Generated after training
```

---

## 🚀 STEP-BY-STEP SETUP

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset from Kaggle
1. Go to: https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset
2. Click **Download** (you need a free Kaggle account)
3. Extract the zip file
4. Rename the extracted folder to `dataset` and place it in this project folder
5. Make sure the folder structure looks like the one above (each disease has its own subfolder)

### Step 3 — Train the Model
```bash
python train_model.py
```
- Training takes ~15–30 minutes depending on your hardware
- GPU (NVIDIA) will make it much faster
- After training you'll get:
  - `mango_disease_model.h5` — the trained model
  - `class_indices.json` — disease label mapping
  - `training_results.png` — accuracy/loss graphs

### Step 4 — Run the Web App
```bash
python app.py
```
- Open your browser and go to: **http://localhost:5000**
- Upload any mango leaf image
- The AI will detect the disease instantly!

---

## 🧠 Model Architecture
- **Base:** VGG19 (pre-trained on ImageNet)
- **Fine-tuning:** Last 4 convolutional blocks trainable
- **Head:** GlobalAveragePooling → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.3) → Softmax(8)
- **Optimizer:** Adam (lr=0.0001)
- **Expected Accuracy:** ~92–97% on validation set

## 🌿 Detectable Diseases (8 classes)
| Disease | Severity |
|---|---|
| Anthracnose | High |
| Bacterial Canker | High |
| Cutting Weevil | Medium |
| Die Back | High |
| Gall Midge | Medium |
| Healthy | None |
| Powdery Mildew | Medium |
| Sooty Mould | Low |

---

## 🛠️ Troubleshooting
- **CUDA/GPU error:** Run on CPU by adding `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` at the top of train_model.py
- **Memory error:** Reduce `BATCH_SIZE` from 32 to 16 in train_model.py
- **Dataset not found:** Make sure the `dataset/` folder is in the same directory as `train_model.py`
