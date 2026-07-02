# 🥭 Mango Fruit Disease Detection System
**KLE Technological University, Belagavi**  
 
**Developed By:** Pavan Badiger

---

## ⬇️ Downloads (Model & Dataset)

> ⚠️ Due to GitHub's 100MB file size limit, the trained model and dataset are hosted on Google Drive.

| File | Size | Download |
|---|---|---|
| `mango_disease_model.h5` (Pre-trained Model) | ~171 MB | [Download from Google Drive](https://drive.google.com/file/d/1LZ1TG34-oQ8b1dQMtcYJ9FKDfGy61mjz/view?usp=sharing) |
| `dataset.zip` (Mango Fruit Disease Dataset) | Large | [Download from Google Drive](https://drive.google.com/file/d/1Ycuj4rVSwbzdeObMrqfEq2wy5foLXLu0/view?usp=sharing) |

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
├── mango_disease_model.h5  ← Download from Google Drive (link above)
└── class_indices.json      ← Generated after training
```

---

## 🚀 STEP-BY-STEP SETUP

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset
**Option A: Use our pre-downloaded dataset (Recommended)**
1. Download `dataset.zip` from the [Google Drive link](https://drive.google.com/file/d/1Ycuj4rVSwbzdeObMrqfEq2wy5foLXLu0/view?usp=sharing) above
2. Extract and rename the folder to `dataset`
3. Place it in the project root folder

**Option B: Download from Kaggle directly**
1. Go to: https://www.kaggle.com/datasets/warcoder/mangofruitdds
2. Click **Download** (free Kaggle account required)
3. Extract and rename to `dataset`, place in project root

Make sure the folder structure matches the one shown above (each disease in its own subfolder).

### Step 3 — Get the Trained Model

**Option A: Use Pre-trained Model (Skip Training — Recommended)**
1. Download `mango_disease_model.h5` from [Google Drive](https://drive.google.com/file/d/1LZ1TG34-oQ8b1dQMtcYJ9FKDfGy61mjz/view?usp=sharing)
2. Place it in the project root folder (same location as `app.py`)
3. Skip to **Step 4** directly!

**Option B: Train from Scratch**
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
- Upload any mango fruit image
- The AI will detect the disease instantly!

---

## 🧠 Model Architecture
- **Base:** VGG19 (pre-trained on ImageNet)
- **Fine-tuning:** Last 4 convolutional blocks trainable
- **Head:** GlobalAveragePooling → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.3) → Softmax(8)
- **Optimizer:** Adam (lr=0.0001)
- **Expected Accuracy:** ~92–97% on validation set

---

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
- **CUDA/GPU error:** Run on CPU by adding `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` at the top of `train_model.py`
- **Memory error:** Reduce `BATCH_SIZE` from 32 to 16 in `train_model.py`
- **Dataset not found:** Make sure the `dataset/` folder is in the same directory as `train_model.py`
- **Model not found:** Make sure `mango_disease_model.h5` is downloaded and placed in the root project folder

---

## 👨‍💻 Author
**Pavan Badiger**  
[GitHub](https://github.com/MrPavanBadiger3)
[LinkedIn] (linkedin.com/in/pavan-badiger-2b749928a)

---

## 📄 License
This project is open source and available for educational purposes.
