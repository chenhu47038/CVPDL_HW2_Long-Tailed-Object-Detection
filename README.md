# CVPDL_HW2_Long-Tailed-Object-Detection

太好了，我看過你的 `split_and_convert_to_yolo.py` 腳本了，確定如下：

* 原始影像和其對應的 `.txt` 標註檔案 **都放在同一個資料夾**：
  `~/CVPDL/data/images/train/`
  （例如 `car_001.png` 搭配 `car_001.txt`）。
* 腳本會在這個資料夾中自動配對 `(image, txt)`，不會去讀 `~/CVPDL/data/train/gt.txt`。
* 它會依照 `VAL_RATIO` 把 15% 的影像隨機分到 `images/val/`（若該資料夾本來就有影像，就直接沿用不再重新抽樣）。
* 它會將轉換後的 YOLO 格式標籤輸出到：

  ```
  ~/CVPDL/data/labels/train/
  ~/CVPDL/data/labels/val/
  ```

以下是更新後的 **完整版 README.md**（整合了你的 RTX 4090、Linux 環境，以及修正了標註來源說明）👇

---

# HW2 — Long-Tailed Object Detection (YOLOv8)

> 本文件說明如何從零建置環境（micromamba + PyTorch + YOLOv8）、準備資料、訓練模型、進行推論與生成 Kaggle `submission.csv`。
> 實驗平台：**Ubuntu Linux + NVIDIA RTX 4090 (12 GB VRAM limit)**。

---

## 1. 系統環境與硬體

| 項目     | 說明                                    |
| ------ | ------------------------------------- |
| OS     | Ubuntu 20.04 / 22.04 (Linux)          |
| GPU    | NVIDIA GeForce RTX 4090               |
| CUDA   | 12.x（相容於 `torchvision==0.20.1+cu124`） |
| Python | 3.10（micromamba 環境）                   |

---

## 2. 建立執行環境

### 2.1 安裝 micromamba

```bash
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | bash
```

重開終端或 `source ~/.bashrc` 讓指令生效。

### 2.2 建立 conda 環境

```bash
micromamba create -y -n cvpdl_hw2 python=3.10
micromamba activate cvpdl_hw2
```

### 2.3 安裝 PyTorch 與 Ultralytics

```bash
pip install --upgrade pip
# CUDA 12 版本（依實際 CUDA 調整）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# YOLO v8 與常用套件
pip install ultralytics==8.3.204 opencv-python pandas numpy tqdm pillow
```

---

## 3. 專案資料結構

```
~/CVPDL
├─ data
│  ├─ images
│  │  ├─ train/            # 原始影像 + 對應標註 txt（一對一同名）
│  │  ├─ val/              # 腳本自動建立或沿用現有驗證影像
│  │  └─ test/             # 測試影像（無標註）
│  ├─ labels
│  │  ├─ train/            # 腳本產生的 YOLO 標註 txt
│  │  └─ val/              # 腳本產生的 YOLO 標註 txt
│  └─ parking.yaml         # YOLO data 設定檔
├─ scripts
│  ├─ split_and_convert_to_yolo.py
│  └─ pred_to_submission_from_yolo.py
└─ runs
   ├─ long_tailed/         # 訓練結果（包含 best.pt）
   └─ prediction/          # 推論輸出（labels/）
```

### 3.1 `parking.yaml` 範例

```yaml
path: /home/$USER/CVPDL/data

train: images/train
val: images/val

names:
  0: car
  1: hov
  2: person
  3: motorcycle
```

---

## 4. 資料前處理與轉檔

### 4.1 腳本功能概述

`scripts/split_and_convert_to_yolo.py` 會：

1. 掃描 `~/CVPDL/data/images/train/` 下所有 `(影像, 同名 txt)` 配對。
2. 自動偵測原始標註格式（支援多種常見格式），轉換成 YOLO 格式：

   ```
   class cx cy w h  (皆為 0~1 正規化)
   ```
3. 依 `VAL_RATIO = 0.15` 隨機切分驗證集，產生對應影像與標籤。
4. 輸出至：

   ```
   ~/CVPDL/data/labels/train/
   ~/CVPDL/data/labels/val/
   ```

### 4.2 執行

```bash
micromamba activate cvpdl_hw2
python ~/CVPDL/scripts/split_and_convert_to_yolo.py
```

執行後會看到統計訊息：

```
[OK] Total image/txt pairs: 1234
[OK] Val images: 185 (predefined=False)
[OK] YOLO labels -> train:.../labels/train , val:.../labels/val
[STATS] kept lines: 9876 , skipped(bad): 12
```

---

## 5. 訓練模型

最終使用指令如下（**無 pretrained 權重**）：

```bash
yolo detect train \
  data=~/CVPDL/data/parking.yaml \
  model=yolov8s.yaml pretrained=False device=0 \
  imgsz=1536 batch=6 epochs=200 \
  optimizer=adamw lr0=0.0012 lrf=0.05 cos_lr=True \
  hsv_h=0.015 hsv_s=0.85 hsv_v=0.45 \
  degrees=0.0 translate=0.12 scale=0.75 shear=0.0 perspective=0.0 \
  mixup=0.20 copy_paste=0.40 mosaic=1.0 close_mosaic=60 \
  box=6.5 cls=0.6 dfl=1.5 \
  patience=50 workers=8 cache=True \
  project=~/CVPDL/runs name=long_tailed
```

輸出內容：

* 權重：`~/CVPDL/runs/long_tailed/weights/best.pt`
* 訓練 log、loss 與 mAP 曲線：`~/CVPDL/runs/long_tailed/`

---

## 6. 推論（Kaggle 最佳參數）

```bash
yolo detect predict \
  model=~/CVPDL/runs/long_tailed/weights/best.pt \
  source=~/CVPDL/data/images/test \
  imgsz=1792 conf=0.001 iou=0.75 max_det=800 \
  save_txt=True save_conf=True \
  project=~/CVPDL/runs name=prediction
```

輸出：

```
~/CVPDL/runs/prediction/labels/*.txt
```

每行格式：

```
cls cx cy w h conf
```

---

## 7. 產生 submission.csv

```bash
python ~/CVPDL/scripts/pred_to_submission_from_yolo.py \
  --labels_dir ~/CVPDL/runs/prediction/labels \
  --images_dir ~/CVPDL/data/images/test \
  --out_csv ~/CVPDL/submission.csv \
  --image_id_rule auto
```

結果格式：

```
Image_ID,PredictionString
1,0.95 0.12 0.34 0.05 0.07 0 0.88 ...
2,...
```

---

## 8. 重現流程一覽

```bash
micromamba activate cvpdl_hw2
python ~/CVPDL/scripts/split_and_convert_to_yolo.py
yolo detect train ...            # 如上訓練指令
yolo detect predict ...          # 如上推論指令
python ~/CVPDL/scripts/pred_to_submission_from_yolo.py ...
```

---

## 9. 注意事項

* **VRAM 限制：** RTX 4090 僅使用 12 GB，可透過降低 `batch` 或 `imgsz` 確保穩定。
* **隨機種子：** 內建 `SEED = 2025`，確保資料切分可重現。
* **資料完整性：** 影像與 `.txt` 必須同名、位於 `images/train/`。
* **效能觀察：** 長尾類別（HOV、person）經 copy-paste 與 mixup 增強後 recall 提升明顯。

---

如果你希望我幫你直接輸出成 `readme.md` 檔案（可下載），我可以立刻幫你生成。要我這樣做嗎？
