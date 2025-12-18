# Smart Library Book Pose Recognition System (智慧圖書館書籍姿態辨識系統)

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **NCU ME5301 深度學習專案設計 - 第 1 組：無人圖書館 (Unmanned Library)**

## 📖 專案簡介 (Introduction)

在現代圖書館管理中，雖然借還流程已大幅數位化，但**實體書架的盤點與整理**仍高度依賴人力。書籍常出現倒放、平放或書背未朝外等情況，導致書架混亂且尋書困難。

本專案旨在開發一套**書籍姿態與封面辨識系統**，透過深度學習 (Deep Learning) 技術自動偵測書籍在架上的擺放狀態。

### 核心目標
* **自動化整架**：解決人工檢查耗時費力的痛點。
* **姿態偵測**：精準辨識四種常見擺放狀態（正常、倒放、平放、書背不朝外）。
* **未來應用**：結合機械手臂實現自動歸位與精準尋書。

## 📂 資料集 (Dataset)

本專案資料集為 **100% 自行建立**，拍攝於**中央大學圖書館 (NCU Library)** 3~6 樓藏書區。

* **資料量**: 約 600+ 張影像 (持續擴充中)。
* **資料劃分**: Train (80%) / Validation (10%) / Test (10%)。
* **標註工具**: [Roboflow](https://roboflow.com/) (支援 AI 輔助標註與多人協作)。
* [cite_start]**資料擴增 (Augmentation)**: 幾何變換 (旋轉/翻轉)、亮度對比調整、遮擋與裁切 [cite: 341-344]。

### 標註類別 (Classes)

我們定義了以下四種書籍擺放姿態進行模型訓練：

| Class Name | Label | 說明 (Description) | 標註框顏色 |
| :--- | :--- | :--- | :--- |
| **正常擺放** | `book` | 書籍垂直站立，書背朝外，文字方向正確 | 🟣 紫色 |
| **倒放** | `reverse` | 書籍垂直站立，但書背文字上下顛倒 | 🟡 黃色 |
| **平放** | `flat` | 書籍橫躺或斜躺，非垂直狀態 | 🟠 橘色 |
| **書背不朝外** | `backward` | 書的封面、封底或切口朝外，無法辨識書背 | 🔴 紅色 |

## 🏗️ 模型架構 (Model Architectures)

為符合課程要求與探索不同神經網路特性，本團隊採用 **PyTorch** 實現了四種截然不同的模型架構進行效能比較：

| 成員 | 模型名稱 | 架構類型 | 特點與職責 |
| :--- | :--- | :--- | :--- |
| **黃紹輔** | **Pure CNN** | **Custom CNN** | **[Baseline]** 從零建構的輕量級卷積神經網路，驗證基礎特徵提取能力，不依賴預訓練權重。 |
| **陳昱誠** | **YOLOv8** | **Object Detection** | **[Real-time]** 使用單階段偵測算法 (One-stage Detector)，專注於多目標偵測與即時推論速度。 |
| **王宇廷** | **ResNet-18** | **Transfer Learning** | **[Deep Residual]** 引入殘差連接 (Skip Connection) 解決深層梯度問題，並使用 ImageNet 預訓練權重進行微調。 |
| **林宥臣** | **ViT** | **Transformer** | **[Attention]** Vision Transformer。捨棄傳統卷積，利用 Self-Attention 機制捕捉全域特徵，探索非 CNN 架構的可能性。 |

## 🚀 環境安裝與執行 (Installation & Usage)

本專案開發環境基於 **Python 3.10** 與 **PyTorch 2.0**。

### 1. Clone Repository
```bash
git clone [https://github.com/HuangShaoFu111/DLPD_finalproject.git](https://github.com/HuangShaoFu111/DLPD_finalproject.git)
cd DLPD_finalproject
