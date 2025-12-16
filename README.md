# Smart Library Book Pose Recognition System (智慧圖書館書籍姿態辨識系統)

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **NCU ME5301 深度學習專案設計 - 第 1 組：無人圖書館 (Unmanned Library)**

## 📖 專案簡介 (Introduction)

[cite_start]在現代圖書館管理中，雖然借還流程已大幅數位化，但**實體書架的盤點與整理**仍高度依賴人力。書籍常出現倒放、平放或書背未朝外等情況，導致書架混亂且尋書困難 [cite: 185-187, 296-298]。

[cite_start]本專案旨在開發一套**書籍姿態與封面辨識系統**，透過深度學習 (Deep Learning) 技術自動偵測書籍在架上的擺放狀態 [cite: 189, 300]。

### 核心目標
* [cite_start]**自動化整架**：解決人工檢查耗時費力的痛點 [cite: 192]。
* [cite_start]**姿態偵測**：精準辨識四種常見擺放狀態（正常、倒放、平放、書背不朝外）[cite: 190]。
* [cite_start]**未來應用**：結合機械手臂實現自動歸位與精準尋書 [cite: 192]。

## 📂 資料集 (Dataset)

[cite_start]本專案資料集為 **100% 自行建立**，拍攝於**中央大學圖書館 (NCU Library)** 3~6 樓藏書區 [cite: 196, 197]。

* [cite_start]**資料量**: 約 600+ 張影像 (持續擴充中) [cite: 285]。
* [cite_start]**資料劃分**: Train (80%) / Validation (10%) / Test (10%) [cite: 284]。
* [cite_start]**標註工具**: [Roboflow](https://roboflow.com/) (支援 AI 輔助標註與多人協作) [cite: 203, 334]。
* [cite_start]**資料擴增 (Augmentation)**: 幾何變換 (旋轉/翻轉)、亮度對比調整、遮擋與裁切 [cite: 341-344]。

### 標註類別 (Classes)

[cite_start]我們定義了以下四種書籍擺放姿態進行模型訓練 [cite: 266-270, 328-332]：

| Class Name | Label | 說明 (Description) | 標註框顏色 |
| :--- | :--- | :--- | :--- |
| **正常擺放** | `book` | 書籍垂直站立，書背朝外，文字方向正確 | 🟣 紫色 |
| **倒放** | `reverse` | 書籍垂直站立，但書背文字上下顛倒 | 🟡 黃色 |
| **平放** | `flat` | 書籍橫躺或斜躺，非垂直狀態 | 🟠 橘色 |
| **書背不朝外** | `backward` | 書的封面、封底或切口朝外，無法辨識書背 | 🔴 紅色 |

## 🏗️ 模型架構 (Model Architectures)

[cite_start]本團隊針對此任務設計並比較了兩種神經網路架構 [cite: 349-358]：

### 1. Pure CNN (Custom Architecture)
> [cite_start]**特點**: 輕量級、結構簡單，作為 Baseline 模型 [cite: 349]。
* **Input**: 640x640 RGB 影像
* **架構**: 
    * 2 層卷積層 (Convolutional Layers, 3x3 kernel)
    * 2 層最大池化層 (Max Pooling)
    * 2 層全連接層 (Fully Connected Layers)
* **Activation**: ReLU (Hidden), Softmax (Output)

### 2. ResNet-18 (Transfer Learning)
> [cite_start]**特點**: 利用預訓練權重提升特徵提取能力，解決深層網路梯度消失問題 [cite: 351, 355]。
* **Backbone**: PyTorch 官方預訓練 ResNet-18
* **修改**: 
    * 將最後一層 FC layer 修改為 4 類輸出。
    * 加入 Dropout Layer 以提升泛化能力。
* **Weights**: ImageNet Pre-trained weights

## 🚀 環境安裝與執行 (Installation & Usage)

[cite_start]本專案開發環境基於 **Python 3.10** 與 **PyTorch 2.0** [cite: 361]。

### 1. Clone Repository
```bash
git clone [https://github.com/HuangShaoFu111/DLPD_finalproject.git](https://github.com/HuangShaoFu111/DLPD_finalproject.git)
cd DLPD_finalproject
