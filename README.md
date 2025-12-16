# Intelligent Robot Grab (機器人抓取智能化)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **NCU ME5301 深度學習專案設計 (Deep Learning Project Design) - 2025 期末專題**

## 📖 專案簡介 (Introduction)

本專案旨在解決工業自動化中「傳統型 (Traditional)」機械手臂抓取的限制。傳統機械手臂依賴固定座標，缺乏彈性；本專案透過深度學習技術，實現「智能型 (Intelligent)」機器人抓取，結合電腦視覺與深度神經網路，使機械手臂能夠辨識並抓取非結構化擺放的物體。

### 核心目標
* **智能化識別**：取代傳統示教 (Teaming) 或固定座標模式。
* **多樣化抓取**：針對吸盤 (Suction Cup) 或平行夾爪 (Parallel-Jaw Gripper) 進行優化。

## 📂 資料集 (Dataset)

> **⚠️ Note:** 本專案使用**100% 自行建立**的資料集，以符合實際場域需求。

* **資料集名稱**: (請填寫，例如: NCU-Grab-2025)
* **資料蒐集方式**: 
    * (在此描述你們如何拍攝，例如：使用 Realsense D435 拍攝、架設環境、光源設定)
    * (描述資料增強 Data Augmentation 的方法)
* **資料量**: 
    * Training Set: X 張
    * Validation Set: Y 張
    * Test Set: Z 張
* **標註工具**: (例如: LabelImg, Roboflow)

## 🏗️ 模型架構 (Model Architectures)

本專案由團隊成員分別開發不同的神經網路模型，並進行效能比較。

| 成員 | 模型名稱 | 架構類型 | 特點/創新點 | 準確率 (Accuracy) |
| :--- | :--- | :--- | :--- | :--- |
| 成員 A | **Baseline-CNN** | **Pure CNN** (必選) | 使用純捲積神經網路架構，無預訓練模型 | 92.5% |
| 成員 B | (例如: ResNet50-Transfer) | Transfer Learning | 使用 ImageNet 預訓練權重進行微調 | 95.1% |
| 成員 C | (例如: YOLOv8-Custom) | Object Detection | 針對即時抓取速度優化 | mAP 0.88 |
| 成員 D | (例如: Vision Transformer) | Transformer | 使用 Attention 機制捕捉全局特徵 | 94.8% |

*(註：依據課程規範，本專案包含至少一組純 CNN 架構模型。)*

## 🚀 環境安裝與執行 (Installation & Usage)

本專案主要基於 **Python** 與 **PyTorch** 開發。

### 1. Clone Repository
```bash
git clone [https://github.com/your-username/intelligent-robot-grab.git](https://github.com/your-username/intelligent-robot-grab.git)
cd intelligent-robot-grab
