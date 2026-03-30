# Suicide-Risk-Detection
Interpretable Suicide Risk Detection in Traditional Chinese: A Hybrid NLP Approach combining Chinese-MentalBERT with C-SSRS Grounded Linguistic Phenotyping.

# 繁體中文社群之自殺風險可解釋性偵測模型
> Interpretable Suicide Risk Detection in Traditional Chinese: A Hybrid NLP Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uh5zJflIHOCRdso0Fn98Gx5okrqNP84_?usp=sharing)

## 專案簡介
於非結構化的社群文本精準區分「即時自殺風險（High Risk）」與「一般憂鬱情緒（General Depressive Affect）」是一大挑戰。

本專案針對台灣社群（如 PTT Prozac 板）的文本特性，建構了一個強調高度可解釋性 (Interpretability) 的機器學習框架。有別於傳統的黑盒子深度學習模型，本研究試驗了 `Chinese-MentalBERT` 的深層語意表徵與多項手工萃取的語言學特徵，不僅提升預測準確度，也驗證了語言學特徵能夠帶來更多的可解釋性。

## 核心特色
* **在地化 NLP 前處理**：整合 `OpenCC` 進行辭典繁簡轉換，並使用 `ckip-transformers` 進行斷詞與詞性標註（POS Tagging）。系統內建針對在地社群用語（如：「登出人生」、「重刷首抽」、「FM2」）的保護機制，避免關鍵字被錯誤切分。
* **嚴謹的統計與語言學特徵**：
    * 萃取包含絕對化用語 (Absolutist words)、否定詞、第一人稱代名詞 (Self-focus)、未來詞 (Future words) 以及各類詞性密度（如認知動詞、情緒狀態）等指標。
    * 系統自動計算 Cohen's d 與 Cliff's delta 效果量，並執行 Mann-Whitney U 檢定，產出統計顯著性報表。
* **混合型特徵工程與降維**：引入動態 PCA (Principal Component Analysis) 將 BERT 高維度向量降維（保留 90% 變異數），確保語意特徵在隨機森林模型中不會過度主導，完美融合解釋性高的手工語言特徵。
* **處理極端不平衡資料**：採用 SMOTEENN (SMOTE + ENN) 進行過採樣與決策邊界清理，大幅改善高風險樣本極少的類別不平衡問題。
* **SHAP 模型可解釋性**：整合 SHAP (SHapley Additive exPlanations) 套件，提供 Token 等級的視覺化解釋，直觀呈現 MentalBERT 模型是如何根據特定詞彙增加或降低風險預測機率。

## 系統架構與執行流程
本專案程式碼涵蓋以下四個主要執行階段：

1. **文本正規化與斷詞 (NLP Preprocessing & Segmentation)**
   - 載入並轉換自殺防治辭典。
   - 執行 CKIP 斷詞與詞性標註，並進行領域關鍵字修復。
2. **語言學特徵工程 (Linguistic Feature Engineering)**
   - 計算文本長度、各類詞典特徵密度與特定詞性出現頻率。
   - 執行統計顯著性檢定與效果量分析。
3. **基準與融合模型建構 (Baseline & Hybrid Modeling)**
   - 訓練 Model A (手工特徵基準模型)、Model B (MentalBERT 語意模型)、Model C (融合模型)。
   - 輸出隨機森林特徵重要性 (Feature Importance) 排行與視覺化圖表。
4. **進階機器學習優化 (Advanced ML Pipeline)**
   - 實作 StandardScaler 特徵標準化。
   - 套用動態 PCA 與 SMOTEENN。
   - 產出交叉錯誤分析 (Error Analysis) 報表，比較不同模型在預測盲區的互補性。

## 環境依賴
建議使用支援 CUDA 的環境執行。核心依賴套件如下：

```bash
pip install opencc-python-reimplemented
pip install ckip-transformers
pip install xlsxwriter
pip install transformers
pip install shap
pip install imbalanced-learn
pip install scikit-learn pandas numpy matplotlib seaborn torch
