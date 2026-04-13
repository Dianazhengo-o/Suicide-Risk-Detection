# Suicide-Risk-Detection

# 繁體中文社群之自殺風險可解釋性偵測模型

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uh5zJflIHOCRdso0Fn98Gx5okrqNP84_?usp=sharing)

## 專案簡介
於非結構化的社群文本精準區分「即時自殺風險（High Risk）」與「一般憂鬱情緒（General Depressive Affect）」是一大挑戰。

本專案針對台灣社群（如 PTT Prozac 板）的文本特性，建構了一個強調高度可解釋性 (Interpretability) 的機器學習框架。有別於傳統的黑盒子深度學習模型，本研究試驗了 `Chinese-MentalBERT` 的深層語意表徵與多項手工萃取的語言學特徵，不僅提升預測準確度，也驗證了語言學特徵能夠帶來更多的可解釋性。

## 特色
* **在地化 NLP 前處理**：整合 `OpenCC` 進行辭典繁簡轉換，並使用 `ckip-transformers` 進行斷詞與詞性標註（POS Tagging）。系統內建針對在地社群用語（如：「登出人生」、「重刷首抽」、「FM2」）的保護機制，避免關鍵字被錯誤切分。
* **嚴謹的統計與語言學特徵**：
    * 萃取包含絕對化用語 (Absolutist words)、否定詞、第一人稱代名詞 (Self-focus)、未來詞 (Future words) 以及各類詞性密度（如認知動詞、情緒狀態）等指標。
    * 系統自動計算 Cohen's d 與 Cliff's delta 效果量，並執行 Mann-Whitney U 檢定，產出統計顯著性報表。
* **混合型特徵工程與降維**：引入動態 PCA (Principal Component Analysis) 將 BERT 高維度向量降維（保留 90% 變異數），確保語意特徵在隨機森林模型中不會過度主導，融合解釋性高的手工語言特徵。
* **處理極端不平衡資料**：採用 SMOTEENN (SMOTE + ENN) 進行過採樣與決策邊界清理，大幅改善高風險樣本極少的類別不平衡問題。
* **SHAP 模型可解釋性**：整合 SHAP (SHapley Additive exPlanations) 套件，提供視覺化解釋，直觀呈現 MentalBERT 模型是如何根據特定詞彙增加或降低風險預測機率。

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

## 優化後模型結果

為了改善原始模型在類別不平衡、高維語意特徵與泛化能力上的限制，本研究在進階版本中加入了三項前處理：手工特徵標準化（StandardScaler）、MentalBERT 特徵的動態 PCA 降維（保留 90% 變異量），以及針對訓練集進行 SMOTEENN 過採樣與邊界雜訊清理。 
此外，隨機森林模型也進一步調整為較保守的設定，包括 `n_estimators=300`、`max_depth=15`、`min_samples_split=5`，並在融合模型中額外設定 `max_features=0.15`，以降低高維特徵主導風險並抑制過度擬合。

在優化後的結果中，三個模型對高風險類別的偵測能力皆有不同程度改善，顯示前處理與重採樣策略確實改變了模型的決策偏好。
其中，Model B（MentalBERT + PCA）在高風險類別上取得最佳 F1-score 0.42，precision 為 0.32，recall 為 0.62，整體 accuracy 為 0.86，為三者中兼顧整體表現與高風險辨識能力的最佳平衡點。 
Model C（融合模型）則將高風險類別 recall 進一步提升至 0.71，顯示融合特徵能更積極地捕捉高風險樣本，但其 precision 降至 0.28，accuracy 亦下降至 0.82，反映出更高召回率伴隨更多誤報的 trade-off。 
相較之下，Model A（手工特徵模型）在優化後的高風險類別 recall 為 0.42、F1-score 為 0.30，整體 accuracy 為 0.84，顯示單靠可解釋特徵雖能提供穩定基線，但在少數類別辨識上仍不如語意模型。

整體而言，優化後版本的主要進步不在於單純追求更高 accuracy，而是成功將模型由偏向多數類別的保守預測，調整為更願意辨識高風險樣本的模式。 
若以高風險類別的 recall 作為重點指標，Model B 與 Model C 分別提升至 0.62 與 0.71，顯示加入 PCA、SMOTEENN 與參數調整後，模型對少數類別的敏感度明顯提升。  
因此，本研究認為優化策略雖未完全解決小樣本與類別不平衡的根本限制，但已有效改善高風險樣本的捕捉能力，並提供了後續模型設計與資料擴充的重要方向。

### 優化後模型比較

| 模型 | 方法 | Accuracy | High Risk Precision | High Risk Recall | High Risk F1 |
|---|---|---:|---:|---:|---:|
| Model A | 手工特徵 + StandardScaler + SMOTEENN + RF  | 0.84  | 0.24 | 0.42  | 0.30  |
| Model B | MentalBERT + PCA + SMOTEENN + RF  | 0.86 | 0.32 | 0.62  | 0.42 |
| Model C | Manual + BERT 融合 + PCA + SMOTEENN + RF  | 0.82 | 0.28  | 0.71  | 0.40  |

### 結果解讀

- Model B 在整體表現與高風險偵測之間取得最佳平衡，可視為本研究優化後最穩定的模型。
- Model C 具有最高的高風險召回率，適合用於較重視「降低漏判」的情境。
- Model A 雖然可解釋性較高，但在少數類別辨識上仍有明顯限制。
- 這些結果說明，優化策略成功讓模型從「整體準確但忽略高風險」轉向「更能主動捕捉高風險樣本」的方向。

## 資料隱私與學術倫理聲明

本研究所分析之語料原始來源雖為公開之網路社群（PTT Prozac 板），但文本內容涉及高度敏感之個人心理健康狀態、負面情緒傾訴與潛在的急性自殺意念。
將此類帶有極端負面情緒的零散文章進行系統性蒐集與心理學量表標註後，資料集具備極高的敏感性，為防止對發文者造成二次傷害，**本專案不公開原始文本資料集**。

本專案所附之 Colab 執行連結，其唯一目的為展示研究方法之可行性，並供檢視流程與執行結果以驗證真實性。所有公開之數據僅限於量化統計指標與特徵重要性分析，任何可能識別出特定使用者之文本皆已被嚴格阻擋與排除。

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




