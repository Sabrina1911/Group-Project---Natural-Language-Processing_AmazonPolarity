# **Text Classification with Dimensionality Reduction**
## **TF-IDF • Naive Bayes • SVD • PCA • Logistic Regression**
### **Amazon Polarity Dataset (5000-row Balanced Subset)**
#### **Project Overview**
This project performs binary sentiment classification on a balanced subset of the Amazon Polarity dataset using three different machine learning approaches:
Dataset Source : https://huggingface.co/datasets/mteb/amazon_polarity

   - Model 1 — TF-IDF + Naive Bayes (Baseline)
   - Model 2 — TF-IDF → SVD (100 components) → Logistic Regression
   - Model 3 — TF-IDF → PCA (100 components) → Logistic Regression

**We compare all models using:**
   - Confusion matrices
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Dimensionality reduction variance curves (SVD/PCA)

**The goal is to understand:**
   - How TF-IDF encodes text
   - How dimensionality reduction affects model performance
   - Whether SVD or PCA produces better dense feature representations for classification

### **Team Member**

| **Name**                         | **Student ID**| Role                                                                   |
| -------------------------------- | --------------|------------------------------------------------------------------------|
| *Sabrina Ronnie George Karippat* | 8991911       | Model 1 (TF-IDF + Naive Bayes), preprocessing, TF-IDF, repo management |
| *Jose George*                    | 9082825       | Model 2 (SVD + Logistic Regression)                                    |
| *Aiswarya Thekkuveettil Thazhath*         | 8993970       | Model 3 (PCA + Logistic Regression)                                    |

### **Project Structure**

├── data/
│   └── amazon_polarity_5000.csv     # Balanced subset (2500 pos / 2500 neg)
│
├── notebooks/
│   └── Classification_amazonpolarity.ipynb   # Main project notebook
│
├── requirements.txt                  # Dependencies for reproducibility
├── README.md                         # Project documentation
└── .gitignore

### **Dataset Description**

We use the Amazon Polarity dataset from HuggingFace (mteb/amazon_polarity).
From the 3.6 million available reviews, we sampled:
   - 2500 positive reviews (label = 1)
   - 2500 negative reviews (label = 0)

This produced a balanced 5000-row dataset saved as:   data/amazon_polarity_5000.csv

| Column  | Description                |
| ------- | -------------------------- |
| `text`  | The Amazon review text     |
| `label` | 1 = Positive, 0 = Negative |

### **Setup Instructions**

**Clone the repository**
git clone https://github.com/Sabrina1911/Group-Project---Natural-Language-Processing_AmazonPolarity.git
cd Group-Project---Natural-Language-Processing_AmazonPolarity

### **Create and activate a virtual environment**

python -m venv venv_textclf
source venv_textclf/bin/activate     # Mac/Linux
venv_textclf\Scripts\activate        # Windows

### **Project Workflow**

**Preprocessing**
   - Lowercasing
   - Balanced sampling
   - Missing value checks
   - 75/25 train-test split
   - TF-IDF vectorization (max_features=5000)

**Model 1 — TF-IDF + Naive Bayes**
   - MultinomialNB classifier
   - Confusion matrix + metrics
   - Strong baseline performance

**Model 2 — SVD + Logistic Regression**
   - TruncatedSVD (100 components)
   - Cumulative explained variance curve
   - Logistic Regression classifier
   - Confusion matrix + metrics

**Model 3 — PCA + Logistic Regression**
   - StandardScaler (with_mean=False)
   - PCA (100 components)
   - Cumulative explained variance curve
   - Logistic Regression classifier
   - Confusion matrix + metrics

**Final Comparison**

A combined metric table is generated:
| Model                | Accuracy | Precision | Recall | F1-Score |
| -------------------- | -------- | --------- | ------ | -------- |
| TF-IDF + Naive Bayes | ~0.834   | ~0.846    | ~0.818 | ~0.832   |
| SVD + LR             | ~0.802   | ~0.814    | ~0.782 | ~0.798   |
| PCA + LR             | ~0.790   | ~0.794    | ~0.784 | ~0.789   |

A side-by-side confusion matrix visualization compares error patterns across all models.

### **Final Conclusion**
   - Model 1 — TF-IDF + Naive Bayes achieved the best overall performance.
   - SVD + Logistic Regression preserved much of the TF-IDF information but did not outperform Naive Bayes.
   - PCA + Logistic Regression performed slightly below SVD, reflecting PCA’s variance-focused rather than semantic-focused component extraction.

*Final takeaway:*

Sparse TF-IDF + Naive Bayes remains a highly effective baseline for text classification, even when compared to dimensionality-reduction-based models.

### **Dependencies**

See requirements.txt.

Typical stack includes:

numpy
pandas
scikit-learn
matplotlib
seaborn
datasets   # HuggingFace

### **How to Run the Project**
   - Install environment → pip install -r requirements.txt
   - Open the notebook → run cells top-to-bottom
   - All models will execute and generate:
     * TF-IDF matrices
     * SVD variance plot
     * PCA variance plot
     * Three confusion matrices
     * Final comparison table
     * Side-by-side visualization

## *Best model: TF-IDF + Naive Bayes (Accuracy 83.44%, F1-score 83.16%).*


Reference:

Massive Text Embedding Benchmark (MTEB). (n.d.). amazon_polarity [Dataset]. Hugging Face. Retrieved December 9, 2025, from https://huggingface.co/datasets/mteb/amazon_polarity.