# Amazon-ML-Challenge-2025
Developed a machine learning model to solve a large-scale real-world prediction problem provided by Amazon, involving data preprocessing, feature engineering, and model optimization. Achieved competitive leaderboard performance among thousands of teams nationwide.

**Team Name:** UNIQ  
**Team Members:** 
1. Vaasu Agarwal
2. Mohit Goyal 
3. Manan Batra
4. Akshit Pathak    

**Submission Date:** 13-Oct-2025


## 1. Executive Summary
We figure out the best price for products by first digging into all the details, from written descriptions to raw data. We find the key features that truly influence a product's value. Then, instead of relying on just one method, we use a team of different computer models to predict the price. By blending their predictions and giving more weight to the most reliable ones, we arrive at a final price that is consistently accurate and dependable.

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted the challenge as a regression problem where the key difficulty lies in extracting structured, meaningful features from unstructured catalog_content. Our Exploratory Data Analysis (EDA) focused on parsing this content into distinct fields like 'Item Name', 'Bullet Points', and 'Product Description'.

Key Observations:

The target variable, price, was heavily right-skewed, making a log transformation (log(1 + price)) essential for stabilizing variance and aligning with the SMAPE metric's relative error focus.

Critical pricing information such as pack quantity, brand name, and product dimensions/volume were embedded within the text and could be extracted using regular expressions.

Textual features, including word counts, sentence length, and TF-IDF representations of the name, bullets, and description, proved to be strong predictors of price.

Outliers in price significantly impacted performance; we handled them by clipping the training data to the 1st and 99th percentiles.

### 2.2 Solution Strategy

Our strategy prioritized feature quality over model complexity. We built a robust baseline using traditional machine learning models fed with a rich set of engineered features. The final solution is an ensemble that leverages the strengths of multiple models to improve generalization and accuracy.

Approach Type: Ensemble (Weighted Averaging)
Core Innovation: A comprehensive feature engineering pipeline combined with a performance-weighted ensemble of diverse regressors. This approach avoids the complexity of deep learning models while still capturing intricate data patterns.


## 3. Model Architecture

### 3.1 Architecture Overview
3.1 Architecture Overview
The final model is a weighted ensemble of five distinct base regressors. The predictions of these models are combined, with weights assigned based on their individual SMAPE scores (lower SMAPE gets a higher weight). This method is simple, effective, and less prone to overfitting than more complex stacking architectures.

The base models included:

Random Forest Regressor

Gradient Boosting Regressor

AdaBoost Regressor

ElasticNet

Bayesian Ridge

### 3.2 Model Components

Text Processing & Feature Engineering Pipeline:

Parsing: The raw catalog_content was parsed into structured fields: Item Name, Value, Unit, Bullet Points, and Product Description.

Feature Extraction:

Regex-based: Extracted pack_size, brand, and numeric values for size/volume from the item name.

Text Statistics: Calculated word count, character count, and average word length for the name, description, and bullet points.

Keyword Features: Counted occurrences of specific material (steel, plastic) and quality (premium, deluxe) keywords.

Categorical Encoding: Applied LabelEncoder to brand and Unit features.

Vectorization (TF-IDF):

TfidfVectorizer was used to create semantic features from the item name, bullet points, and description.

Key parameters included max_features (30-50), stop_words='english', and ngram_range=(1, 2).

Image Processing Pipeline:

Not Implemented: Based on the provided code, image features from image_link were not used in the final model. The primary focus was on maximizing the signal from the available text and tabular data.

## 4. Model Performance

### 4.1 Validation Results

The models were trained on a log-transformed target variable and evaluated on the SMAPE metric after inverse transformation. The final weighted ensemble produced the best results on the validation set.

- **SMAPE Score:** 50.1009% (Achieved by the Weighted Ensemble)


## 5. Conclusion
Our approach demonstrates that a well-crafted feature engineering pipeline combined with a robust ensemble of traditional machine learning models can achieve highly competitive performance. By focusing on extracting clear, interpretable features and blending diverse models, we created an accurate pricing solution without relying on image data or computationally expensive deep learning architectures.

## Appendix

### A. Code artefacts
The complete Python script for data processing, feature engineering, model training, and prediction generation is available.

Drive Link: https://colab.research.google.com/drive/1lwd9oRddVFq0ctC-g3kf564-3wAj9XeA?usp=sharing

