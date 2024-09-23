# Multi-label Text Classification Using DistilBERT

## Project Overview
This project involves building a **multi-label text classification model** using the **IMDB Movie Data**, where the goal is to classify movie descriptions into multiple genres. The project demonstrates the use of **Transformer-based models** (DistilBERT) for multi-label text classification, focusing on natural language processing (NLP).

## 1. Data Preparation
- **Data Source**: The dataset contains information about movies, including their **descriptions** and **genres**. It is loaded using `pandas`, and basic exploration is performed using methods like `.shape`, `.info()`, and `.duplicated()`.
- **Genre Extraction**: The 'Genre' column is preprocessed to extract genre labels for each movie. Genres are converted into a multi-hot encoded format using `MultiLabelBinarizer`.
- **Description Texts**: Movie descriptions are extracted from the 'Description' column and stored in a list for tokenization and model input.

## 2. Data Splitting
The dataset is split into training and validation sets using `train_test_split`. Movie descriptions (text) serve as input, while genre labels are the target labels.

## 3. Model Selection
The model used is **DistilBERT** (`distilbert-base-uncased`), a lightweight variant of BERT, specifically designed for **sequence classification tasks**. Multi-label classification is handled using a Sigmoid activation function instead of the Softmax typically used in single-label classification.

## 4. Custom Dataset Class
A custom `Dataset` class is created to manage the tokenization of movie descriptions using the **DistilBERT tokenizer**. Descriptions are tokenized into input tensors (`input_ids` and `attention_mask`), while genre labels are converted into tensors. This custom dataset is utilized for both training and validation data.

## 5. Training
- **HuggingFace’s Trainer API** is employed to train the model. Key training arguments include:
  - Batch size (`per_device_train_batch_size`)
  - Number of training epochs (`num_train_epochs`)
  - Model saving strategy (`save_steps`, `save_total_limit`)
- The model is fine-tuned on the training data to predict multiple genres for each movie description.

## 6. Evaluation Metrics
**Multi-label evaluation metrics** are defined, including:
- **F1 Score (Macro)**: Measures the balance between precision and recall for multi-label classification.
- **ROC-AUC (Macro)**: Evaluates model performance across multiple label classifications.
- **Hamming Loss**: Measures the fraction of incorrectly predicted labels.

These metrics are computed by applying a threshold (e.g., 0.3) to model predictions (probabilities from the Sigmoid activation function) to determine which genres should be predicted as present or absent.

## 7. Model Saving and Downloading
After training, the fine-tuned model and the `MultiLabelBinarizer` (used to encode and decode genres) are saved locally. The model is zipped for easy download and further use.

## 8. Running the Project Locally
The project can be run locally by following these steps:
1. **Install Necessary Libraries**:  
   Run the following command to install required libraries:  
   ```bash
   pip install transformers torch pandas scikit-learn
   ```
2. **Adjust File Paths**:  
   Update the file paths for the dataset (IMDB-Movie-Data.csv) and model saving locations to fit your local system setup.
3. **Execute the Code**:  
   Run the script in a Python environment (Jupyter Notebook, PyCharm, or any Python IDE) that supports these libraries.

## 9. Outcome
The model is trained to predict multiple genres for movie descriptions with high accuracy. Once trained, it can classify any new movie description provided as input.

### Sample Output
![Sample Output](https://github.com/dhanushmekala04/MovieGenre_Classification-Multilabeltextclassification/blob/main/Screenshot%202024-09-23%20155949.png)

The output demonstrates the model's ability to accurately classify a movie description into multiple genres, reflecting its effectiveness in multi-label text classification.

## Conclusion
This project illustrates the application of DistilBERT for multi-label text classification, achieving reliable genre predictions for movie descriptions. The trained model is suitable for practical applications in film recommendation systems and content tagging.

