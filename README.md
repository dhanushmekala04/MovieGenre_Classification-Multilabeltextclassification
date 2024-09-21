###  **Multi-label text classification using DistilBERT model**
This project involves building a **multi-label text classification model** using the **IMDB Movie Data**, where the goal is to classify movie descriptions into multiple genres. The project demonstrates the use of **Transformer-based models** (DistilBERT) for multi-label text classification, with a focus on natural language processing (NLP). Below is an outline of the project components and how it works:

### 1. **Data Preparation:**
   - **Data Source:** The project uses a dataset containing information about movies, including their **descriptions** and **genres**. The dataset is loaded using `pandas` and basic exploration is done using `.shape`, `.info()`, and `.duplicated()` methods.
   - **Genre Extraction:** Since each movie can belong to multiple genres, the 'Genre' column is preprocessed to extract the genre labels for each movie. The genres are converted into a multi-hot encoded format using `MultiLabelBinarizer`.
   - **Description Texts:** The movie descriptions are extracted from the 'Description' column and stored as a list for further tokenization and model input.

### 2. **Data Splitting:**
   - The data is split into training and validation sets using `train_test_split`. The movie descriptions (text) serve as input, while the genre labels serve as the target labels.

### 3. **Model Selection:**
   - The model used in this project is **DistilBERT** (`distilbert-base-uncased`), a lightweight variant of BERT, specifically designed for **sequence classification tasks**.
   - **Multi-label classification:** The project involves a multi-label problem, where a movie can belong to multiple genres simultaneously. This is handled using a Sigmoid activation function instead of the Softmax typically used in single-label classification.

### 4. **Custom Dataset Class:**
   - A custom `Dataset` class is created to handle the tokenization of the movie descriptions using the **DistilBERT tokenizer**.
   - The descriptions are tokenized and converted into input tensors (`input_ids` and `attention_mask`), while the genre labels are also converted into tensors. This custom dataset is used for both training and validation data.

### 5. **Training:**
   - **HuggingFace’s Trainer API** is used to train the model. Key training arguments include:
     - Batch size (`per_device_train_batch_size`).
     - Number of training epochs (`num_train_epochs`).
     - Model saving strategy (`save_steps`, `save_total_limit`).
   - The model is fine-tuned on the training data to predict multiple genres for each movie description.

### 6. **Evaluation Metrics:**
   - **Multi-label evaluation metrics** are defined, including:
     - **F1 Score (Macro):** Measures the balance between precision and recall for multi-label classification.
     - **ROC-AUC (Macro):** Measures the performance of the model across multiple label classifications.
     - **Hamming Loss:** Measures the fraction of labels that are incorrectly predicted.
   - These metrics are computed by applying a threshold (e.g., 0.3) to the model’s predictions (which are probabilities due to the Sigmoid activation function) to determine which genres should be predicted as present or absent.

### 7. **Model Saving and Downloading:**
   - After training, the fine-tuned model and the `MultiLabelBinarizer` (used to encode and decode the genres) are saved locally.
   - The model is zipped and can be downloaded for further use.

### 8. **Running the Project Locally:**
   - The project can be run locally by:
     1. **Installing the necessary libraries:**  
        Run `pip install transformers torch pandas scikit-learn` to install required libraries.
     2. **Adjusting file paths:**  
        Update the file paths for the dataset (IMDB-Movie-Data.csv) and the model saving locations to fit your local system setup.
     3. **Running the code in a Python environment:**  
        Execute the script in a Python environment (Jupyter Notebook, PyCharm, or any Python IDE) that supports these libraries.

### 9. **Outcome:**
   - The model is trained to predict multiple genres for movie descriptions with high accuracy. Once trained, it can be used for **multi-label genre classification** of any new movie description provided as input.

---

