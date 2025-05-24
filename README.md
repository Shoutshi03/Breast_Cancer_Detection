# Breast Cancer Detection Using ResNet50 Architecture

## Project Overview
This project implements a deep learning model for breast cancer detection using the ResNet50 architecture to classify histopathology images as benign or malignant. The model leverages the Kaggle "breast-histopathology-images" dataset, achieving an accuracy of 82.4% and a ROC AUC of 0.8328. The project demonstrates the application of transfer learning and data augmentation to address challenges like class imbalance in medical imaging.

## Dataset
The project uses the [Breast Histopathology Images dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) from Kaggle, containing approximately 277,524 RGB images (50x50 pixels) from 162 breast cancer patients. Images are labeled as:
- **Benign (Class 0)**: No invasive ductal carcinoma (IDC), ~71.5% of the dataset.
- **Malignant (Class 1)**: IDC present, ~28.5% of the dataset.

The dataset is split into:
- **Training**: 70% (~194,267 images)
- **Validation**: 10% (~27,752 images)
- **Testing**: 20% (~55,505 images)

## Requirements
To run the project, install the following dependencies:
- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to set up project workflow ?



## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/breast-cancer-detection.git
   cd breast-cancer-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) and place it in the `data/` directory.

## Usage
1. **Preprocess Data**:
   Run the preprocessing script to normalize images and apply data augmentation:
   ```bash
   python scripts/preprocess.py
   ```

2. **Train Model**:
   Train the ResNet50 model using the preprocessed data:
   ```bash
   python scripts/train_model.py
   ```

3. **Evaluate Model**:
   Evaluate the trained model on the test set:
   ```bash
   python scripts/evaluate_model.py
   ```

4. **Jupyter Notebooks**:
   Explore the data preprocessing and model training steps interactively using the notebooks in the `notebooks/` directory:
   ```bash
   jupyter notebook notebooks/data_preprocessing.ipynb
   jupyter notebook notebooks/model_training.ipynb
   ```

## Model Details
- **Architecture**: ResNet50, pre-trained on ImageNet, with a custom head for binary classification.
- **Preprocessing**: Images are normalized to [0, 1], and data augmentation (rotation, flipping, zooming, shearing) is applied to the training set.
- **Training**:
  - Optimizer: Adam with a decaying learning rate.
  - Loss: Binary cross-entropy.
  - Epochs: 20.
  - Batch Size: 32.
- **Performance**:
  - Accuracy: 88.40%
  - Precision (Malignant): 0.93
  - Recall (Malignant): 0.90
  - F1-Score (Malignant): 0.91

## Results
The model effectively identifies malignant cases (high recall of 0.85), crucial for minimizing missed diagnoses. However, lower precision (0.64) for malignant cases indicates some false positives, likely due to class imbalance. The ROC AUC of 0.8328 suggests good discriminative ability.

## Challenges
- **Class Imbalance**: The dataset has 71.5% benign vs. 28.5% malignant images, potentially biasing the model.
- **Small Image Size**: 48x48 pixel images limit feature extraction.
- **Overfitting**: Mitigated through data augmentation and dropout but remains a concern.

## Future Work
- Explore advanced techniques like weighted loss functions or oversampling to address class imbalance.
- Implement attention mechanisms (e.g., CBAM) to enhance feature extraction.
- Test ensemble methods to improve accuracy.
- Validate the model in clinical settings for real-world applicability.


- [Kaggle Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *arXiv:1512.03385*.

