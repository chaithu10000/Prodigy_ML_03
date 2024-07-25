# SVM Image Classification: Cats and Dogs

This repository provides a comprehensive implementation of a Support Vector Machine (SVM) to classify images of cats and dogs using the popular [Kaggle Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data). The project includes data preprocessing, model training, evaluation, and saving the trained model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates how to build and train a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The steps include:
1. Loading and preprocessing the dataset.
2. Extracting features from the images.
3. Training the SVM model.
4. Evaluating the model's performance.
5. Saving the trained model for future use.

## Dataset

The dataset used in this project is the [Kaggle Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data). It contains 25,000 images of cats and dogs, labeled accordingly.

## Requirements

- Python 3.7+
- NumPy
- pandas
- scikit-learn
- OpenCV
- matplotlib
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-svm-classifier.git
   cd cat-dog-svm-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

1. Download and extract the dataset from Kaggle.
2. Place the extracted dataset in the `data/` directory:
   ```
   data/
   ├── train/
   │   ├── cat.0.jpg
   │   ├── cat.1.jpg
   │   ├── dog.0.jpg
   │   ├── dog.1.jpg
   │   └── ...
   ```

### Feature Extraction

Run the feature extraction script to process the images and extract features:
```bash
python feature_extraction.py
```

### Model Training

Train the SVM model using the preprocessed data:
```bash
python train_model.py
```

### Evaluation

Evaluate the trained model on the test set:
```bash
python evaluate_model.py
```

### Saving the Model

Save the trained model for future use:
```python
import pickle

with open('model.sav', 'wb') as file:
    pickle.dump(model, file)
```

## Results

The results of the model evaluation will be displayed, including accuracy, precision, recall, and F1-score.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Code for this is attached to the repository.


