# 🧠 Brain Tumor Detection with Deep Learning and FastAI

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.x-green)](https://docs.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Amit-netizen/brain_tumor_detection_resnet_fastai?style=social)](https://github.com/Amit-netizen/brain_tumor_detection_resnet_fastai)

This repository implements a deep learning solution for brain tumor detection in MRI images using ResNet-based architectures and the FastAI library. The project leverages transfer learning to accurately classify MRI scans, distinguishing between images with and without tumors. It provides a complete workflow from data loading and preprocessing to model training, evaluation, and visualization.

## 🎯 Project Goals

* **Accurate Brain Tumor Detection:** Develop a high-accuracy model for binary classification of brain MRI images (tumor vs. no tumor).
* **Efficient Training with FastAI:** Utilize FastAI's high-level API to streamline model training, validation, and inference.
* **Model Evaluation and Interpretation:** Thoroughly evaluate model performance using appropriate metrics and visualize results for better understanding.
* **Reproducible Research:** Provide a well-organized and documented Jupyter Notebook for reproducibility and further experimentation.

## 🔬 Methodology

The core methodology involves:

1.  **Data Loading and Preprocessing:** Loading MRI image data and preparing it for model training using FastAI's `ImageDataLoaders`.
2.  **Transfer Learning with ResNet:** Fine-tuning pre-trained ResNet architectures (specifically, ResNet18 in the provided notebook) on the brain MRI dataset. Transfer learning significantly accelerates training and improves performance by leveraging knowledge learned from large datasets like ImageNet.
3.  **Model Training with FastAI:** Utilizing FastAI's `Learner` class to train the model, including techniques like learning rate finding and optimization.
4.  **Model Evaluation:** Evaluating the trained model's performance using metrics like accuracy, precision, recall, and F1-score.
5.  **Visualization:** Visualizing the model's predictions and performance through confusion matrices.

## 📊 Key Results

The notebook demonstrates excellent results, achieving near-perfect accuracy on the validation set.

* **Accuracy:** The model achieved approximately 100% accuracy on the validation set in the provided notebook run.

**Note:** While the initial README provided results for different ResNet architectures, the current notebook primarily focuses on ResNet18. The high accuracy achieved suggests the dataset might be relatively simple, or the model has learned the patterns very effectively. Further investigation with a more diverse and challenging dataset is recommended for real-world applications.

## 📁 Repository Contents

* **`brain_tumor_detection_resnet_fastai.ipynb`:** Jupyter Notebook containing the complete deep learning workflow:
    * Data loading and preprocessing
    * Model definition and training (ResNet18)
    * Model evaluation (accuracy, confusion matrix)
    * Results visualization
    * Detailed explanations and code comments
* **`Data/`:** (If present, or specify download) Directory that should contain the MRI image dataset.  If the dataset is not in the repository due to size limitations, clear instructions or a download link will be provided here.

## 🚀 Getting Started

To run this project, you'll need the following:

* **Python 3.7 or later:** The programming language used for the project.
* **PyTorch 1.x or later:** The deep learning framework.
* **FastAI 2.x or later:** The deep learning library that simplifies training.
* **Dependencies (Install with pip or conda):**
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `scikit-learn`

**Installation Steps:**

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Amit-netizen/brain_tumor_detection_resnet_fastai.git](https://github.com/Amit-netizen/brain_tumor_detection_resnet_fastai.git)
    cd brain_tumor_detection_resnet_fastai
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt  # If a requirements.txt file exists
    # OR
    pip install torch fastai numpy pandas matplotlib scikit-learn
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook brain_tumor_detection_resnet_fastai.ipynb
    ```

## 📈 Further Improvements

* **Dataset Exploration:** Analyze the dataset more thoroughly (class balance, image variations) to understand its characteristics and potential challenges.
* **Data Augmentation:** Implement more robust data augmentation techniques to improve model generalization and reduce overfitting.
* **Hyperparameter Tuning:** Systematically tune hyperparameters (learning rate, batch size, etc.) to optimize model performance.
* **Architectural Exploration:** Experiment with different ResNet architectures (ResNet34, ResNet50, etc.) and other CNN models.
* **Advanced Evaluation:** Perform more comprehensive evaluation using techniques like cross-validation and ROC curve analysis.
* **Explainability:** Use visualization techniques (e.g., Grad-CAM) to understand which parts of the MRI images the model focuses on for prediction.

## 🤝 Contributing

Contributions are welcome!  Please feel free to submit pull requests or open issues for bug fixes, improvements, or new features.

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## 🙏 Acknowledgements

* Thanks to the FastAI and PyTorch communities for providing excellent tools and resources.
* Acknowledgement to the creators of the brain MRI dataset. (Please provide the specific citation if available).
