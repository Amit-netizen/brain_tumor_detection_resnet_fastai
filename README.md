# 🧠 Brain Tumor Detection with Deep Learning and FastAI

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.x-green)](https://docs.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Amit-netizen/brain_tumor_detection_resnet_fastai?style=social)](https://github.com/Amit-netizen/brain_tumor_detection_resnet_fastai)

This repository contains a deep learning solution for brain tumor detection using ResNet-based architectures and the FastAI library. The project focuses on leveraging pre-trained convolutional neural networks (CNNs) for medical image analysis to accurately classify brain MRI images. It includes code for model training, evaluation, and inference, along with detailed analysis and benchmarking.

## 🎯 Project Goals

* **Accurate Brain Tumor Detection:** To develop a robust and accurate model for identifying the presence of brain tumors in MRI images.
* **Performance Evaluation:** To systematically evaluate and compare the performance of various ResNet architectures (ResNet-18/34/50/101/152, Wide ResNet (WRN), ResNeXt-50/101).
* **FastAI Integration:** To utilize the FastAI library for efficient training, validation, and deployment of deep learning models.
* **Code Reproducibility:** To provide well-documented code and Jupyter notebooks for easy reproducibility and further research.

## 🔬 Methodology

This project employs transfer learning, a technique that leverages pre-trained models on large datasets (like ImageNet) to solve similar but smaller tasks.  Specifically, we fine-tune ResNet-based architectures on a dataset of brain MRI images. FastAI simplifies the process of training these models, handling many best practices like learning rate finding, differential learning rates, and data augmentation.

## 📊 Key Results

The project achieved high performance in brain tumor detection. Here's a summary of the results for some key models:

| Model         | Accuracy | Precision | F1-Score |
|---------------|----------|-----------|----------|
| ResNet-50     | 99.00%   | 97.88%    | 98.10%   |
| ResNet-34     | 94.05%   | 98.21%    | 96.00%   |
| ResNeXt-101   | 95.04%   | 97.30%    | 97.51%   |

**Observations:**

* **High Accuracy:** ResNet-50 demonstrated excellent accuracy (99.00%), indicating the model's ability to correctly classify MRI images.
* **Precision and F1-Score Trade-off:** Different architectures exhibit varying trade-offs between precision and F1-score. ResNet-34 achieved the highest precision (98.21%), minimizing false positives, while ResNeXt-101 showed a strong balance between precision and recall, resulting in a high F1-score (97.51%).

## 📁 Repository Contents

* **`brain_tumor_detection_resnet_fastai.ipynb`:** Jupyter Notebook containing the complete workflow for data loading, model training, evaluation, and visualization. It includes detailed explanations of each step.
* **Trained Model Logs:** (If present) Contains logs of the training process, including metrics such as loss, accuracy, precision, and recall over epochs.  Useful for analyzing model convergence and performance.
* **Evaluation Scripts:** (If present) Scripts for evaluating the trained models on a test set, generating metrics like confusion matrices, ROC curves, and classification reports.
* **Data:** (If present or specify download location) Information about the dataset used for training and testing.  If the data is too large for the repository, provide a link to download it.

## 🚀 Getting Started

To run this project, you'll need the following:

* **Python 3.6 or later:** The programming language used for the project.
* **PyTorch 1.x or later:** The deep learning framework.
* **FastAI 2.x or later:** The deep learning library that simplifies training.
* **Other Dependencies:** Libraries like NumPy, Pandas, and Matplotlib.  These are typically installed via `pip` or `conda`.

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
    pip install -r requirements.txt  # If a requirements.txt file is provided
    # OR
    pip install torch fastai numpy pandas matplotlib scikit-learn  # Install individually
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook brain_tumor_detection_resnet_fastai.ipynb
    ```

## 📈 Further Improvements

* **Data Augmentation:** Explore more advanced data augmentation techniques to improve model generalization.
* **Hyperparameter Tuning:** Fine-tune hyperparameters like learning rate, batch size, and weight decay for optimal performance.
* **Ensemble Methods:** Combine predictions from multiple models to potentially boost accuracy and robustness.
* **Explainability:** Investigate techniques to visualize and understand the model's decision-making process.
* **Deployment:** Develop a system for real-world deployment of the model.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## 🙏 Acknowledgements

* Thanks to the FastAI and PyTorch communities for providing excellent tools and resources.
* Acknowledgement to the creators of the brain MRI dataset.  (Please provide specific citation if available)
