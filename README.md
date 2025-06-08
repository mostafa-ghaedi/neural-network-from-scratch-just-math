# Neural Network from Scratch for MNIST Digit Recognition

## Overview
This project implements a fully functional neural network from scratch to classify handwritten digits from the MNIST dataset, achieving an accuracy of approximately 85%. Built using only NumPy for matrix operations and Pandas for data handling, this project demonstrates a deep understanding of neural network fundamentals, including forward propagation, backpropagation, and gradient-based optimization. By avoiding deep learning frameworks like TensorFlow or PyTorch, the implementation showcases my ability to translate mathematical concepts into efficient code, highlighting skills in Python programming, machine learning, and problem-solving.

The goal of this project is to not only solve the digit classification task but also to document my mastery of neural network theory and practical implementation, making it a valuable addition to my portfolio.

## Dataset
The [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) from Kaggle consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9), each represented as a 28x28 grayscale image. The data is provided in CSV format, making it easy to process with Pandas.

## Features
- **From-Scratch Implementation**: A fully custom-built neural network with manual forward and backward propagation, coded without reliance on deep learning libraries.
- **Multi-Layer Perceptron (MLP)**: Comprises an input layer (784 neurons for 28x28 pixels), multiple hidden layers with ReLU activation, and an output layer with softmax for 10-class classification.
- **Optimization**: Utilizes stochastic gradient descent (SGD) with cross-entropy loss to update weights.
- **Demonstrated Skills**:
  - Deep understanding of neural network mathematics, including forward/backward propagation, gradient computation, and loss functions.
  - Proficient Python programming with clean, modular code.
  - Data preprocessing and management using Pandas.
  - Problem-solving through hyperparameter tuning and performance optimization.

## Prerequisites
- Python 3.8 or higher
- NumPy (`pip install numpy`)
- Pandas (`pip install pandas`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-neural-network.git
   cd mnist-neural-network
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place `mnist_train.csv` and `mnist_test.csv` in the project directory.

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Evaluate the model on the test set:
   ```bash
   python evaluate.py
   ```
3. The model outputs predictions and achieves approximately 85% accuracy on the test set.

## Code Structure
- `train.py`: Handles data preprocessing, model training, and implementation of forward/backward propagation.
- `evaluate.py`: Loads the trained model and evaluates its performance on the test set.
- `utils.py`: Contains helper functions for activation functions (ReLU, softmax), loss computation, and gradient calculations.
- `data/`: Directory for storing MNIST CSV files.

## Results
- **Accuracy**: ~85% on the MNIST test set.
- **Training Time**: Approximately 10-15 minutes on a standard CPU (varies by hardware).
- **Loss**: Cross-entropy loss decreases consistently during training.
- **Key Challenges Overcome**:
  - Managed numerical stability issues, such as gradient overflow, through careful implementation.
  - Optimized hyperparameters (e.g., learning rate, batch size) to balance accuracy and training speed.
  - Designed an efficient MLP architecture to achieve high accuracy with minimal computational resources.

## Skills Demonstrated
This project highlights the following competencies:
- **Mathematical Understanding**: Manual implementation of forward/backward propagation, gradient computation, and loss optimization.
- **Programming Proficiency**: Clean, modular, and efficient Python code using NumPy for matrix operations.
- **Problem-Solving**: Addressed challenges like tuning hyperparameters and handling large datasets.
- **Independent Learning**: Developed a complex neural network without relying on pre-built frameworks, showcasing a deep grasp of machine learning concepts.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, such as enhancing accuracy or adding visualization features.

## Contact
For questions or feedback, reach out via [GitHub](https://github.com/yourusername) or email at your.email@example.com.

## License
This project is licensed under the MIT License.