# Handwritten Digit Recognition with a Neural Network

## What is this project?
This project is a neural network I built from scratch to recognize handwritten digits (0-9) from the MNIST dataset. It can correctly identify about 85% of the digits in the test data! I used only Python, NumPy, and Pandas, without any fancy machine learning libraries like TensorFlow.
and it shows my ability to understand and code the math behind neural networks, solve problems, and write clean Python code.


## What’s the MNIST Dataset?
The [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) is a collection of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a 28x28 pixel grayscale picture of a number from 0 to 9. The data comes in CSV files, which I loaded using Pandas to make it easy to work with.

## Why is this project cool?
- **Built from Scratch**: I coded the entire neural network myself, including how it learns and makes predictions, without using pre-made tools. This helped me understand the math and logic behind neural networks.
- **Simple Tools**: I only used NumPy for math calculations and Pandas for handling data, keeping things lightweight.
- **My Learning Journey**: This project shows how I tackled a complex problem, learned new concepts, and built something that works!
- **Skills Showcased**:
  - Writing clear Python code.
  - Understanding how neural networks learn (like how they adjust to get better).
  - Solving problems when things didn’t work at first.
  - Working with data to prepare it for the model.

## What’s inside the neural network?
Think of the neural network like a brain with layers of “neurons”:
- **Input Layer**: Takes the 784 pixels (28x28) from each image.
- **Hidden Layers**: Processes the data using ReLU (a math function that helps the network learn patterns).
- **Output Layer**: Gives a guess for which digit (0-9) the image is, using softmax (a way to pick the most likely answer).
- **Learning Process**: The network learns by trying, making mistakes, and adjusting its “weights” (like tuning a musical instrument) using a method called stochastic gradient descent (SGD) and a loss function called cross-entropy.

I built all this step-by-step, which was challenging but super rewarding!

## What you need to run it
- Python 3.8 or newer
- NumPy (for math stuff): `pip install numpy`
- Pandas (for handling data): `pip install pandas`

## Results
- **Accuracy**: ~85% on the test data.
- **What I Learned**: I figured out how to make the network learn better by tweaking things like the learning speed and fixing issues like numbers getting too big during calculations.
- **Challenges**: It wasn’t easy! I had to learn how to balance the network’s settings to get good results without making it too slow.

## Skills I’m Showing Off
This project is proof of what I can do:
- **Coding**: I wrote clean, organized Python code that’s easy to follow.
- **Math**: I understood and coded the math behind neural networks, like how they learn from mistakes.
- **Problem-Solving**: I fixed bugs and improved the network when it wasn’t working well.
- **Learning**: I taught myself how to build this without relying on pre-made libraries, which was a big challenge!


## Get in touch
Have questions or want to chat about the project? Reach out on [GitHub](https://github.com/mostafa-ghaedi) or email me at mustafa.ghaedi@gmail.com.
