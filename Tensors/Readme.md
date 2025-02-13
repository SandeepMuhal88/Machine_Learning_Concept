# Tensors in Machine Learning

## What Are Tensors in Machine Learning?
A **tensor** is a fundamental data structure in machine learning, especially in deep learning. It is a multi-dimensional array, similar to a matrix but generalized to more dimensions. Tensors are used to store and manipulate data efficiently in ML frameworks like TensorFlow and PyTorch.

## Why Are Tensors Important?
- **Efficient Computation** – Tensors enable optimized mathematical operations using GPUs.
- **Generalization of Scalars, Vectors, and Matrices** – Tensors can represent complex data structures used in ML.
- **Support for Automatic Differentiation** – Essential for backpropagation in deep learning.

## Types of Tensors (Based on Dimensions)
| Tensor Type  | Dimensions | Example                           |
|-------------|------------|----------------------------------|
| **Scalar**  | 0D         | `5` (Single number)              |
| **Vector**  | 1D         | `[1, 2, 3]` (List of numbers)    |
| **Matrix**  | 2D         | `[[1, 2], [3, 4]]` (Table of numbers) |
| **3D Tensor** | 3D      | `[[[1,2],[3,4]], [[5,6],[7,8]]]` |
| **n-D Tensor** | nD      | Used for complex data like images and videos |

## Tensors in Machine Learning
- **Images** → Represented as 3D tensors (`height × width × channels`)
- **Videos** → 4D tensors (`frames × height × width × channels`)
- **Text Data** → Converted into tensors using embeddings (word vectors)
