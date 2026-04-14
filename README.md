# Data Science for the Linear Algebraist

A practical, linear-algebra-first introduction to data science.

This repository demonstrates how core linear algebra concepts -- least squares, matrix decompositions, and spectral methods -- directly power modern data science and machine learning workflows. We finish off with a mini-project involving image denoising using the truncated SVD.

Rather than treating data science as a collection of tools, this project builds everything from first principles and connects theory to implementation through jupyter notebooks. 

The compiled notebooks in this project can be viewed as a single webpage on my [website](https://pawelsarkowicz.xyz/posts/ds_for_la). Note that if you view in the notebooks in Gitlab/Github, they have a tendency to not render the latex properly. 


## Structure

This project is organized as a collection of focused notebooks:

```text
images/           # saved images/visualizations
notebooks/        # jupyter notebooks containing theory, code, visuals
bibliography.md   # references for essentially everything
requirements.txt  # python requirements
LICENSE           # project license
```

Each notebook is self-contained and moves from theory to implementation to visualization.

## Dependencies

* **Python 3**
* **NumPy** -- linear algebra
* **Pandas** -- data handling
* **Matplotlib** -- visualization
* **Pillow** -- imaging library
* **scikit-learn** -- machine learning utilities

## How to Run

```bash
git clone https://gitlab.com/psark/ds-for-la.git
cd ds-for-la

pip install requirements.txt

jupyter notebook
```

Open any notebook inside the `notebooks/` folder.

---

## Topics

### 1. Least Squares Regression

* Overdetermined systems
* Normal equations
* Geometric interpretation (projection onto column space)
* Implementation using NumPy

### 2. QR Decomposition & SVD

* Numerical stability vs. normal equations
* Orthogonal bases and conditioning
* Solving linear systems without forming $X^T X$

### 3. Some Notes & What Can Go Wrong

* Other vector norms ($L^1, L^\infty$), as well as matrix norms (Frobenius, Operator)
* What can go wrong?

### 4. Principal Component Analysis (PCA)

* Dimensionality reduction via spectral methods
* Relationship between covariance matrices and eigenvectors
* Handling correlated features

### 5. Project: Spectral Image Denoising via Truncated SVD

* Low-rank approximation of images
* Noise removal using singular value truncation
* RGB images (channel-wise SVD)
* Quantitative evaluation (MSE, PSNR)

### 6. Modelling 101

* Train/test splits
* Regression (Linear, Ridge, LASSO, PCR)
* Gradient descent
* Decision trees & random forests
* Logistic regression
* Cross-validation
* Feature scaling
* Hyperparameter tuning

---

## Example: Image Denoising via SVD

Given an image matrix $A$ (for simplicity, let's go with greyscale), we compute its singular value decomposition:

$$
A = U \Sigma V^T
$$

We approximate the image using only the top $k$ singular values:

$$
A_k = U_k \Sigma_k V_k^T
$$

This produces:

* **Noise reduction**
* **Compression**
* A direct application of the **Eckart–Young–Mirsky theorem**

For color images, this is applied independently to each channel (R, G, B).

---

## Key Takeaways

* Data science problems can be framed as:

  > *approximate solutions to linear systems*

* Numerical linear algebra is necessary; it determines:

  * stability
  * performance
  * model reliability

* Spectral methods (SVD, PCA) provide:

  * structure
  * compression
  * interpretability

* Regularization connects directly to linear algebra:
  * Ridge shifts singular values, improving condition number
  * Lasso exploits $L^1$ geometry to product sparse solutions

* Gradient descent convergence is governed by singular value structure
  * Condition number determines learning rate stability
  * Feature scaling reshapes the optimization landscape


---

## Purpose

This project is part of a broader effort to translate a background in pure mathematics into practical data science and machine learning skills.


## Future Work

- ~~Add regularization (Ridge, LASSO)~~
- Extend PCA to real datasets
- Compare SVD vs. autoencoders for compression
- Add performance benchmarks (QR vs SVD vs normal equations)
- Add neural networks

---

# License

This project is licensed under the MIT License.
See the [`LICENSE`](./LICENSE) file for details.