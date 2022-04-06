# KMMLDC | Kernel Methods for Machine-Learning Data Challenge (2022)

The 2022 Kernel Methods for Machine Learning Data Challenge presented participants with the task of performing image classification on 10 classes using solely kernel-based machine learning methods. In the following report, we present our results along with our intuitions and some alternative approaches we tried.\\

Our best submission achieves a classification accuracy of $57.3\%$ on the public test set. The main contribution of our work is in using a relatively fast method with closed-form solution that does not rely on an optimization solver. All experiments are reproducible and our code are available on this repository.

## Data

Data used to train the algorithm is available on [the page](https://www.kaggle.com/competitions/mva-mash-kernel-methods-2021-2022/data) of the challenge.

No additional data was used to train the models.

## Usage (reproducibility)

To reproduce these results, please start by cloning the repository locally:

```
git clone https://github.com/bglbrt/SSMNR.git
```

Then, install the required libraries:

```
pip install -r requirements.txt
```

After adding the data to the `data` folder, our best submission can be reproduced by running:

```
python main.py
```

#### Usage (options)

Additional available options are:

* `--...`:
  ...
  - default: *...*

## Required libraries

The files present on this repository require only the following libraries (also listed in requirements.txt):
 - [NumPy](https://numpy.org)
 - [Pandas](https://pandas.pydata.org)
 - [SciPy](https://scipy.org)
 - [SymPy](https://www.sympy.org/en/index.html)
