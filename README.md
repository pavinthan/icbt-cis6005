# Kaggle House Prices - Advanced Regression Techniques
This project is based on the [House Prices Prediction - Advanced Regression Techniques competition on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

## Requirements

This project requires **Python 3** and the following Python libraries:

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

You will also need [Jupyter Notebook](https://jupyter.org/) installed to run the notebooks.

## Project Structure

The Jupyter Notebooks in this project are used for data preprocessing, feature transformation, and outlier detection.

The main scripts are located in the root directory. The `ensemble.py` script is used for ensemble learning, and the `base.py` script contains the base model. The input data is located in the `data` folder. You can find a detailed description of the data on the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## Execution

To run a single model, use the following command: 

```bash
python base_model.py
```

To perform an ensemble run, use the following command:
```bash
python ensemble.py
```

Please ensure to update the data directory and parameters as needed before running the models.