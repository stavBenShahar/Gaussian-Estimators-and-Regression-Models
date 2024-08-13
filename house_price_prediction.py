from typing import NoReturn, Optional
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
from pygments.lexers import go
import plotly.express as px

from IMLearn.learners.regressors import LinearRegression as LR
from IMLearn.metrics.loss_functions import mean_square_error as mse
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def remove_invalid_values(X: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values
    X = X.dropna()
    X = X.astype(float)

    # define bad values
    pos_cond = (X["sqft_living"] > 0) & (X["sqft_lot"] > 0)
    non_neg_cond = (X["sqft_above"] >= 0) & (X["sqft_basement"] >= 0)
    floor_cond = X["floors"].isin(range(1, 5))
    bed_cond = X["bedrooms"].isin(range(1, 20))
    bath_cond = X["bathrooms"].isin(range(1, 20))
    wf_cond = X["waterfront"].isin([0, 1])
    view_cond = X["view"].isin(range(5))
    cond_cond = X["condition"].isin(range(1, 6))
    grade_cond = X["grade"].isin(range(1, 15))

    # Apply conditions
    X = X[pos_cond & non_neg_cond & floor_cond & bed_cond & bath_cond & wf_cond & view_cond & cond_cond & grade_cond]
    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Remove NaN from the DF and duplicates that will affect the results badly
    X = X.drop_duplicates()
    # Remove features who don't benefit the results
    X = X.drop(["id", "date", "lat", "long", "sqft_lot15", "sqft_living15"], axis=1)
    # Remove bad samples who have invalid values:
    X = remove_invalid_values(X)
    # It's not beneficial information if a house was renovated 3 decades ago or more
    current_year = datetime.datetime.now().year
    X["recently_renovated"] = ((current_year - X["yr_renovated"]) <= 30).astype(int)
    X = X.drop("yr_renovated", axis=1)
    # One hot encoding for Zipcode:
    X = pd.get_dummies(X, columns=['zipcode'])
    # Remove from the response vector samples who were removed from the DataFrame

    if y is not None:
        y = y.loc[X.index]
        return X, y
    else:
        return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.filter(regex='^(?!zipcode_|decade_built_)')

    for feature in X.columns:
        plt.scatter(X[feature], y)
        coef = pearson_correlation(X[feature], y)
        plt.title(f"{feature} (corr: {coef:.2f})")
        plt.xlabel(feature)
        plt.ylabel("Response")
        plt.savefig(f"{output_path}/{feature}_scatter.png")


def pearson_correlation(feature_col: pd.Series, response: pd.Series) -> float:
    result = np.cov(feature_col, response)[0, 1] / (np.std(feature_col) * np.std(response))
    return result


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - split data into train and test sets
    X = pd.read_csv("../datasets/house_prices.csv")
    price = X.pop("price")
    train_X, train_y, test_X, test_y = split_train_test(X, price)
    # Question 2 - Preprocessing of housing prices datase
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X, test_y = preprocess_data(test_X, test_y)
    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X,train_y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = list(range(10, 101))
    results = np.zeros((len(percentages), 10))

    for i, p in enumerate(percentages):
        for j in range(10):
            sample_X = train_X.sample(frac=p / 100)
            sample_y = train_y.loc[sample_X.index]
            model = LR(include_intercept=True).fit(sample_X, sample_y)
            mse = model.loss(test_X, test_y)
            results[i, j] = mse

    means = np.mean(results, axis=1)
    stds = np.std(results, axis=1)

    plt.errorbar(percentages, means, yerr=2 * stds)
    plt.xlabel("Percentage of training data used")
    plt.ylabel("Mean Squared Error")
    plt.title("Effect of training data size on prediction error")
    plt.show()
