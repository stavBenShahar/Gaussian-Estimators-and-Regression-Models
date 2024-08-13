from matplotlib import pyplot as plt

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    min_temp = -72  # Remove all the redundant -72 from the DF
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df[df['Temp'] > min_temp]
    df = df.dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def explore_data_for_country(country: str, df: pd.DataFrame):
    country_data = df[df["Country"] == country]
    fig1, ax1 = plt.subplots()
    ax1.scatter(country_data["DayOfYear"], country_data["Temp"], c=country_data["Year"])
    ax1.set_xlabel("Day of Year")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Daily Temperatures in Israel")
    fig1.savefig("israel_daily_temperatures.png")

    monthly_std = country_data.groupby("Month")["Temp"].std().reset_index()
    fig2 = px.bar(monthly_std, x="Month", y="Temp", title="Temperature Standard Deviation by Month in Israel")
    fig2.update_xaxes(title="Month")
    fig2.update_yaxes(title="Temperature Standard Deviation (°C)")
    fig2.write_image("israel_monthly_temperature_std.png")


def fit_model_up_to_k(t: int, data: pd.DataFrame):
    train_X, train_y, test_X, test_y = split_train_test(data.DayOfYear, data.Temp)
    ks = list(range(1,t))
    test_loss = []
    for k in ks:
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        mse = np.round(model.loss(test_X, test_y),2)
        test_loss.append(mse)

    loss_df = pd.DataFrame({"k": ks, "test_loss": test_loss})
    fig = px.bar(loss_df, x="k", y="test_loss", text="test_loss", title="Test Error for Different Values of k")
    fig.write_image("test_lost_different_values_model.png")
    print(loss_df)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    explore_data_for_country("Israel", df)

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")),
            x="Month", y="mean", error_y="std", color="Country") \
        .update_layout(title="Average Monthly Temperatures",
                       xaxis_title="Month",
                       yaxis_title="Mean Temperature") \
        .write_image("average_monthly_temperature.png")

    # Question 4 - Fitting model for different values of `k`
    israel_data = df[df["Country"] == "Israel"]
    fit_model_up_to_k(10, israel_data)

    # Question 5 - Evaluating fitted model on different countries

    # The smallest loss was for the polynomial model was for k=5
    model = PolynomialFitting(5).fit(israel_data.DayOfYear.to_numpy(), israel_data.Temp.to_numpy())
    test_errors = []
    for country in ["Jordan", "South Africa", "The Netherlands"]:
        country_df = df[df.Country == country]
        test_error = model.loss(country_df.DayOfYear, country_df.Temp)
        test_errors.append(round(test_error, 2))

    px.bar(x=["Jordan", "South Africa", "The Netherlands"], y=test_errors,
           text=test_errors, color=["Jordan", "South Africa", "The Netherlands"],
           title="Test Error of Model Fitted Over Israel").write_image("test_error_of_different_countries.png")
