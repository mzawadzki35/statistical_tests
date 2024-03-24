import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstwobign
import math
from scipy.stats import poisson
from scipy.stats import chi2


def kolmogorow_smirnow_test(tested_points, theoretical_points, alpha):
    """
    Parameters
    -------
    tested_points: DataFrame
        Tablica zawierająca kolumnę ze współrzędnymi punktów testowanego rozkładu opisaną jako "X" lub "Y".
    theoretical_points: DataFrame
        Tablica zawierająca kolumnę ze współrzędnymi punktów teoretycznego rozkładu opisaną jako "X" lub "Y".
    alpha: float
        Wartość z zakresu [0,1] określająca poziom istotności.

    Returns
    -------
    l: float
        Wyliczona na podstawie próby losowej wartość statystyki lambda.
    l_alpha: float
        Wyliczona wartość statystyki lambda_alpha.
    H: int
        Wynik testu statystycznego, przyjmuje wartość:
        0 - gdy wynik testu istotności nie daje podstaw do odrzucenia H0 na rzecz H1 na poziomie istotności 1-alpha,
        1 - gdy następuje odrzucenie H0 na rzecz H1 na poziomie istotności 1-alpha.
    """
    print(f"Test Kołmogorowa-Smirnowa dla współrzędnej {tested_points.name}")
    print("H0: Testowana zmienna ma przyjęty rozkład teoretyczny")
    print("H1: Testowana zmienna nie ma przyjętego rozkładu teoretycznego")
    kl = stats.kstest(tested_points, theoretical_points)
    l = kl.statistic * math.sqrt(len(tested_points))
    l_alpha = kstwobign.isf(alpha)
    print(f"lambda = {l} lambda_alfa = {l_alpha}")
    if l < l_alpha:
        H = 0
        print("l<l_alpha")
        print(f"Odrzucenie H0 na rzecz H1 na poziomie istotności 1-alpha = {1 - alpha}")
    else:
        H = 1
        print("l>=l_alpha")
        print(
            f"Wynik testu istotności nie daje podstaw do odrzucenia H0 na rzecz H1 na poziomie istotności 1-alpha = {1 - alpha}")
    return l, l_alpha, H
def homogeneous_poisson_on_rectangle(intensity, x_lim, y_lim):
    """
    Parameters
    -------
    intensity: float
        Liczba dodatnia określająca intensywność procesu punktowego.
    x_lim: list
        Lista określająca zakres wartości współrzędnej X.
        Przykład: [0, 10]
    y_lim: list
        Lista określająca zakres wartości współrzędnej Y.
        Przykład: [0, 10]

    Returns
    -------
    points: DataFrame
        Tablica zawierająca dwie kolumny ze współrzędnymi punktów opisane jako "X" i "Y".
    """
    # YOUR CODE HERE
    n = sp.stats.poisson.rvs(intensity * (x_lim[-1] - x_lim[0]) * (y_lim[-1] - y_lim[0]))

    X = np.random.random_sample(n) * (x_lim[-1] - x_lim[0]) + x_lim[0]
    Y = np.random.random_sample(n) * (y_lim[-1] - y_lim[0]) + y_lim[0]

    d = {"X": X, "Y": Y}
    df = pd.DataFrame(data=d)
    return df


points_1 = pd.read_pickle('points_1.pkl')
points_2 = pd.read_pickle('points_2.pkl')
points_3 = pd.read_pickle('points_3.pkl')

test_data_1 = np.load("test_data_1.npy")
test_data_2 = np.load("test_data_2.npy")
test_data_3 = pd.read_pickle('test_data_3.pkl')


poisson_points=homogeneous_poisson_on_rectangle(20, [10,30], [-15, -5])
results_3X=kolmogorow_smirnow_test(points_3["X"], poisson_points["X"], 0.05)
results_3Y=kolmogorow_smirnow_test(points_3["Y"], poisson_points["Y"], 0.05)