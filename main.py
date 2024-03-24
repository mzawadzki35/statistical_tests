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

def point_count_on_subregions(points, bins, x_lim, y_lim):
    """
    Parameters
    -------
    points: DataFrame
        Tablica zawierająca dwie kolumny ze współrzędnymi punktów opisane jako "X" i "Y".
    bins: list
        Lista określająca liczbę podobszarów w poziomie i pionie.
        Przykład: [10, 10]
    x_lim: list
        Lista określająca zakres wartości współrzędnej X.
        Przykład: [0, 10]
    y_lim: list
        Lista określająca zakres wartości współrzędnej Y.
        Przykład: [0, 10]
    Returns
    -------
    bin_data: list
        Lista zawierająca trzy macierze:
        - 2D z liczbą punków przypisanych do każdego z podobszarów.
        - 1D ze współrzędnymi krawędzi podobszarów na osi X,
        - 1D ze współrzędnymi krawędzi podobszarów na osi Y,
        Na przykład: [array([[7, 2], [4, 5]]), array([0, 1, 2]), array([0, 1, 2])]
    """
    gridx = np.linspace(x_lim[0], x_lim[1], bins[0] + 1)
    gridy = np.linspace(y_lim[0], y_lim[1], bins[1] + 1)
    grid, _, _ = np.histogram2d(points["X"], points["Y"], bins=[gridx, gridy])
    bin_data = [grid.T, gridx, gridy]
    return bin_data


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


def distribution_table(bin_counts):
    """
    Parameters
    -------
    bin_counts: array
        Macierz 2D z liczbą punków przypisanych do każdego z podobszarów.

    Returns
    -------
    table: DataFrame
        Tablica zawierająca 2 kolumny:
        - "K", która zawiera wszystkie wartości całkowite z zakresu od minimalnej do maksymalnej liczby zliczeń w obrębie podobszarów,
        - "N(K)", która zawiera liczby podobszarów, którym zostały przypisane poszczególne liczby punktów.
    """

    # YOUR CODE HERE
    N = []
    K = np.linspace(np.min(bin_counts), np.max(bin_counts), int(np.max(bin_counts) - np.min(bin_counts) + 1))
    for i in K:
        N.append(np.count_nonzero(bin_counts == i))

    df = {"K": K, "N(K)": N}

    return pd.DataFrame(df)

    # raise NotImplementedError()


def poisson_distribution_table(k, mu):
    """
    Parameters
    -------
    k: array
        Macierz 1D z wariantami badanej cechy, dla którym ma zostać wyliczone prawdopodobieństwo.
    mu: int
        Wartość oczekiwana rozkładu Poissona.

    Returns
    -------
    table: DataFrame
        Tablica zawierająca 2 kolumny:
        - "K", która zawiera warianty badanej cechy,
        - "P(K)", która zawiera wartości prawdopodobieństw rozkładu Poissona wyliczone dla wartości oczekiwanej mu
        oraz poszczególnych wariantów badanej cechy znormalizowane do sumy wartości równej 1.
    """
    # YOUR CODE HERE
    probabilities = poisson.pmf(k, mu)
    normalized_probabilities = probabilities / np.sum(probabilities)
    table = pd.DataFrame({'K': k, 'P(K)': normalized_probabilities})
    return table
    # raise NotImplementedError()


def pearsons_chi2_test(tested_distribution, theoretical_distribution, alpha):
    """
    Parameters
    -------
    tested_distribution: DataFrame
        Tablica opisująca testowany rozkład i zawierająca 2 kolumny:
        - "K", która zawiera warianty badanej cechy, wartości muszą być identycznej jak kolumna "K" zmiennej lokalnej theoretical_distribution,
        - "N(K)", która zawiera liczebności poszczególnych wariantów badanej cechy.
    theoretical_distribution: DataFrame
        Tablica opisująca rozkład teoretyczny i zawierająca 2 kolumny:
        - "K", która zawiera warianty badanej cechy, wartości muszą być identycznej jak kolumna "K" zmiennej lokalnej tested_distribution,
        - "P(K)", która zawiera prawdopodobieństwa poszczególnych wariantów badanej cechy. Wartości z tej kolumny muszą sumować się do 1.
    alpha: float
        Wartość z zakresu [0,1] określająca poziom istotności.

    Returns
    -------
    chi2: float
        Wyliczona na podstawie próby losowej wartość statystyki chi2.
    chi2_alpha: float
        Wyliczona wartość statystyki chi2_alpha.
    H: int
        Wynik testu statystycznego, przyjmuje wartość:
        0 - gdy wynik testu istotności nie daje podstaw do odrzucenia H0 na rzecz H1 na poziomie istotności 1-alpha,
        1 - gdy następuje odrzucenie H0 na rzecz H1 na poziomie istotności 1-alpha.
    """
    # print(tested_distribution) # K: Xi N(K): ni
    # print(theoretical_distribution) # K: Xi P(K): pi
    ni = tested_distribution["N(K)"]
    pi = theoretical_distribution["P(K)"]
    n = sum(tested_distribution["N(K)"])

    npi = n * pi
    r_kw = (ni - npi) ** 2  # różnica kwadrat
    r_kw_d = r_kw / npi  # różnica kwadrat dzielenie
    chi2 = sum(r_kw_d)

    st_sw = tested_distribution["N(K)"].size - 1  # stopnie swobody. liczba stopni swobody: r-1
    # print(st_sw)
    chi2_alpha = sp.stats.chi2.isf(alpha, df=st_sw)

    print("Test chi-kwadrat Pearsona")
    print("H0: Testowana zmienna ma przyjęty rozkład teoretyczny")
    print("H1: Testowana zmienna nie ma przyjętego rozkładu teoretycznego")
    print("chi2 = ", chi2, "chi2_alpha = ", chi2_alpha)
    H = 1
    if (chi2_alpha > chi2):
        H = 0  # nie daje podstaw do odrzucenia
        print("chi2_alpha > chi2")
        print(
            "Wynik testu istotności nie daje podstaw do odrzucenia H0 na rzecz H1 na poziomie istotności 1 - alpha = ",
            1 - alpha)
    else:
        print("chi2_alpha < chi2")
        print("Wynik testu istotności daje podstawy do odrzucenia H0 na rzecz H1 na poziomie istotności 1 - alpha = ",
              1 - alpha)
    data = [chi2, chi2_alpha, H]
    return data


pcos1=point_count_on_subregions( points_1, [40, 20], [10,30], [-15,-5] )
dt1=distribution_table(pcos1[0])
tdt1 = poisson_distribution_table(dt1["K"],5)
results_1=pearsons_chi2_test(dt1,tdt1,0.05)
print(results_1)


pcos2=point_count_on_subregions( points_2, [40, 20], [10,30], [-15,-5] )
dt2=distribution_table(pcos2[0])
tdt2 = poisson_distribution_table(dt2["K"],5)
results_2=pearsons_chi2_test(dt2,tdt2,0.05)
print(results_2)