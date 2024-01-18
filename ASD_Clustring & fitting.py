import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats

# Function to load COVID-19 data from a CSV file
def load_covid_data(file_path):
    """
    Load COVID-19 data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded COVID-19 data.
    """
    covid_df = pd.read_csv(file_path)
    return covid_df

# Function to perform K-means clustering
def perform_kmeans_clustering(features, n_clusters=2):
    """
    Perform K-means clustering on the given features.

    Parameters:
    - features (pd.DataFrame): Features for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - tuple: Cluster labels and KMeans model.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)

    return cluster_labels, kmeans

# Function to plot COVID-19 clusters
def plot_covid_clusters(covid_df, cluster_labels, feature1, feature2, cluster_color1, cluster_color2, center_color, title, x_label, y_label, kmeans_model):
    """
    Plot K-means clustering of COVID-19 data.

    Parameters:
    - covid_df (pd.DataFrame): COVID-19 data.
    - cluster_labels (np.ndarray): Cluster labels.
    - feature1 (str): Feature for the x-axis.
    - feature2 (str): Feature for the y-axis.
    - cluster_color1 (str): Color for the first cluster.
    - cluster_color2 (str): Color for the second cluster.
    - center_color (str): Color for centroids.
    - title (str): Plot title.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - kmeans_model: Fitted KMeans model.
    """
    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.facecolor'] = 'lightyellow'
    plt.grid(True, linestyle='--', alpha=0.9)

    colors = np.where(cluster_labels == 0, cluster_color1, cluster_color2)

    for cluster_id in np.unique(cluster_labels):
        cluster_data = covid_df.loc[cluster_labels == cluster_id]
        label = f'Cluster {cluster_id + 1}' if cluster_id < 2 else 'Centroid'
        plt.scatter(cluster_data[feature1], cluster_data[feature2], c=colors[cluster_labels == cluster_id], edgecolor='k', s=100, label=label)

    centroids = kmeans_model.cluster_centers_
    for i, centroid in enumerate(centroids):
        # Only label the centroids for the first cluster in the plot
        plt.scatter(centroid[0], centroid[1] + 2020, s=100, c=center_color, marker='X', label='Centroid' if i == 0 else None)

    plt.title(f'{title}', fontsize=16, fontweight='bold')
    plt.xlabel(f'{x_label}', fontsize=14, fontweight='bold')
    plt.ylabel(f'{y_label}', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)  # Add legend in the top right corner

    covid_df_transposed = covid_df.T

    plt.show()

# Function for the modified exponential growth model
def modified_exponential_growth_model(x, a, b, c, d):
    """
    Modified exponential growth model function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a, b, c, d (float): Parameters of the model.

    Returns:
    - np.ndarray: Model predictions.
    """
    return a * np.exp(b * x) + c * x**2 + d

# Function to calculate confidence intervals
def err_ranges(x, y, model_func, popt, pcov, alpha=0.05):
    """
    Calculate confidence intervals for the model parameters.

    Parameters:
    - x (np.ndarray): Independent variable.
    - y (np.ndarray): Observed values.
    - model_func: Model function.
    - popt (np.ndarray): Fitted model parameters.
    - pcov (np.ndarray): Covariance matrix of the parameters.
    - alpha (float): Significance level for confidence intervals.

    Returns:
    - tuple: Lower and upper bounds of the confidence intervals.
    """
    n = len(y)
    p = len(popt)
    dof = max(0, n - p)

    t_val = np.abs(stats.t.ppf(alpha / 2, dof))
    sigma = np.sqrt(np.sum((y - model_func(x, *popt))**2) / dof)
    perr = np.sqrt(np.diag(pcov))

    lower_bound = popt - t_val * perr
    upper_bound = popt + t_val * perr

    return lower_bound, upper_bound

# Load COVID-19 data
file_path = r"C:\Users\heman\OneDrive\Desktop\covid_data.csv"
covid_df = load_covid_data(file_path)

# Select relevant columns for clustering
features = covid_df[['confirmed', 'recovered', 'death']]

# Perform K-means clustering
cluster_labels, kmeans_model = perform_kmeans_clustering(features)

# Plot for 'confirmed' vs 'recovered'
plot_covid_clusters(covid_df, cluster_labels, 'confirmed', 'recovered', 'yellow', 'green', 'red', 'Clustering of COVID-19 Data (Confirmed vs Recovered)', 'Confirmed Cases', 'Recovered Cases', kmeans_model)

# Plot for 'confirmed' vs 'death'
plot_covid_clusters(covid_df, cluster_labels, 'confirmed', 'death', 'red', 'blue', 'violet', 'Clustering of COVID-19 Data (Confirmed vs Death)', 'Confirmed Cases', 'Death Cases', kmeans_model)

# Perform K-means clustering for 'deaths' vs 'year' with only 2020, 2021, 2022 on the y-axis
deaths_year_features = covid_df[['death', 'year']]
cluster_labels_deaths_year, kmeans_model_deaths_year = perform_kmeans_clustering(deaths_year_features, n_clusters=3)

# Plot the clustering for 'deaths' vs 'year' with only 2020, 2021, 2022 on the y-axis
plt.figure(figsize=(10, 8))
plt.rcParams['axes.facecolor'] = 'lightyellow'
plt.grid(True, linestyle='--', alpha=0.9)

colors = np.where(cluster_labels_deaths_year == 0, 'yellow', np.where(cluster_labels_deaths_year == 1, 'green', 'red'))

for cluster_id in np.unique(cluster_labels_deaths_year):
    cluster_data = covid_df.loc[cluster_labels_deaths_year == cluster_id]
    label = f'Cluster {cluster_id + 1}' if cluster_id < 2 else f'Cluster {cluster_id + 1}'
    plt.scatter(cluster_data['death'], [2020, 2021, 2022][cluster_id] * np.ones_like(cluster_data['death']), c=colors[cluster_labels_deaths_year == cluster_id], edgecolor='k', s=100, label=label)

centroids_deaths_year = kmeans_model_deaths_year.cluster_centers_
for i, centroid in enumerate(centroids_deaths_year):
    # Only label the centroids for the first cluster in the plot
    plt.scatter(centroid[0], centroid[1] + 2020, s=100, c='red', marker='X', label=f'Centroid {i + 1}' if i == 0 else None)

# Set y-axis ticks to display only 2019, 2020, 2021, 2022
plt.yticks([2019, 2020, 2021, 2022], fontsize=12)

plt.title('Clustering of COVID-19 Data (Deaths vs Year)', fontsize=16, fontweight='bold')
plt.xlabel('Death Cases', fontsize=14, fontweight='bold')
plt.ylabel('Year', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)  # Add legend in the top right corner

plt.show()

# Fit modified exponential growth model to confirmed cases
x = np.arange(len(covid_df))
y = covid_df['confirmed'].values

# Provide initial guesses for parameters
modified_initial_guess = [1.0, 0.2, 1.0, 1.0]

# Increase maxfev further
modified_popt, modified_pcov = curve_fit(modified_exponential_growth_model, x, y, p0=modified_initial_guess, maxfev=10000)

# Estimate confidence intervals
modified_lower_bound, modified_upper_bound = err_ranges(x, y, modified_exponential_growth_model, modified_popt, modified_pcov)

# Plot the best-fitting function and confidence range for modified model
plt.figure(figsize=(10, 8))
plt.plot(x, modified_exponential_growth_model(x, *modified_popt), label='Best Fit (Modified)', color='green')
plt.fill_between(x, modified_exponential_growth_model(x, *modified_lower_bound), modified_exponential_growth_model(x, *modified_upper_bound), color='green', alpha=0.3, label='Confidence Interval (Modified)')

plt.scatter(x, y, color='red', label='Forecast')
plt.title('Exponential Growth Model Fitting for Confirmed Cases', fontsize=16, fontweight='bold')
plt.xlabel('Days', fontsize=14, fontweight='bold')
plt.ylabel('Confirmed Cases', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=12)  # Add legend

plt.show()