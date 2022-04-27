import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def plot_z_mean(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)

    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.scatter(x, z, c=intensity, s=0.05)
    plt.axhline(y=np.mean(z), color="r", linestyle="-")
    plt.title("First view")
    plt.xlabel("X-axis ")
    plt.ylabel("Z-axis ")

    plt.subplot(1, 2, 2)  # index 2
    plt.scatter(y, z, c=intensity, s=0.05)
    plt.axhline(y=np.mean(z), color="r", linestyle="-")
    plt.title("Second view")
    plt.xlabel("Y-axis ")
    plt.ylabel("Z-axis ")
    plt.show()


def filter_ground_points(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)

    pcd = np.column_stack((x, y, z))
    mask = z > np.mean(z)
    spatial_query = pcd[z > np.mean(z)]

    # plotting the results 3D
    ax = plt.axes(projection="3d")
    ax.scatter(x[mask], y[mask], z[mask], c=intensity[mask], s=0.1)
    plt.show()

    # plotting the results 2D
    plt.scatter(x[mask], y[mask], c=intensity[mask], s=0.1)
    plt.show()


def kmeans_planes(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)
    mask = z > np.mean(z)
    X = np.column_stack((x[mask], y[mask]))
    kmeans = KMeans(n_clusters=2).fit(X)
    plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
    plt.show()


def elbow_rule(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)

    mask = z > np.mean(z)

    X = np.column_stack((x[mask], y[mask], z[mask]))
    wcss = []
    for i in tqdm(range(1, 20)):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 20), wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()


def save_results(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)
    mask = z > np.mean(z)
    X = np.column_stack((x[mask], y[mask]))
    kmeans = KMeans(n_clusters=2).fit(X)
    result_path = os.path.join("data/point_cloud_kmeans_segmentation/results",
                               path.split(os.sep)[-1].split(".")[0] + "_result.xyz")
    np.savetxt(result_path, np.column_stack((x[mask], y[mask], z[mask], kmeans.labels_)), fmt="%1.4f", delimiter=";")

    x, y, z, label = np.loadtxt(result_path, skiprows=False, delimiter=";", unpack=True)
    # plotting the results 3D
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, c=label, s=0.1)
    plt.show()

    # plotting the results 2D
    plt.scatter(x, y, c=label, s=0.1)
    plt.show()


def playing_with_features(path):
    x, y, z, illuminance, reflectance, intensity, nb_of_returns = np.loadtxt(path, skiprows=1, delimiter=";",
                                                                             unpack=True)
    mask = z > np.mean(z)
    X = np.column_stack((x[mask], y[mask], z[mask], illuminance[mask], nb_of_returns[mask], intensity[mask]))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
    plt.show()

    X = np.column_stack((z[mask], z[mask], intensity[mask]))
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
    plt.show()


def kmeans_car_segmentation(path):
    x, y, z, r, g, b = np.loadtxt(path, skiprows=1, delimiter=';', unpack=True)
    X = np.column_stack((x, y, z))
    kmeans = KMeans(n_clusters=3).fit(X)
    plt.scatter(x, y, c=kmeans.labels_, s=0.1)
    plt.show()


def dbscan_car_segmentation(path):
    # analysis on dbscan
    x, y, z, r, g, b = np.loadtxt(path, skiprows=1, delimiter=';', unpack=True)
    X = np.column_stack((x, y, z))
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
    plt.scatter(x, y, c=clustering.labels_, s=20)
    plt.show()


if __name__ == "__main__":
    data_folder = "data/point_cloud_kmeans_segmentation"
    dataset = "KME_planes.xyz"
    plane_path = os.path.join(data_folder, dataset)

    # посмотрим, можно ли отсечь точки земли через среднее значение координаты Z
    # plot_z_mean(plane_path)

    # удалим точки земли
    # filter_ground_points(plane_path)

    # сегментируем самолеты
    # kmeans_planes(plane_path)

    # посмотрим лучшее значение количества кластеров с помощью правила локтя,
    # там где локоть сгибается, там и лучшее значение
    # elbow_rule(plane_path)

    # сохраняем результат и потом визуализируем
    save_results(plane_path)

    # используем другие фичи из файла как источник данных для сегментации
    playing_with_features(plane_path)

    # ---------------------------------
    # работа с сегментацией машин
    data_folder = "data/point_cloud_kmeans_segmentation"
    dataset = "KME_cars.xyz"
    car_path = os.path.join(data_folder, dataset)

    # сегментируем машины с помощью kmeans - вышло неплохо
    # kmeans_car_segmentation(car_path)

    # сегментируем машины с помощью DBSCAN - вышло не очень
    dbscan_car_segmentation(car_path)
