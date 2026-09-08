import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("dataset.csv")

print("========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# =========================================================
# 2. SELECT NUMERICAL FEATURES
# =========================================================

features = [
    'cpu_usage',
    'memory_usage',
    'disk_usage'
]

X = df[features]

print("\nSelected Features:")
print(features)


# =========================================================
# 3. STANDARDIZE DATA
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Standardization Completed.")


# =========================================================
# 4. PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n========== PCA RESULTS ==========")

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Variance Explained:")
print(round(total_variance, 4))


# PCA visualization

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Cloud Workload Dataset")
plt.grid(True)

plt.show()


# =========================================================
# 5. K-MEANS - TEST DIFFERENT VALUES OF K
# =========================================================

print("\n========== K-MEANS TESTING ==========")

k_values = range(2, 7)

kmeans_silhouette_scores = []
kmeans_inertia = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(
        X_pca,
        labels
    )

    kmeans_silhouette_scores.append(silhouette)

    kmeans_inertia.append(
        kmeans.inertia_
    )

    print(
        f"K = {k}, "
        f"Silhouette Score = {silhouette:.4f}, "
        f"Inertia = {kmeans.inertia_:.4f}"
    )


# Find best K

best_k = list(k_values)[
    kmeans_silhouette_scores.index(
        max(kmeans_silhouette_scores)
    )
]

print("\nBest K:")
print(best_k)

print("\nBest K-Means Silhouette Score:")
print(
    round(
        max(kmeans_silhouette_scores),
        4
    )
)


# =========================================================
# 6. ELBOW METHOD
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    kmeans_inertia,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.grid(True)

plt.show()


# =========================================================
# 7. FINAL K-MEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_pca)


# K-Means evaluation

kmeans_silhouette = silhouette_score(
    X_pca,
    kmeans_labels
)

kmeans_db_index = davies_bouldin_score(
    X_pca,
    kmeans_labels
)


print("\n========== K-MEANS EVALUATION ==========")

print(
    "Silhouette Score:",
    round(kmeans_silhouette, 4)
)

print(
    "Davies-Bouldin Index:",
    round(kmeans_db_index, 4)
)


# K-Means visualization

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    f"K-Means Clustering (K = {best_k})"
)

plt.grid(True)

plt.show()


# =========================================================
# 8. DBSCAN PARAMETER TESTING
# =========================================================

print("\n========== DBSCAN PARAMETER TESTING ==========")

eps_values = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00
]

min_samples = 5

best_dbscan = None

best_dbscan_score = -1


for eps in eps_values:

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_pca)

    # Remove noise

    mask_temp = labels != -1

    valid_data = X_pca[mask_temp]

    valid_labels = labels[mask_temp]

    # Count clusters

    if len(valid_labels) > 0:

        number_of_clusters = len(
            set(valid_labels)
        )

    else:

        number_of_clusters = 0

    number_of_noise = list(
        labels
    ).count(-1)


    # Evaluate only when there are at least 2 clusters

    if number_of_clusters >= 2:

        score = silhouette_score(
            valid_data,
            valid_labels
        )

        print(
            f"eps = {eps:.2f}, "
            f"Clusters = {number_of_clusters}, "
            f"Noise = {number_of_noise}, "
            f"Silhouette = {score:.4f}"
        )


        # Select best DBSCAN based on Silhouette Score

        if score > best_dbscan_score:

            best_dbscan_score = score

            best_dbscan = {
                "eps": eps,
                "labels": labels,
                "clusters": number_of_clusters,
                "noise": number_of_noise
            }


    else:

        print(
            f"eps = {eps:.2f}, "
            f"Clusters = {number_of_clusters}, "
            f"Noise = {number_of_noise}, "
            f"Silhouette = Not Available"
        )


# =========================================================
# 9. FINAL DBSCAN MODEL
# =========================================================

print("\n========== BEST DBSCAN RESULT ==========")


if best_dbscan is not None:

    best_eps = best_dbscan["eps"]

    dbscan_labels = best_dbscan["labels"]

    dbscan_clusters = best_dbscan["clusters"]

    dbscan_noise = best_dbscan["noise"]


    print(
        "Best eps:",
        best_eps
    )

    print(
        "Number of Clusters:",
        dbscan_clusters
    )

    print(
        "Number of Noise Points:",
        dbscan_noise
    )


    # Remove noise for evaluation

    mask = dbscan_labels != -1

    valid_data = X_pca[mask]

    valid_labels = dbscan_labels[mask]


    # DBSCAN evaluation

    dbscan_silhouette = silhouette_score(
        valid_data,
        valid_labels
    )

    dbscan_db_index = davies_bouldin_score(
        valid_data,
        valid_labels
    )


    print(
        "DBSCAN Silhouette Score:",
        round(dbscan_silhouette, 4)
    )

    print(
        "DBSCAN Davies-Bouldin Index:",
        round(dbscan_db_index, 4)
    )


    # =====================================================
    # 10. DBSCAN VISUALIZATION
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=dbscan_labels,
        alpha=0.6
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        f"DBSCAN Clustering (eps = {best_eps})"
    )

    plt.grid(True)

    plt.show()


else:

    dbscan_silhouette = None

    dbscan_db_index = None

    print(
        "No suitable DBSCAN parameters "
        "were found."
    )


# =========================================================
# 11. FINAL COMPARISON
# =========================================================

print("\n")
print("==========================================")
print("           FINAL COMPARISON")
print("==========================================")


print("\nK-Means:")

print(
    "Best K:",
    best_k
)

print(
    "Silhouette Score:",
    round(kmeans_silhouette, 4)
)

print(
    "Davies-Bouldin Index:",
    round(kmeans_db_index, 4)
)


if dbscan_silhouette is not None:

    print("\nDBSCAN:")

    print(
        "Best eps:",
        best_eps
    )

    print(
        "Number of Clusters:",
        dbscan_clusters
    )

    print(
        "Number of Noise Points:",
        dbscan_noise
    )

    print(
        "Silhouette Score:",
        round(dbscan_silhouette, 4)
    )

    print(
        "Davies-Bouldin Index:",
        round(dbscan_db_index, 4)
    )


    print("\n========== INTERPRETATION ==========")

    print(
        "Higher Silhouette Score = Better"
    )

    print(
        "Lower Davies-Bouldin Index = Better"
    )


    # Determine best clustering method

    if (
        kmeans_silhouette > dbscan_silhouette
        and
        kmeans_db_index < dbscan_db_index
    ):

        print(
            "\nBEST METHOD: K-Means"
        )

        print(
            "K-Means performs better because "
            "it has a higher Silhouette Score "
            "and a lower Davies-Bouldin Index."
        )


    elif (
        dbscan_silhouette > kmeans_silhouette
        and
        dbscan_db_index < kmeans_db_index
    ):

        print(
            "\nBEST METHOD: DBSCAN"
        )

        print(
            "DBSCAN performs better because "
            "it has a higher Silhouette Score "
            "and a lower Davies-Bouldin Index."
        )


    else:

        print(
            "\nRESULT: MIXED PERFORMANCE"
        )

        print(
            "The evaluation measures give "
            "mixed results. Cluster structure "
            "should also be considered."
        )


else:

    print(
        "\nDBSCAN could not produce at least "
        "two valid clusters."
    )

    print(
        "A direct numerical comparison with "
        "K-Means is not possible."
    )
