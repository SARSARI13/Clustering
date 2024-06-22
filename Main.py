import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import sys

# Function to load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to create features
def create_features(data_training):
    features = []
    object_labels = []

    for obj in data_training:
        object_id = obj['Object_id']
        obj_images = obj['Images']
        feature_dimensions = [[] for _ in range(40)]

        for img in obj_images:
            for i, dim in enumerate(img['Dimensions']):
                feature_dimensions[i].append(dim[f'Dimension_{i}'])

        means = [np.mean(dim) for dim in feature_dimensions]
        std_devs = [np.std(dim) for dim in feature_dimensions]
        medians = [np.median(dim) for dim in feature_dimensions]
        
        features.append({
            'Object_id': object_id,
            'Means': means,
            'Standard_Deviations': std_devs,
            'Medians': medians
        })

        object_labels.append(object_id)

    return np.array(features), np.array(object_labels)

# Function to perform clustering and return adjusted Rand index
def perform_clustering(training_path, test_path):
    data_training = load_data(training_path)
    features, object_labels = create_features(data_training)

    X_train = np.array([feature['Means'] + feature['Standard_Deviations'] + feature['Medians'] for feature in features])

    num_clusters = 100  # One cluster per object
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    rand_score_train = adjusted_rand_score(object_labels, kmeans.predict(X_train))
    data_test = load_data(test_path)
    features_test, object_labels_test = create_features(data_test)
    X_test = np.array([feature['Means'] + feature['Standard_Deviations'] + feature['Medians'] for feature in features_test])

    predicted_clusters_test = kmeans.predict(X_test)
    rand_score_test = adjusted_rand_score(object_labels_test, predicted_clusters_test)

    return kmeans, rand_score_train, rand_score_test

# Function to identify the closest objects
def identify_closest_objects(data):
    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    np.fill_diagonal(distances, np.inf)
    obj1, obj2 = np.unravel_index(distances.argmin(), distances.shape)
    feature_difference = np.abs(data[obj1] - data[obj2])
    different_feature_index = np.argmax(feature_difference)

    return obj1, obj2, different_feature_index


if __name__ == "__main__":
    # Check if enough arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python your_script.py /path/to/data_training.json /path/to/data_test.json")
        sys.exit(1)

    training_path = sys.argv[1]
    test_path = sys.argv[2]
    

    model, rand_score_train, rand_score_test = perform_clustering(training_path, test_path)
    print(f"Adjusted Rand Index on training data: {rand_score_train}")
    print(f"Adjusted Rand Index on test data: {rand_score_test}")

    features, _ = create_features(load_data(training_path))
    X_train = np.array([feature['Means'] + feature['Standard_Deviations'] + feature['Medians'] for feature in features])

    obj1, obj2, different_feature_index = identify_closest_objects(X_train)
    print(f"Objects with Closest Dimensions: {obj1} and {obj2}")
    print(f"Different Feature Index: {different_feature_index}")
