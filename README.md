# Clustering Model Training and Testing

## Objective

The aim of this project is to train and test a clustering model using training and test data and to identify, among all objects, the two objects whose dimensions are the closest, and identify the sole characteristic that distinguishes them.

## Prerequisites

- **Python:** The script has been developed and tested with Python 3.8.2.
- **Libraries:** The following libraries are required:
  - numpy 1.24.4
  - scikit-learn 1.3.2
  - sys
  - json

## Data Structure

The structure of the training data is as follows:

- 100 different objects
- 30 images per object
- 40 dimensions per image
- Each object is identified by an integer from 0 to 99
- Each image is identified by an integer from 0 to 29
- The 40 dimensions are the same for all images and all objects.

## Usage

This script is ready to be executed from the command line. You can pass the paths to your training and test files as arguments when running the script.

Example: `python Main.py /path/to/data_training.json /path/to/data_test.json`

## Expected Output Data

The script produces the following outputs:

1. Adjusted Rand Index: The adjusted Rand index is calculated to evaluate the quality of the clusters formed on the train and test data.
2. Objects with Closest Dimensions: The indices of the objects in the training set that have the closest clusters.
3. Index of Differentiating Characteristic: The index of the differentiating characteristic between the identified objects.

