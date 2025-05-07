#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>
#include <omp.h>
#include <chrono>
#include <iomanip>
// #include <bits/valarray_after.h>

using namespace std;
using namespace std::chrono;

time_point<system_clock> start_distance, end_distance;
size_t time_distance_value = 0;
time_point<system_clock> start_sort, end_sort;
size_t time_sort_value = 0;
const size_t NUM_THREADS = 8;

// Reads a flattened (NUM_SAMPLES x FEATURE_DIM) binary file of floats
vector<vector<float>> read_features(const string& filename, size_t num_samples, size_t feature_dim)
{
    size_t total_elements = num_samples * feature_dim;
    vector<float> flat(total_elements);

    ifstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Could not open file: " + filename);
    }

    file.read(reinterpret_cast<char*>(flat.data()), total_elements * sizeof(float));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();

    // Reshape the flat vector into a vector of vectors.
    vector<vector<float>> features;
    features.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i)
    {
        vector<float> sample(flat.begin() + i * feature_dim, flat.begin() + (i + 1) * feature_dim);
        features.push_back(move(sample));
    }

    return features;
}

// Reads a binary file containing labels.
vector<int> read_labels(const string& filename, size_t num_samples)
{
    vector<int> labels(num_samples);

    ifstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Could not open file: " + filename);
    }

    file.read(reinterpret_cast<char*>(labels.data()), num_samples * sizeof(int));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();

    return labels;
}

vector<pair<vector<float>, int>> make_list(vector<vector<float>>& features, vector<int>& labels)
{
    vector<pair<vector<float>, int>> items;
    pair<vector<float>, int> item;

    for (int i = 0; i < features.size(); i++)
    {
        item = {features[i], labels[i]};
        items.push_back(item);
    }

    return items;
}

float compute_euclidean(vector<float>& qi, vector<float>& pi)
{
    float euclidean = 0;

    for (int i = 0; i < pi.size(); i++)
    {
        euclidean += powf(qi[i] - pi[i], 2);
    }

    return powf(euclidean, 0.5);
}

tuple<vector<float>, int, float> euclidean_value(pair<vector<float>, int>& p, vector<float>& qi)
{
    tuple<vector<float>, int, float> euclidean_label_tuple;
    float euclidean = compute_euclidean(qi, p.first);
    euclidean_label_tuple = make_tuple(p.first, p.second, euclidean);
    return euclidean_label_tuple;
}
