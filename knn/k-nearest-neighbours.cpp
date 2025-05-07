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

vector<tuple<vector<float>, int, float>> euclidean_list(vector<pair<vector<float>, int>>& train,
                                                        vector<float>& test_point, bool parallel = false)
{
    vector<tuple<vector<float>, int, float>> list;
    tuple<vector<float>, int, float> point;

    if (parallel)
    {
        int size = train.size();
        int fraction = size / NUM_THREADS;
#pragma omp parallel for schedule(static,fraction)  private(point)
        for (int i = 0; i < train.size(); i++)
        {
            point = euclidean_value(train[i], test_point);
#pragma omp critical
            {
                list.push_back(point);
            }
        }
    }
    else
    {
        for (pair<vector<float>, int> train_point : train)
        {
            point = euclidean_value(train_point, test_point);
            list.push_back(point);
        }
    }
    return list;
}

int partition(int left, int right, vector<tuple<vector<float>, int, float>>& items)
{
    // Added reference
    float item = get<2>(items[right]);

    int i = left - 1;

    for (int j = left; j < right; j++)
    {
        if (get<2>(items[j]) <= item)
        {
            i++;
            items[i].swap(items[j]);
        }
    }
    items[i + 1].swap(items[right]);
    return i + 1;
}

void q_sort_items(int left, int right, vector<tuple<vector<float>, int, float>>& items)
{
    // Added reference
    if (left < right)
    {
        int q = partition(left, right, items);
        q_sort_items(left, q - 1, items);
        q_sort_items(q + 1, right, items);
    }
}


void q_sort_sections(int left, int right, vector<tuple<vector<float>, int, float>>& items,
                     int depth = 0)
{
    if (left < right)
    {
        int q = partition(left, right, items);

        // Switch to sequential for small subarrays or when we've created enough parallelism
        if ((right - left) < 1000 || depth >= omp_get_max_threads())
        {
            q_sort_sections(left, q - 1, items, depth + 1);
            q_sort_sections(q + 1, right, items, depth + 1);
        }
        else
        {
#pragma omp parallel sections
            {
#pragma omp section
                {
                    q_sort_sections(left, q - 1, items, depth + 1);
                }
#pragma omp section
                {
                    q_sort_sections(q + 1, right, items, depth + 1);
                }
            }
        }
    }
}

void q_sort_tasks(int left, int right, vector<tuple<vector<float>, int, float>>& items, int depth = 0)
{
    if (left < right)
    {
        int q = partition(left, right, items);

        // Switch to sequential for small subarrays or when we've created enough parallelism
        if ((right - left) < 1000 || depth >= omp_get_max_threads())
        {
            q_sort_tasks(left, q - 1, items, depth + 1);
            q_sort_tasks(q + 1, right, items, depth + 1);
        }
        else
        {
#pragma omp task shared(items) firstprivate(left, q)
            q_sort_tasks(left, q - 1, items, depth + 1);

#pragma omp task shared(items) firstprivate(q, right)
            q_sort_tasks(q + 1, right, items, depth + 1);
#pragma omp taskwait
        }
    }
}

pair<vector<float>, int> majority_vote(vector<tuple<vector<float>, int, float>>& list, vector<float>& qi, size_t K)
{
    pair<int, int> most = {get<1>(list[0]), 1};

    for (int i = 1; i < K; i++)
    {
        if (most.first == get<1>(list[i]))
        {
            most.second++;
        }
        else if (most.second > 1)
        {
            most.second--;
        }
        else
        {
            most.first = get<1>(list[i]);
        }
    }

    if (most.second == 1)
    {
        most.first = get<1>(list[0]);
    }

    pair<vector<float>, int> vote = {qi, most.first};

    return vote;
}

