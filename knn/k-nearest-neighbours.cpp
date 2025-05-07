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
const size_t NUM_THREADS = 16;

// Reads a flattened (NUM_SAMPLES x FEATURE_DIM) binary file of floats
vector<vector<float>> read_features(const string &filename, size_t num_samples, size_t feature_dim)
{
    size_t total_elements = num_samples * feature_dim;
    vector<float> flat(total_elements);

    ifstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Could not open file: " + filename);
    }

    file.read(reinterpret_cast<char *>(flat.data()), total_elements * sizeof(float));
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
vector<int> read_labels(const string &filename, size_t num_samples)
{
    vector<int> labels(num_samples);

    ifstream file(filename, ios::binary);
    if (!file)
    {
        throw runtime_error("Could not open file: " + filename);
    }

    file.read(reinterpret_cast<char *>(labels.data()), num_samples * sizeof(int));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();

    return labels;
}

vector<pair<vector<float>, int>> make_list(vector<vector<float>> &features, vector<int> &labels)
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

float compute_euclidean(vector<float> &qi, vector<float> &pi)
{
    float euclidean = 0;

    for (int i = 0; i < pi.size(); i++)
    {
        euclidean += powf(qi[i] - pi[i], 2);
    }

    return powf(euclidean, 0.5);
}

tuple<vector<float>, int, float> euclidean_value(pair<vector<float>, int> &p, vector<float> &qi)
{
    tuple<vector<float>, int, float> euclidean_label_tuple;
    float euclidean = compute_euclidean(qi, p.first);
    euclidean_label_tuple = make_tuple(p.first, p.second, euclidean);
    return euclidean_label_tuple;
}

vector<tuple<vector<float>, int, float>> euclidean_list(vector<pair<vector<float>, int>> &train,
                                                        vector<float> &test_point, bool parallel = false)
{
    vector<tuple<vector<float>, int, float>> list;
    tuple<vector<float>, int, float> point;

    if (parallel)
    {
        int size = train.size();
        int fraction = size / NUM_THREADS;
#pragma omp parallel for schedule(static, fraction) private(point)
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

int partition(int left, int right, vector<tuple<vector<float>, int, float>> &items)
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

void q_sort_items(int left, int right, vector<tuple<vector<float>, int, float>> &items)
{
    // Added reference
    if (left < right)
    {
        int q = partition(left, right, items);
        q_sort_items(left, q - 1, items);
        q_sort_items(q + 1, right, items);
    }
}

void q_sort_sections(int left, int right, vector<tuple<vector<float>, int, float>> &items,
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

void q_sort_tasks(int left, int right, vector<tuple<vector<float>, int, float>> &items, int depth = 0)
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

pair<vector<float>, int> majority_vote(vector<tuple<vector<float>, int, float>> &list, vector<float> &qi, size_t K)
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

vector<pair<vector<float>, int>> train_list(vector<pair<vector<float>, int>> &feature_labels,
                                            vector<vector<float>> &test_features, size_t K, bool parallel = false, string mode = "sections")
{
    vector<tuple<vector<float>, int, float>> list;
    pair<vector<float>, int> vote;
    vector<pair<vector<float>, int>> trained;
    time_distance_value = 0;
    time_sort_value = 0;
    for (int i = 0; i < test_features.size(); i++)
    {
        if (!parallel)
        {
            start_distance = high_resolution_clock::now();
            list = euclidean_list(feature_labels, test_features[i]);
            end_distance = high_resolution_clock::now();
            auto time_distance = duration_cast<nanoseconds>(end_distance - start_distance);
            time_distance_value += time_distance.count();

            start_sort = high_resolution_clock::now();
            q_sort_items(0, list.size() - 1, list);
            end_sort = high_resolution_clock::now();
            auto time_sort = duration_cast<nanoseconds>(end_sort - start_sort);
            time_sort_value += time_sort.count();

            vote = majority_vote(list, test_features[0], K);
            trained.push_back(vote);
        }
        else
        {
            if (mode == "Sections")
            {
                start_distance = high_resolution_clock::now();
                list = euclidean_list(feature_labels, test_features[i], parallel);
                end_distance = high_resolution_clock::now();
                auto time_distance = duration_cast<nanoseconds>(end_distance - start_distance);
                time_distance_value += time_distance.count();

                start_sort = high_resolution_clock::now();
                q_sort_sections(0, list.size() - 1, list);
                end_sort = high_resolution_clock::now();
                auto time_sort = duration_cast<nanoseconds>(end_sort - start_sort);
                time_sort_value += time_sort.count();

                vote = majority_vote(list, test_features[0], K);
                trained.push_back(vote);
            }
            else if (mode == "Tasks")
            {
                start_distance = high_resolution_clock::now();
                list = euclidean_list(feature_labels, test_features[i], parallel);
                end_distance = high_resolution_clock::now();
                auto time_distance = duration_cast<nanoseconds>(end_distance - start_distance);
                time_distance_value += time_distance.count();

                start_sort = high_resolution_clock::now();
#pragma omp parallel
                {
#pragma omp single
                    q_sort_tasks(0, list.size() - 1, list);
                }
                end_sort = high_resolution_clock::now();
                auto time_sort = duration_cast<nanoseconds>(end_sort - start_sort);
                time_sort_value += time_sort.count();

                vote = majority_vote(list, test_features[0], K);
                trained.push_back(vote);
            }
        }
    }

    return trained;
}

float compute_accuracy(vector<pair<vector<float>, int>> &trained_list,
                       vector<pair<vector<float>, int>> &test_feature_labels)
{
    int correct = 0;
    for (int i = 0; i < trained_list.size(); i++)
    {
        if (trained_list[i].second == test_feature_labels[i].second)
        {
            correct++;
        }
    }

    int n = trained_list.size();

    float accuracy = (static_cast<float>(correct) / n);

    return accuracy;
}

int main()
{
    constexpr size_t NUM_SAMPLES = 50000;
    constexpr size_t FEATURE_DIM = 512;
    vector<size_t> K = {3, 5, 7};
    constexpr size_t NUM_TESTS = NUM_SAMPLES / 5;
    omp_set_num_teams(NUM_THREADS);

    vector<vector<float>> features = read_features("train/train_features.bin", NUM_SAMPLES, FEATURE_DIM);
    vector<int> labels = read_labels("train/train_labels.bin", NUM_SAMPLES);

    vector<vector<float>> test_features = read_features("test/test_features.bin", NUM_TESTS, FEATURE_DIM);
    vector<int> test_labels = read_labels("test/test_labels.bin", NUM_TESTS);

    vector<pair<vector<float>, int>> feature_labels = make_list(features, labels);
    vector<pair<vector<float>, int>> test_feature_labels = make_list(test_features, test_labels);

    time_point<system_clock> start_total, end_total;

    vector<string> modes = {"Sections", "Tasks"};

    for (size_t k : K)
    {

        start_total = high_resolution_clock::now();
        vector<pair<vector<float>, int>> trained_list = train_list(feature_labels, test_features, k);
        end_total = high_resolution_clock::now();

        auto time_total = duration_cast<nanoseconds>(end_total - start_total);

        float accuracy = compute_accuracy(trained_list, test_feature_labels);

        size_t total_time = time_total.count();

        size_t serial_time = total_time;

        // cout << "Sample size:\t" << NUM_SAMPLES << "\n" << "K:\t" << K << "\n" <<  "Accuracy:\t" << setprecision(3) << accuracy << "\n" << "Times\n" << "===========\n" << "Distance\t|Sort\t\t|TOTAL\t\t|\n" << setprecision(8) << static_cast<float>(time_distance_value)*pow(10,-9) << "\t|" << static_cast<float>(time_sort_value)*pow(10,-9) << "\t|" << static_cast<float>(total_time)*pow(10,-9) << "|\n";

        cout << "K-NEAREST NEIGHBOURS\n====================\n";
        cout << "INPUTS\n--------------------\n";
        cout << "Training Sample Size:\t" << NUM_SAMPLES << "\n";
        cout << "Test Sample Size:\t" << NUM_TESTS << "\n";
        cout << "K:\t\t\t" << k << "\n";
        cout << "====================\n";
        cout << "ANALYSIS\n--------------------\n";
        cout << "Accuracy:\t" << accuracy << "\n";
        cout << "====================\n";
        cout << "EXECUTION TIME\n--------------------\n";
        cout << "--------------------\n";
        cout << "Serial Execution Time\n--------------------\n";
        cout << "Distance:\t" << setprecision(8) << static_cast<float>(time_distance_value) * pow(10, -9) << "\n";
        cout << "Sort:\t\t" << setprecision(8) << static_cast<float>(time_sort_value) * pow(10, -9) << "\n";
        cout << "TOTAL:\t\t" << setprecision(8) << static_cast<float>(total_time) * pow(10, -9) << "\n";
        cout << "--------------------\n";
        cout << "Parallel Execution Time\n--------------------\n";

        float speedup = 0;

        for (string mode : modes)
        {
            cout << mode << "\n\n";
            start_total = high_resolution_clock::now();
            trained_list = train_list(feature_labels, test_features, K, true, mode);
            end_total = high_resolution_clock::now();
            time_total = duration_cast<nanoseconds>(end_total - start_total);
            total_time = time_total.count();

            cout << "Distance:\t" << setprecision(8) << static_cast<float>(time_distance_value) * pow(10, -9) << "\n";
            cout << "Sort:\t\t" << setprecision(8) << static_cast<float>(time_sort_value) * pow(10, -9) << "\n";
            cout << "TOTAL:\t\t" << setprecision(8) << static_cast<float>(total_time) * pow(10, -9) << "\n";

            speedup = static_cast<float>(serial_time) / total_time;
            cout << "\n"
                 << "SPEEDUP:\t\t" << speedup << "\n";

            cout << "--------------------\n";
        }

        cout << "\n====================\n";
    }

    return 0;
}
