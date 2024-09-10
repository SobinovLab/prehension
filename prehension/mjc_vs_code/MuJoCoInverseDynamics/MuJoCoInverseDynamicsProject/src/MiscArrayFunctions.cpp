#include "MiscArrayFunctions.h"

double MiscArrayFunctions::arr_max(const double* arr, const int len)
{
    double max = arr[0];
    for (int i = 1; i < len; ++i)
        if (arr[i] > max)
            max = arr[i];
    return max;
}

double MiscArrayFunctions::arr_max(const double* arr, const int len, const int* indices)
{
    double max = arr[indices[0]];
    int it;
    for (int i = 1; i < len; ++i) {
        it = indices[i];
        if (arr[i] > max)
            max = arr[i];
    }
    return max;
}

double MiscArrayFunctions::rms(const int len, const double* arr1, const double* arr2)
{
    double val = 0;
    for (int i = 0; i < len; ++i)
        val += pow(arr1[i] - arr2[i], (double)2.0);
    val /= len;
    val = pow(val, 0.5);
    return val;
}

float MiscArrayFunctions::rms(const int len, const float* arr1, const float* arr2)
{
    double val = 0;
    for (int i = 0; i < len; ++i)
        val += pow((double) arr1[i] - arr2[i], (double)2.0);
    val /= len;
    val = pow(val, 0.5);
    return (float) val;
}

double MiscArrayFunctions::rms(const int len, const double* arr)
{
    double val = 0;
    for (int i = 0; i < len; ++i)
        val += pow(arr[i], (double)2.0);
    val /= len;
    val = pow(val, 0.5);
    return val;
}

void MiscArrayFunctions::normalize_arr(double* arr, const int len)
{
    double maxval = -1;
    for (int i = 0; i < len; ++i)
        if (abs(arr[i]) > maxval)
            maxval = abs(arr[i]);

    if (maxval == 0)
        for (int i = 0; i < len; ++i)
            arr[i] = 0;
    else
        for (int i = 0; i < len; ++i)
            arr[i] /= maxval;
}

void MiscArrayFunctions::normalize_arr(float* arr, const int len)
{
    float maxval = -1;
    for (int i = 0; i < len; ++i)
        if (abs(arr[i]) > maxval)
            maxval = abs(arr[i]);

    if (maxval == 0)
        for (int i = 0; i < len; ++i)
            arr[i] = 0;
    else
        for (int i = 0; i < len; ++i)
            arr[i] /= maxval;
}

bool MiscArrayFunctions::val_in_array(const int val, const int len, const int* arr)
{
    bool answ = false;
    for (int i = 0; i < len; ++i)
        if (arr[i] == val) {
            answ = true;
            break;
        }
    return answ;
}

bool MiscArrayFunctions::val_in_array(const int val, const std::vector<int> arr)
{
    bool answ = true;
    if (arr.end() == std::find(arr.begin(), arr.end(), val))
        answ = false;
    return answ;
}

double MiscArrayFunctions::median(std::vector<double> arr)
{
    std::sort(arr.begin(), arr.end());
    return arr[arr.size() / 2];
}

float MiscArrayFunctions::median(std::vector<float> arr)
{
    std::sort(arr.begin(), arr.end());
    return arr[arr.size() / 2];
}

int MiscArrayFunctions::median(std::vector<int> arr)
{
    std::sort(arr.begin(), arr.end());
    return arr[arr.size() / 2];
}

bool MiscArrayFunctions::median(std::vector<bool> arr)
{
    struct {
        bool operator()(bool a, bool b) const
        {
            return (int)a < (int)b;
        }
    } customLess;
    std::sort(arr.begin(), arr.end(), customLess);
    return arr[arr.size() / 2];
}

float MiscArrayFunctions::scalar_multiplication(const int len, const float* arr1, const float* arr2)
{
    float answ = 0;
    for (int i = 0; i < len; ++i)
        answ += arr1[i] * arr2[i];
    return answ;
}

double* MiscArrayFunctions::average_position(const double* nPos, const int M, const int N)
{
    double* nPosAverage = new double[M];
    for (int j = 0; j < M; ++j)
    {
        nPosAverage[j] = 0;
        for (int i = 0; i < N; ++i)
            nPosAverage[j] += nPos[i * M + j];
        nPosAverage[j] /= M;
    }
    return nPosAverage;
}

std::vector<double> MiscArrayFunctions::elementwise_sum(const std::vector<double>& v1, const std::vector<double>& v2)
{
    std::vector<double> answ;
    for (size_t i_v = 0; i_v < v1.size() && i_v < v2.size(); i_v++)
        answ.push_back(v1[i_v] + v2[i_v]);
    return answ;
}
