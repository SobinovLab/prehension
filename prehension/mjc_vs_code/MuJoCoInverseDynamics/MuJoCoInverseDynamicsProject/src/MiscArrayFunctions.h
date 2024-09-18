#pragma once
#include <math.h>
#include <vector>
#include <algorithm>

class MiscArrayFunctions
{
public:
    static double arr_max(const double* arr, const int len);
    static double arr_max(const double* arr, const int len, const int* indices);

    static double rms(const int len, const double* arr1, const double* arr2);
    static float rms(const int len, const float* arr1, const float* arr2);
    static double rms(const int len, const double* arr);

    static void normalize_arr(double* arr, const int len);
    static void normalize_arr(float* arr, const int len);

    static bool val_in_array(const int val, const int len, const int* arr);
    static bool val_in_array(const int val, const std::vector<int> arr);

    static double median(std::vector<double> arr);
    static float median(std::vector<float> arr);
    static int median(std::vector<int> arr);
    static bool median(std::vector<bool> arr);

    static float scalar_multiplication(const int len, const float* arr1, const float* arr2);

    static double* average_position(const double* nPos, const int M, const int N);

    static std::vector<double> elementwise_sum(const std::vector<double>& v1, const std::vector<double>& v2);
};

