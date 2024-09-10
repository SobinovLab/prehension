#pragma once
#include "mjxmacro.h"
#include "uitools.h"

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <regex>

#include "tsm.h"

// @https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
class CSVRow
{
public:
    std::string_view operator[](std::size_t index) const;
    std::size_t size() const;
    void readNextRow(std::istream& str);
private:
    std::string         m_line;
    std::vector<int>    m_data;
};

// @https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
std::istream& operator>>(std::istream& str, CSVRow& data);


class IOFunctions
{
public:
    static std::vector<std::string> load_mot_file(
        const char* filename, int* M, int* N, mjtNum** nTime, mjtNum** nPos,
        const int verbose);
    /// <summary>
    /// 
    /// </summary>
    /// <param name="m"></param>
    /// <param name="filename"></param>
    /// <param name="ja_indices"></param>
    /// <param name="ja_names"></param>
    /// <param name="joint_angles">time, dof</param>
    /// <param name="times"></param>
    /// <returns></returns>
    static int import_csv_file(
        const mjModel* m,
        const std::string filename,
        std::vector<int>& ja_indices,
        std::vector<std::string>& ja_names,
        std::vector<std::vector<mjtNum>>& joint_angles,
        std::vector<mjtNum>& times,
        const int verbose);
    /// <summary>
    /// 
    /// </summary>
    /// <param name="filename"></param>
    /// <param name="matrix">time, row, column</param>
    /// <param name="times"></param>
    /// <returns></returns>
    static int import_csv_matrix(
        const std::string filename,
        std::vector<std::vector<std::vector<mjtNum>>>& matrix,
        std::vector<mjtNum>& times,
        const int verbose);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="filename"></param>
    /// <param name="matrix">time, row, column</param>
    /// <param name="times"></param>
    /// <returns></returns>
    static int import_tsm_matrix(
        const std::string filename,
        std::vector<std::vector<std::vector<mjtNum>>>& matrix,
        std::vector<mjtNum>& times,
        const int verbose);

    /// <summary>
    /// Automatically chooses CSV or TSM importer
    /// </summary>
    /// <param name="filename"></param>
    /// <param name="matrix"></param>
    /// <param name="times"></param>
    /// <param name="verbose"></param>
    /// <returns></returns>
    static int import_matrices(
        const std::string filename,
        std::vector<std::vector<std::vector<mjtNum>>>& matrix,
        std::vector<mjtNum>& times,
        const int verbose);

    static int export_adjustment_file(
        const std::string filename,
        const mjModel* m,
        const std::vector<std::vector<int>> fitting_dof_indices,
        const std::vector<double> x_vec);

    static int import_adjustment_file(
        const std::string filename,
        const std::vector<std::string>& ja_names,
        std::vector<int>& adjusted_dof_indices,
        std::vector<double>& adjustments
    );

    static int export_optimization_logs(
        const std::string filename,
        const mjModel* m,
        const std::vector<std::vector<int>> fitting_dof_indices,
        const std::vector<std::vector<double>> x_vecs,
        const std::vector<double> fvals);
};

