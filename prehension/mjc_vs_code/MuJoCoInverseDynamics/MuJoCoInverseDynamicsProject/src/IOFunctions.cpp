#include "IOFunctions.h"

using namespace std;
namespace fs = std::filesystem;

std::string_view CSVRow::operator[](std::size_t index) const
{
    return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));

}

std::size_t CSVRow::size() const
{
    return m_data.size() - 1;
}

void CSVRow::readNextRow(std::istream& str)
{
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line.find(',', pos)) != std::string::npos)
    {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
}

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}

std::vector<std::string> IOFunctions::load_mot_file(const char* filename, int* M, int* N, mjtNum** nTime, mjtNum** nPos, const int verbose)
{
    std::ifstream infile(filename);

    std::string line, buf;
    std::getline(infile, line);  // Coordinates
    std::getline(infile, line);  // version

    std::getline(infile, line);
    *N = std::stoi(line.substr(line.find('=') + 1));

    std::getline(infile, line);
    *M = std::stoi(line.substr(line.find('=') + 1)) - 1;

    while (line != "endheader")
        std::getline(infile, line);
    std::getline(infile, line);

    std::stringstream ls(line);
    ls >> buf;  // time
    std::vector<std::string> dof_names;
    while (ls >> buf)
        dof_names.push_back(buf);

    *nTime = new mjtNum[(*N)];
    *nPos = new mjtNum[(*N) * (*M)];

    int iTime = 0;
    while (std::getline(infile, line)) {
        ls = std::stringstream(line);
        ls >> (*nTime)[iTime];

        for (int i = 0; i < *M; ++i)
            ls >> (*nPos)[iTime * (*M) + i];

        iTime++;
    }

    if (verbose) {
        std::cout << "Read kinematics file" << std::endl;
        std::cout << "\tN = " << *N << std::endl;
        std::cout << "\tM = " << *M << std::endl;
    }

    // std::cout << "\tDOFs: ";
    // for (int i = 0; i < dof_names.size(); ++i)
    //     std::cout << dof_names[i] << '\t';
    // std::cout << std::endl;

    // std::cout << "\tnTime: ";
    // for (int i = 0; i < *N; ++i)
    //     std::cout << (*nTime)[i] << '\t';
    // std::cout << std::endl;

    // std::cout << "\tnPos:" << std::endl;
    // for (int i = 0; i < *N; ++i){
    //     for (int j = 0; j < *M; ++j)
    //         std::cout << (*nPos)[i*(*M)+j] << '\t';
    //     std::cout << std::endl;
    // }

    return dof_names;
}

int IOFunctions::import_csv_file(const mjModel* m, const std::string filename, std::vector<int>& ja_indices, std::vector<std::string>& ja_names, std::vector<std::vector<mjtNum>>& joint_angles, std::vector<mjtNum>& times, const int verbose)
{
    ja_indices.clear();
    ja_names.clear();
    joint_angles.clear();
    times.clear();

    std::ifstream f;
    f.open(filename);

    if (!f.is_open()) {
        std::cout << "Could not open joint angle file " << filename << "." << std::endl;
        return -1;
    }
    CSVRow row;
    char buf_char[mjMAXUINAME];

    // make a local copy of model joint names
    std::vector<std::string> joint_names;
    for (int i = 0; i < m->njnt; i++)
    {
        mju_strncpy(buf_char, m->names + m->name_jntadr[i], mjMAXUINAME);
        joint_names.push_back(buf_char);
    }

    // process column headers
    int time_idx = -1;
    std::vector<int> ja_column_idxs;
    f >> row;
    for (size_t i_column = 0; i_column < row.size(); i_column++)
    {
        if (time_idx < 0 && !row[i_column].compare("time")) {
            time_idx = i_column;
            continue;
        }
        // find the DOF
        ja_names.push_back(std::string(row[i_column]));
        for (size_t i_jnt = 0; i_jnt < joint_names.size(); i_jnt++)
        {
            if (!row[i_column].compare(joint_names[i_jnt])) {
                ja_indices.push_back(i_jnt);
                ja_column_idxs.push_back(i_column);
                break;
            }
        }

        if (ja_names.size() != ja_indices.size()) {
            // did not find a corresponding joint
            std::cout << "Could not find a joint corresponding to DOF " << row[i_column] << " in the CSV." << std::endl;
            ja_names.pop_back();
        }

    }

    if (time_idx < 0) {
        std::cout << "Could not find time column in joint angle CSV." << std::endl;
        return -2;
    }

    int num_joints = ja_names.size();

    int irow = 0;
    while (f >> row)
    {
        // empty line in the end of the file or smth
        if (row.size() < 1 + ja_names.size())
            break;
        times.push_back(std::atof(std::string(row[time_idx]).c_str()));
        joint_angles.push_back(std::vector<mjtNum>(num_joints));
        for (size_t i_ja = 0; i_ja < num_joints; i_ja++)
        {
            joint_angles[irow][i_ja] = std::atof(std::string(row[ja_column_idxs[i_ja]]).c_str());
        }
        irow++;
    }

    if (verbose) {
        std::cout << "Loaded " << joint_names.size() << " joint angles for " << times.size() << " time points. Joint names:";
        for (auto jn : joint_names)
            std::cout << " " << jn;
        std::cout << "." << std::endl;
    }

    return 0;
}

int IOFunctions::import_csv_matrix(const std::string filename, std::vector<std::vector<std::vector<mjtNum>>>& matrix, std::vector<mjtNum>& times,
    const int verbose)
{
    matrix.clear();
    times.clear();

    std::ifstream f;
    f.open(filename);

    if (!f.is_open()) {
        std::cout << "Could not open matrix file " << filename << "." << std::endl;
        return -1;
    }
    CSVRow row;

    // process column headers
    f >> row;
    if (row[0].compare("times")) {
        std::cout << "The first column in matrix file is not times." << std::endl;
        return -2;
    }
    std::string lcn(row[row.size() - 1]);  // last column name
    std::regex e("(r)([0-9]+)(c)([0-9]+)");
    std::smatch sm;
    int rows;
    int cols;
    if (std::regex_match(lcn, sm, e)) {
        rows = std::atoi(std::string(sm[2]).c_str());
        cols = std::atoi(std::string(sm[4]).c_str());
    }
    else {
        std::cout << "Could not decode the number of rows and columns from the last column name " << lcn << "." << std::endl;
        return -3;
    }

    int iline = 0;
    while (f >> row)
    {
        // empty line in the end of the file or smth
        if (row.size() < 1 + rows * cols)
            break;
        // get time
        times.push_back(std::atof(std::string(row[0]).c_str()));
        // get pressure sensors
        matrix.push_back(std::vector<std::vector<mjtNum>>(rows, std::vector<mjtNum>(cols)));
        for (int ir = 0; ir < rows; ir++)
            for (int ic = 0; ic < cols; ic++)
                matrix[iline][ir][ic] = std::atof(std::string(row[1 + ir + rows * ic]).c_str());

        iline++;
    }

    if (verbose)
        std::cout << "Loaded " << rows << " by " << cols << " matrix for " << times.size() << " time points." << std::endl;

    return 0;
}

int IOFunctions::import_tsm_matrix(
    const std::string filename, 
    std::vector<std::vector<std::vector<mjtNum>>>& matrix, 
    std::vector<mjtNum>& times, 
    const int verbose)
{
    matrix.clear();
    times.clear();

    if (!fs::exists(filename)) {
        std::cout << "Matrix file " << filename << " does not exist." << std::endl;
        return -1;
    }

    TSM::Tsm tsm_file(filename);
    copy(tsm_file.time.begin(), tsm_file.time.end(), std::back_inserter(times));
    tsm_file.get_matrices_ip(matrix);

    if (verbose)
        std::cout << "Loaded " << tsm_file.dims[0] << " by " << tsm_file.dims[1] << " matrix for " << times.size() << " time points." << std::endl;

    return 0;
}

int IOFunctions::import_matrices(const std::string filename, std::vector<std::vector<std::vector<mjtNum>>>& matrix, std::vector<mjtNum>& times, const int verbose)
{
    if (fs::path(filename).extension().string() == ".tsm")
        return import_tsm_matrix(filename, matrix, times, verbose);
    else if (fs::path(filename).extension().string() == ".csv")
        return import_csv_matrix(filename, matrix, times, verbose);
    cout << "Could not identify matrices file extension." << endl;
    return -2;
}

int IOFunctions::export_adjustment_file(const std::string filename, const mjModel* m, const std::vector<std::vector<int>> fitting_dof_indices, const std::vector<double> x_vec)
{
    std::ofstream fo(filename);
    if (!fo.is_open()) {
        std::cout << "Could not open adjustment file for export." << std::endl;
        return -1;
    }

    char buf_char[mjMAXUINAME];
    
    // first row - names of joints
    for (auto fdi: fitting_dof_indices)
        for (const int& i_dof : fdi) {
            // find the name of the joint
            mju_strncpy(buf_char, m->names + m->name_jntadr[i_dof], mjMAXUINAME);
            fo << buf_char << ",";
        }
    fo << std::endl;

    // second row - associated adjustments
    for (auto fdi : fitting_dof_indices) {
        for (const double& x : x_vec)
            fo << x << ",";
    }
    fo << std::endl;

    fo.close();
    return 0;
}

int IOFunctions::import_adjustment_file(
    const std::string filename, 
    const std::vector<std::string>& ja_names, 
    std::vector<int>& adjusted_dof_indices, 
    std::vector<double>& adjustments)
{
    adjusted_dof_indices.clear();
    adjustments.clear();

    std::ifstream f;
    f.open(filename);

    if (!f.is_open()) {
        std::cout << "Could not open adjustment file " << filename << "." << std::endl;
        return -1;
    }
    CSVRow row;

    // process column headers
    int time_idx = -1;
    int nempty = 0;
    f >> row;
    for (size_t i_column = 0; i_column < row.size(); i_column++)
    {
        if (row[i_column].empty()) {
            nempty++;
            continue;
        }
        for (size_t i_ja = 0; i_ja < ja_names.size(); i_ja++)
        {
            if (!ja_names[i_ja].compare(row[i_column])) {
                adjusted_dof_indices.push_back(i_ja);
                break;
            }
        }
    }
    if (row.size() - nempty > adjusted_dof_indices.size()) {
        std::cout << "Could not interpret all columns of adjustment file. Aborting import of adjustment file." << std::endl;
        adjusted_dof_indices.clear();
        return -2;
    }

    // process values
    f >> row;
    for (size_t i_column = 0; i_column < row.size(); i_column++) {
        if (row[i_column].empty()) 
            continue;
        adjustments.push_back(std::atof(std::string(row[i_column]).c_str()));
    }

    if (adjustments.size() != adjusted_dof_indices.size()) {
        std::cout << "First two rows of adjustment file have different number of columns. Aborting import of adjustment file." << std::endl;
        adjusted_dof_indices.clear();
        adjustments.clear();
        return -3;
    }

    return 0;
}

int IOFunctions::export_optimization_logs(
    const std::string filename, 
    const mjModel* m, 
    const std::vector<std::vector<int>> fitting_dof_indices, 
    const std::vector<std::vector<double>> x_vecs, 
    const std::vector<double> fvals)
{
    std::ofstream fo(filename);
    if (!fo.is_open()) {
        std::cout << "Could not open optimization log file for export.";
        return -1;
    }

    char buf_char[mjMAXUINAME];

    fo << "fval";
    // first row - names of joints
    for (auto fdi : fitting_dof_indices)
        for (const int& i_dof : fdi) {
            // find the name of the joint
            mju_strncpy(buf_char, m->names + m->name_jntadr[i_dof], mjMAXUINAME);
            fo << "," << buf_char;
        }
    fo << std::endl;

    // second row and later - associated adjustments
    for (size_t i_trial = 0; i_trial < fvals.size() && i_trial < x_vecs.size(); i_trial++)
    {
        fo << fvals[i_trial];
        for (auto fdi : fitting_dof_indices) {
            for (const double& x : x_vecs[i_trial])
                fo << "," << x;
        }
        fo << std::endl;
    }

    fo.close();
    return 0;
}

int IOFunctions::export_timed_csv(
    const std::string filename, const std::vector<mjtNum>& times,
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<mjtNum>>& values) {

    std::ofstream csv;

    csv.open(filename, ofstream::out);
    csv << "time,";
    for (size_t i_cn = 0; i_cn < column_names.size() - 1; i_cn++) 
      csv << column_names[i_cn] << ",";
    csv << column_names.back() << std::endl;
    
    for (size_t i_time = 0; i_time < times.size(); i_time++) {
      csv << times[i_time] << ",";
      for (size_t i_cn = 0; i_cn < column_names.size() - 1; i_cn++)
        csv << values[i_time][i_cn] << ",";
      csv << values[i_time].back() << std::endl;
    }
    csv.close();

  return 0;
}
