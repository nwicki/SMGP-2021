#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <vector>
#include <nanoflann.hpp>
#include "LandmarkSelector.h"
#include <boost/filesystem.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
namespace fs = boost::filesystem;
using Viewer = igl::opengl::glfw::Viewer;
using Landmark = LandmarkSelector::Landmark;
typedef Eigen::Triplet<double> T;

typedef KDTreeEigenMatrixAdaptor<MatrixXd> KDTree;

class FaceRegistor {
private:
    KDTree *kd_tree;
    LandmarkSelector* selector;
public:
    string scan_folder_path = "../data/scanned_faces_cleaned/";
    vector<string> scan_names;
    int scan_id = 0;

    string tmpl_folder_path = "../data/face_template/";
    vector<string> tmpl_names;
    int tmpl_id = 3;

    string save_folder_path = "../data/aligned_faces/";

    float m_lambda = 1.0f;
    float m_epsilon = 0.01f;
    bool useLandmarks = true;

    FaceRegistor(LandmarkSelector* landmarkSelector) : selector(landmarkSelector) {
        fill_file_names(scan_names, "../data/scanned_faces_cleaned/", ".obj");
        fill_file_names(tmpl_names, "../data/face_template/", ".obj");
    }

    void fill_file_names(vector<string> &names,  string path, string extension);

    vector<Landmark> get_scan_landmarks();

    MatrixXd get_scan_landmarks_matrix(const MatrixXd &V, const MatrixXi &F);

    vector<Landmark> get_template_landmarks();

    MatrixXd get_template_landmarks_matrix(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl);

    string save_registered_scan(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl);

    void center_and_rescale_mesh(MatrixXd &V, const MatrixXd &P, double factor=1.0);

    void center_and_rescale_scan(MatrixXd &V, const MatrixXi &F);

    void center_and_rescale_template(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const MatrixXd &V, const MatrixXi &F);

    void align_rigid(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl, MatrixXd &V, const MatrixXi &F);

    void align_non_rigid_step(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const MatrixXd &V, const MatrixXi &F);

    void build_octree(const MatrixXd &V);

    void subdivide_template(MatrixXd &V_tmpl, MatrixXi &F_tmpl); //not used currently

    void register_face(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, MatrixXd &V, const MatrixXi &F, int num_iter = 5, float lambda = 1.0f, float epsilon1 = 0.01f, float epsilon2 = 3.0f);

};