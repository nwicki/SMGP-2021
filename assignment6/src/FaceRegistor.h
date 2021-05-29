#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <vector>
#include <nanoflann.hpp>
#include "LandmarkSelector.h"

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using Viewer = igl::opengl::glfw::Viewer;

typedef KDTreeEigenMatrixAdaptor<MatrixXd> KDTree;

class FaceRegistor {
private:
    KDTree *kd_tree;
public:

    void center_and_rescale_mesh(MatrixXd &V, const MatrixXd &P, double factor = 1.0);

    void center_and_rescale_template(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P);

    void align_rigid(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P);

    void align_non_rigid_step(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const vector<LandmarkSelector::Landmark> &landmarks_tmpl, const MatrixXd &V, const MatrixXd &P, float lambda);

    void build_octree(const MatrixXd &V);

};