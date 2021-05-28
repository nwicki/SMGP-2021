#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <vector>
#include <igl/svd3x3.h>
#include "FaceRegistor.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

void FaceRegistor::center_and_rescale_mesh(MatrixXd &V, const MatrixXd &P, double factor) {
    RowVector3d bc = P.colwise().mean();
    V = factor * (V.rowwise() - bc);
}

void FaceRegistor::center_and_rescale_template(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P) {
    RowVector3d tmpl_landmark_mean = P_tmpl.colwise().mean();
    RowVector3d face_landmark_mean = P.colwise().mean();
    double tmpl_avg2mean = (P_tmpl.rowwise() - tmpl_landmark_mean).rowwise().norm().mean();
    double face_avg2mean = (P.rowwise() - face_landmark_mean).rowwise().norm().mean();
    double factor = face_avg2mean / tmpl_avg2mean;
    center_and_rescale_mesh(V_tmpl, P_tmpl, factor);
}

void FaceRegistor::align_rigid(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P) {
    Matrix3d H = P_tmpl.transpose() * P; // both need to be centered first!
    Matrix3d U, V, R;
    Vector3d S;
    igl::svd3x3(H, U, S, V); // with this implementation, we know det U = det V = 1
    R = V * U.transpose();
    V_tmpl = V_tmpl * R.transpose();
}