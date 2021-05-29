#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <vector>
#include <igl/svd3x3.h>
#include <igl/octree.h>
#include <igl/slice_mask.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/knn.h>
#include <igl/cat.h>
#include "FaceRegistor.h"

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using Viewer = igl::opengl::glfw::Viewer;
typedef Eigen::Triplet<double> T;

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

void FaceRegistor::align_non_rigid_step(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const vector<LandmarkSelector::Landmark> &landmarks_tmpl, const MatrixXd &V, const MatrixXd &P, float lambda) {
    // Query dynamic constraints (close to target face)
    double scale = (V_tmpl.colwise().maxCoeff() - V_tmpl.colwise().minCoeff()).norm();
    double epsilon = 1e-2 * scale;
    VectorXi I(V_tmpl.rows());
    for(int i=0; i<V_tmpl.rows(); i++) {
        vector<size_t> ret_index(1);
        vector<double> out_dist_sqr(1);
        KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        kd_tree->index->findNeighbors(resultSet, V_tmpl.row(i).data(), SearchParams(10));
        I(i) = ret_index[0];
    }
    cout << "done knn" << endl;
    MatrixXd C, Cwd; // C = position of nearest neighbor (#V_tmpl x 3), Cwd = reduced C
    igl::slice(V, I, 1, C);
    igl::slice_mask(C, (V_tmpl - C).rowwise().norm().array() < epsilon, 1, Cwd);
    cout << "slice mask done" << endl;
    SparseMatrix<double> Id(V_tmpl.rows(), V_tmpl.rows()), Cd;
    Id.setIdentity();
    igl::slice_mask(Id, (V_tmpl - C).rowwise().norm().array() < epsilon, 1, Cd);
    cout << "dynamic constraints done: Cd: " << Cd.rows() << " x " << Cd.cols() << " Cwd: " << Cwd.rows() << " x " << Cwd.cols() << endl;

    // Laplacian matrix 
    SparseMatrix<double> Laplacian;
    igl::cotmatrix(V_tmpl, F_tmpl, Laplacian);
    MatrixXd Lx = Laplacian * V_tmpl;
    cout << "Laplacian matrix done: L: " << Laplacian.rows() << " x " << Laplacian.cols() << " Lx: " << Lx.rows() << " x " << Lx.cols() << endl;


    // Boundary constraints
    SparseMatrix<double> Csb; // Csb = boundary constraints (original)
    MatrixXd Cwsb;
    VectorXi Bi; // boundary indices
    igl::boundary_loop(F_tmpl, Bi);
    igl::slice(Id, Bi, 1, Csb);
    igl::slice(V_tmpl, Bi, 1, Cwsb);
    cout << "boundary constraints done: Csb: " << Csb.rows() << " x " << Csb.cols() << " Cwsb: " << Cwsb.rows() << " x " << Cwsb.cols() << endl;


    // Target landmarks constraints
    SparseMatrix<double> Csl(23, V_tmpl.rows()); // Csl = landmark constraints (target)
    MatrixXd Cwsl = P;
    vector<T> tripletList;
    tripletList.reserve(23 * 3);
    int index = 0;
    for(LandmarkSelector::Landmark landmark : landmarks_tmpl) {
        RowVector3i vi = F_tmpl.row(landmark.face_index);
        Vector3f bc = landmark.bary_coords;
        tripletList.push_back(T(index, vi(0), bc(0)));
        tripletList.push_back(T(index, vi(1), bc(1)));
        tripletList.push_back(T(index, vi(2), bc(2)));
        index++;
    }
    Csl.setFromTriplets(tripletList.begin(), tripletList.end());
    cout << "landmarks constraints done: Csl: " << Csl.rows() << " x " << Csl.cols() << " Cwsl: " << Cwsl.rows() << " x " << Cwsl.cols() << endl;


    // Set up whole matrix and right hand side
    SparseMatrix<double> Snd = lambda * igl::cat(1, Csb, igl::cat(1, Csl, Cd));
    SparseMatrix<double> A = igl::cat(1, Laplacian, Snd);
    MatrixXd b(A.rows(), 3);
    b << Lx, lambda * Cwsb, lambda * Cwsl, lambda * Cwd;
    cout << "whole matrix set up done: A: " << A.rows() << " x " << A.cols() << " b: " << b.rows() << " x " << b.cols() << endl;

    // Solve system
    // SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > solver;
    LeastSquaresConjugateGradient<SparseMatrix<double> > solver;
    solver.compute(A);
    cout << "Solver compute success: " << int(solver.info() == Success) << endl;
    MatrixXd V_sol = solver.solve(b);
    cout << "Solver solve success: " << int(solver.info() == Success) << endl;
    V_tmpl = V_sol;
    cout << "system solve done: V_sol: " << V_sol.rows() << " x " << V_sol.cols() << endl;

}

void FaceRegistor::build_octree(const MatrixXd &V) {
    cout << "start build octree" << endl;
    kd_tree = new KDTree(3, cref(V), 10);
    kd_tree->index->buildIndex();
    cout << "done build octree" << endl;
}