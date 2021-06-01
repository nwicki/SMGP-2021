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
#include <boost/filesystem.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
namespace fs = boost::filesystem;
using Viewer = igl::opengl::glfw::Viewer;
using Landmark = LandmarkSelector::Landmark;
typedef Eigen::Triplet<double> T;

void FaceRegistor::fill_file_names(vector<string> &names,  string path, string extension) {
    names.clear();
    for (auto const & file : fs::recursive_directory_iterator(path)) {
        if (fs::is_regular_file(file) && file.path().extension() == extension)
            names.emplace_back(file.path().stem().string());
    }
    sort(names.begin(), names.end());
}

vector<Landmark> FaceRegistor::get_scan_landmarks() {
    string face_file_path = scan_folder_path + scan_names[scan_id] + "_landmarks.txt";
    return selector->get_landmarks_from_file(face_file_path);
}

MatrixXd FaceRegistor::get_scan_landmarks_matrix(const MatrixXd &V, const MatrixXi &F) {
    string face_file_path = scan_folder_path + scan_names[scan_id] + "_landmarks.txt";
    return selector->get_landmarks_from_file(face_file_path, V, F);
}

vector<Landmark> FaceRegistor::get_template_landmarks() {
    string tmpl_file_path = tmpl_folder_path + tmpl_names[tmpl_id] + "_landmarks.txt";
    return selector->get_landmarks_from_file(tmpl_file_path);
}

MatrixXd FaceRegistor::get_template_landmarks_matrix(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl) {
    string tmpl_file_path = tmpl_folder_path + tmpl_names[tmpl_id] + "_landmarks.txt";
    return selector->get_landmarks_from_file(tmpl_file_path, V_tmpl, F_tmpl);
}

string FaceRegistor::save_registered_scan(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl) {
    string save_file_path = save_folder_path + scan_names[scan_id] + "_aligned.obj";
    igl::writeOBJ(save_file_path, V_tmpl, F_tmpl);
    return save_file_path;
}

void FaceRegistor::center_and_rescale_mesh(MatrixXd &V, const MatrixXd &P, double factor) {
    RowVector3d bc = P.colwise().mean();
    V = factor * (V.rowwise() - bc);
}

void FaceRegistor::center_and_rescale_scan(MatrixXd &V, const MatrixXi &F) {
    MatrixXd P = get_scan_landmarks_matrix(V, F);
    center_and_rescale_mesh(V, P);
}

void FaceRegistor::center_and_rescale_template(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const MatrixXd &V, const MatrixXi &F) {
    MatrixXd P_tmpl = get_template_landmarks_matrix(V_tmpl, F_tmpl);
    MatrixXd P = get_scan_landmarks_matrix(V, F);
    RowVector3d tmpl_landmark_mean = P_tmpl.colwise().mean();
    RowVector3d face_landmark_mean = P.colwise().mean();
    double tmpl_avg2mean = (P_tmpl.rowwise() - tmpl_landmark_mean).rowwise().norm().mean();
    double face_avg2mean = (P.rowwise() - face_landmark_mean).rowwise().norm().mean();
    double factor = face_avg2mean / tmpl_avg2mean;
    center_and_rescale_mesh(V_tmpl, P_tmpl, factor);
}

void FaceRegistor::align_rigid(const MatrixXd &V_tmpl, const MatrixXi &F_tmpl, MatrixXd &V, const MatrixXi &F) {
    MatrixXd P_tmpl = get_template_landmarks_matrix(V_tmpl, F_tmpl);
    MatrixXd P = get_scan_landmarks_matrix(V, F);
    Matrix3d H = P_tmpl.transpose() * P; // both need to be centered first!
    Matrix3d U, W, R;
    Vector3d S;
    igl::svd3x3(H, U, S, W); // with this implementation, we know det U = det V = 1
    R = W * U.transpose();
    V = V * R;
}

void FaceRegistor::align_non_rigid_step(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, const MatrixXd &V, const MatrixXi &F) {
    // Fetch landmarks
    vector<Landmark> landmarks_tmpl = get_template_landmarks();
    MatrixXd P = get_scan_landmarks_matrix(V, F);
    
    // Laplacian matrix 
    SparseMatrix<double> Laplacian;
    igl::cotmatrix(V_tmpl, F_tmpl, Laplacian);
    MatrixXd Lx = Laplacian * V_tmpl;
    //cout << "Laplacian matrix done: L: " << Laplacian.rows() << " x " << Laplacian.cols() << " Lx: " << Lx.rows() << " x " << Lx.cols() << endl;

    // Boundary constraints
    SparseMatrix<double> Csb; // Csb = boundary constraints
    MatrixXd Cwsb; // corresponding right hand side
    VectorXi Bi; // boundary indices
    igl::boundary_loop(F_tmpl, Bi);
    SparseMatrix<double> Id(V_tmpl.rows(), V_tmpl.rows());
    Id.setIdentity();
    igl::slice(Id, Bi, 1, Csb);
    igl::slice(V_tmpl, Bi, 1, Cwsb);
    MatrixXi Bi_mask = MatrixXi::Ones(V_tmpl.rows(), 1);
    igl::slice_into(MatrixXi::Zero(Bi.rows(), 1), Bi, 1, Bi_mask);
    //cout << "boundary constraints done: Csb: " << Csb.rows() << " x " << Csb.cols() << " Cwsb: " << Cwsb.rows() << " x " << Cwsb.cols() << endl;

    // Target landmarks constraints
    SparseMatrix<double> Csl(23, V_tmpl.rows()); // Csl = landmark constraints (target)
    MatrixXd Cwsl = P; // corresponding right hand side
    vector<T> tripletList;
    tripletList.reserve(23 * 3);
    int index = 0;
    for(LandmarkSelector::Landmark landmark : landmarks_tmpl) {
        RowVector3i vi = F_tmpl.row(landmark.face_index);
        Vector3f bc(landmark.bary0, landmark.bary1, landmark.bary2);
        tripletList.push_back(T(index, vi(0), bc(0)));
        tripletList.push_back(T(index, vi(1), bc(1)));
        tripletList.push_back(T(index, vi(2), bc(2)));
        index++;
    }
    Csl.setFromTriplets(tripletList.begin(), tripletList.end());
    //cout << "landmarks constraints done: Csl: " << Csl.rows() << " x " << Csl.cols() << " Cwsl: " << Cwsl.rows() << " x " << Cwsl.cols() << endl;

    // Query dynamic constraints (close to target face)
    VectorXi I(V_tmpl.rows());
    for(int i=0; i<V_tmpl.rows(); i++) {
        vector<size_t> ret_index(1);
        vector<double> out_dist_sqr(1);
        KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        kd_tree->index->findNeighbors(resultSet, RowVector3d(V_tmpl.row(i)).data(), SearchParams(10));
        I(i) = ret_index[0];
    }
    SparseMatrix<double> Cd;
    MatrixXd C, Cwd; // C = position of nearest neighbor (#V_tmpl x 3), Cwd = reduced C
    igl::slice(V, I, 1, C);
    Array<bool, Dynamic, 1> c_mask = ((V_tmpl - C).rowwise().norm().array() < m_epsilon) * (Bi_mask.cast<bool>().array());
    igl::slice_mask(C, c_mask, 1, Cwd);
    igl::slice_mask(Id, c_mask, 1, Cd);
    //cout << "dynamic constraints done: Cd: " << Cd.rows() << " x " << Cd.cols() << " Cwd: " << Cwd.rows() << " x " << Cwd.cols() << endl;

    // Set up whole matrix and right hand side
    SparseMatrix<double> Snd = useLandmarks? m_lambda * igl::cat(1, Csb, igl::cat(1, Csl, Cd)) : m_lambda * igl::cat(1, Csb, Cd);
    SparseMatrix<double> A = igl::cat(1, Laplacian, Snd);
    MatrixXd b(A.rows(), 3);
    if(useLandmarks){
        b << Lx, m_lambda * Cwsb, m_lambda * Cwsl, m_lambda * Cwd;
    } else {
        b << Lx, m_lambda * Cwsb, m_lambda * Cwd;
    }
    //cout << "whole matrix set up done: A: " << A.rows() << " x " << A.cols() << " b: " << b.rows() << " x " << b.cols() << endl;

    // Solve system
    Eigen::SimplicialLDLT<SparseMatrix<double> > solver; //solve for A'Ax = A'b instead of Ax = b
    solver.compute(A.transpose() * A);
    //cout << "Solver compute success: " << int(solver.info() == Success) << endl;
    MatrixXd V_sol = solver.solve(A.transpose() * b);
    cout << "Solver solve success: " << int(solver.info() == Success) << endl;
    V_tmpl = V_sol;
    //cout << "system solve done: V_sol: " << V_sol.rows() << " x " << V_sol.cols() << endl;

}

void FaceRegistor::build_octree(const MatrixXd &V) {
    //cout << "start build octree" << endl;
    kd_tree = new KDTree(3, cref(V), 10);
    kd_tree->index->buildIndex();
    //cout << "done build octree" << endl;
}

void FaceRegistor::subdivide_template(MatrixXd &V_tmpl, MatrixXi &F_tmpl) {
    Eigen::MatrixXd Vout=V_tmpl;
    Eigen::MatrixXi Fout=F_tmpl;
    // Step 1-1: Compute V" (store in Vout)
    Vout.conservativeResize(V_tmpl.rows() + F_tmpl.rows(), Eigen::NoChange);
    Eigen::MatrixXd M(F_tmpl.rows(), 3);
    igl::barycenter(V_tmpl, F_tmpl, M);
    Vout.bottomRows(F_tmpl.rows()) = M;

    // Step 1-2: Compute F" (store in Fout)
    Fout.resize(3*F_tmpl.rows(), Eigen::NoChange);
    for(int i=0; i<F_tmpl.rows(); i++){
        int mi = V_tmpl.rows() + i; //vertex index of face midpoint
        for(int k=0; k<3; k++){
        Fout.row(3*i+k) = Eigen::Vector3i(F_tmpl(i,k), F_tmpl(i,(k+1)%3), mi);
        }
    }

    // Step 2: Update positions of original vertices (update Vout)
     vector<vector<int> > VV;
    igl::adjacency_list(F_tmpl, VV);
    for(int i=0; i<VV.size(); i++) {
        const auto& v = VV[i]; //vector of neighbors
        int n = v.size();
        double an = (4.0 - 2.0*std::cos(2*M_PI / n)) / 9.0;
        Eigen::RowVector3d si(0,0,0); //sum of vi
        for (const int& vi : v) {
        si += V_tmpl.row(vi);
        }
        Vout.row(i) = (1-an)*Vout.row(i) + (an/n)*si;
    }

    // Step 3: Flip original edges (update Fout)
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(Fout, TT, TTi);
    for(int i=0; i<Fout.rows(); i++){
        if(TT(i,0) != -1){ //not a boundary edge
        //replace first point by "opposite" midpoint
        Fout(i,0) = Fout(TT(i,0),2);
        }
    }
    
    // Set up the viewer to display the new mesh
    V_tmpl = Vout; F_tmpl = Fout;
}

void FaceRegistor::register_face(MatrixXd &V_tmpl, const MatrixXi &F_tmpl, MatrixXd &V, const MatrixXi &F, int num_iter, float lambda, float epsilon1, float epsilon2) {
    center_and_rescale_scan(V, F);
    center_and_rescale_template(V_tmpl, F_tmpl, V, F);
    align_rigid(V_tmpl, F_tmpl, V, F);
    build_octree(V);

    m_lambda = lambda;
    m_epsilon = epsilon1;
    align_non_rigid_step(V_tmpl, F_tmpl, V, F);

    m_epsilon = epsilon2;
    for(int i=0; i<num_iter-1; i++){
        align_non_rigid_step(V_tmpl, F_tmpl, V, F);
    }
}