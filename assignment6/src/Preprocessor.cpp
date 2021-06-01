#include <igl/opengl/glfw/Viewer.h>
#include <igl/remesh_along_isoline.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/facet_components.h>
#include <igl/remove_unreferenced.h>
#include <igl/collapse_small_triangles.h>
#include <igl/slice_mask.h>
#include <igl/slice_into.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/invert_diag.h>
#include <igl/read_triangle_mesh.h>
#include <igl/signed_distance.h>
#include <igl/boundary_facets.h>
#include <igl/on_boundary.h>
#include <igl/repmat.h>
#include <igl/exact_geodesic.h>
#include <igl/cut_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/boundary_loop.h>
#include <igl/is_edge_manifold.h>
#include <vector>
#include "Preprocessor.h"

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using Viewer = igl::opengl::glfw::Viewer;

typedef KDTreeEigenMatrixAdaptor<MatrixXd> KDTree;

void Preprocessor::clean_connected_components(Viewer &viewer, MatrixXd &V, MatrixXi &F) {
    // build triangle triangle adjacency
    std::vector<std::vector<std::vector<int> > > TT;
    igl::triangle_triangle_adjacency(F, TT);

    // compute connected components
    VectorXi C, counts;
    igl::facet_components(TT, C, counts);
    cout << "Found " << counts.rows() << " connected components" << endl;

    // find largest component id
    int max, max_id;
    max = counts.maxCoeff(&max_id);

    // keep only faces of largest component
    MatrixXi NF;
    igl::slice_mask(F, (C.array() == max_id), 1, NF);

    // remove unreferenced vertices
    VectorXi nIM;
    MatrixXd oldV = V;
    igl::remove_unreferenced(oldV, NF, V, F, nIM);

    // draw new mesh
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
}

void Preprocessor::compute_distance_to_boundary(Viewer& viewer, MatrixXd &V, MatrixXi&F) {
    // find boundary vertices
    VectorXi L;
    igl::boundary_loop(F, L);
    MatrixXd BV;
    igl::slice(V, L, 1, BV);

    // build KD-Tree for nearest neighbor search
    KDTree kd_tree(3, cref(BV), 10);
    kd_tree.index->buildIndex();

    // for each vertex in mesh compute distance to boundary
    scalar_field.resize(V.rows());
    for(int i=0; i<V.rows(); i++) {
        vector<size_t> ret_index(1);
        vector<double> out_dist_sqr(1);
        KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        kd_tree.index->findNeighbors(resultSet, RowVector3d(V.row(i)).data(), SearchParams(10));
        scalar_field(i) = (V.row(i) - BV.row(ret_index[0])).norm();
    }
    
    // display scalar field
    MatrixXd C;
    igl::colormap(igl::COLOR_MAP_TYPE_JET, scalar_field, true, C);
    max_dist = scalar_field.maxCoeff();
    cout << "Scalar field distance range = [ " << scalar_field.minCoeff() << " ; " << max_dist << " ]" << endl;
    viewer.data().set_colors(C);
}

void Preprocessor::smooth_distance_field(Viewer& viewer, MatrixXd &V, MatrixXi&F, int num_iter, float w) {
    if(scalar_field.rows() != V.rows()){
        cout << "Smooth: scalar_field has wrong size!" << endl;
        return;
    }
    // smooth scalar distance field by energy optimization
    for (int i=0; i< num_iter; i++) {
        cout << "Smoothing distance field iteration " << i << endl;
        Eigen::SparseMatrix<double> L, M;
        igl::cotmatrix(V, F, L);
        igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
        // Solve (L'ML + w*M) X = w*M X
        const auto & A = L.transpose() * M * L + w * M;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(A);
        assert(solver.info() == Eigen::Success);
        scalar_field = solver.solve(w * M*scalar_field).eval();  
        assert(solver.info() == Eigen::Success);
    }

    // clip values to [0, infty]
    scalar_field = scalar_field.cwiseMax(0.0);

    // display result
    MatrixXd C;
    igl::colormap(igl::COLOR_MAP_TYPE_JET, scalar_field, true, C);
    max_dist = scalar_field.maxCoeff();
    cout << "Scalar field distance range = [ " << scalar_field.minCoeff() << " ; " << max_dist << " ]" << endl;
    viewer.data().set_colors(C);
}

void Preprocessor::remesh(Viewer& viewer, MatrixXd &V, MatrixXi&F) {
    if(scalar_field.rows() != V.rows()){
        cout << "Remesh: scalar_field has wrong size!" << endl;
        return;
    }
    cout << "Is edge manifold (initial):                " << (igl::is_edge_manifold(F) ? "true" : "false") << endl;

    // remesh along scalar field isoline V, F -> U, G
    MatrixXd U;
    MatrixXi G;
    VectorXd SU;
    VectorXi J;
    SparseMatrix<double> BC;
    VectorXd L;
    igl::remesh_along_isoline(V, F, scalar_field, iso_value, U, G, SU, J, BC, L);
    cout << "Is edge manifold (after remesh):           " << (igl::is_edge_manifold(G) ? "true" : "false") << endl;

    // remeshing creates duplicate vertices, clean them U, G -> SV, SF
    MatrixXd SV;
    VectorXi SVI, SVJ;
    MatrixXi SF;
    igl::remove_duplicate_vertices(U, G, 1e-10, SV, SVI, SVJ, SF);

    cout << "Is edge manifold (after remove duplicate): " << (igl::is_edge_manifold(SF) ? "true" : "false") << endl;
    if(!igl::is_edge_manifold(SF)){
        cout << "Remesh: mesh is not edge manifold, cannot perform cut!" << endl;
        return;
    }

    // define edges to be cut and cut the mesh SV, SF -> V, F
    MatrixXi B(SF.rows(), 3); // boolean indicating edges to cut (#F x 3)
    int nf = SF.rows();
    for(int i=0; i<nf; i++) {
        if(J((i+1)%nf) == J(i)) { //first and second new faces
            B.row(i) = RowVector3i(0, 0, 1); 
        } else { //third new face or old faces
            B.row(i) = RowVector3i(0, 0, 0);
        }
    }

    if(B.sum() == 0){
        cout << "Remesh: no edge to cut" << endl;
        return;
    }

    // cut mesh and show
    igl::cut_mesh(SV, SF, B, V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
}

void Preprocessor::preprocess(Viewer& viewer, MatrixXd &V, MatrixXi&F, int num_iter, float w) {
    clean_connected_components(viewer, V, F);
    compute_distance_to_boundary(viewer, V, F);
    smooth_distance_field(viewer, V, F, num_iter, w);
    double prev_iso = iso_value;
    iso_value = 0.03 * max_dist;
    remesh(viewer, V, F);
    clean_connected_components(viewer, V, F);
    iso_value = prev_iso;
}

string Preprocessor::save_mesh(MatrixXd &V, MatrixXi&F) {
    string save_file_path = save_folder_path + mesh_names[mesh_id] + "_preprocessed.obj";
    igl::writeOBJ(save_file_path, V, F);
    return save_file_path;
}
