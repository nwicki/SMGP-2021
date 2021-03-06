#pragma once

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

class LandmarkSelector {
public:

    struct Landmark {
        int face_index;

        float bary0;
        float bary1;
        float bary2;

        template<class Archive>
        void serialize(Archive & archive)
        {
            archive( face_index, bary0, bary1, bary2 );
        }

        RowVector3d get_cartesian_coordinates(const MatrixXd& V, const MatrixXi& F) {
            RowVector3i vertex_indices = F.row(face_index);
            RowVector3d p0 = V.row(vertex_indices(0));
            RowVector3d p1 = V.row(vertex_indices(1));
            RowVector3d p2 = V.row(vertex_indices(2));
            return bary0 * p0 + bary1 * p1 + bary2 * p2;
        }
    };

    vector<Landmark> current_landmarks;

    void display_landmarks(const vector<Landmark>& landmarks, const MatrixXd& V, const MatrixXi& F, Viewer& viewer);

    void clear_landmarks_from_viewer(Viewer& viewer);

    void add_landmark_at_mouse_position(const MatrixXd& V, const MatrixXi& F, Viewer& viewer);

    void delete_last_landmark();

    void delete_all_landmarks();

    void save_landmarks_to_file(vector<Landmark> landmarks, string filename);

    vector<Landmark> get_landmarks_from_file(string filename);

    MatrixXd get_landmarks_from_file(string filename, const MatrixXd& V, const MatrixXi& F);
};
