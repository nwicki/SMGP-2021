#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>
#include <igl/xml/serialize_xml.h>
#include "LandmarkSelector.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;
using Landmark = LandmarkSelector::Landmark;

void LandmarkSelector::display_landmarks(const vector<Landmark>& landmarks, const MatrixXd& V, const MatrixXi& F, Viewer& viewer) {
    int num_landmarks = landmarks.size();
    MatrixXd P(num_landmarks, 3);
    MatrixXd C(num_landmarks, 3);
    int index = 0;
    for (Landmark landmark: landmarks) {

        RowVector3d cartesian_point = landmark.get_cartesian_coordinates(V, F);

        P.row(index) << cartesian_point;
        C.row(index) << RowVector3d(1, 0, 0);

        index++;
    }
    viewer.data().set_points(P, C);
}

void LandmarkSelector::clear_landmarks_from_viewer(Viewer& viewer) {
    MatrixXd P(1, 3);
    MatrixXd C(1, 3);
    viewer.data().set_points(P, C);
}

void LandmarkSelector::add_landmark_at_mouse_position(const MatrixXd& V, const MatrixXi& F, Viewer& viewer) {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, V, F,
                                    fid, bc)) {
        Eigen::RowVector3i face_point_indices = F.row(fid);
        Eigen::RowVector3d p0 = V.row(face_point_indices(0));
        Eigen::RowVector3d p1 = V.row(face_point_indices(1));
        Eigen::RowVector3d p2 = V.row(face_point_indices(2));

        // Add landmark to current_landmarks
        Landmark new_landmark = Landmark();
        new_landmark.face_index = fid;
        new_landmark.bary_coords = Vector3f(bc(0), bc(1), bc(2));
        current_landmarks.push_back(new_landmark);

        // Display new landmark
        display_landmarks(current_landmarks, V, F, viewer);
    }
}

void LandmarkSelector::delete_last_landmark() {
    if (!current_landmarks.empty()) {
        current_landmarks.pop_back();
    }
}

void LandmarkSelector::delete_all_landmarks() {
    current_landmarks = {};
}

void LandmarkSelector::save_landmarks_to_file(vector<Landmark> landmarks, string filename) {
    igl::xml::serialize_xml(landmarks, "landmarks", filename, false, true);
}

vector<Landmark> LandmarkSelector::get_landmarks_from_file(string filename) {
    vector<Landmark> deserialized_landmarks;
    igl::deserialize(deserialized_landmarks, "landmarks", filename);
    return deserialized_landmarks;
}

MatrixXd LandmarkSelector::get_landmarks_from_file(string filename, const MatrixXd& V, const MatrixXi& F) {
    vector<Landmark> landmarks = get_landmarks_from_file(filename);
    int num_landmarks = landmarks.size();
    MatrixXd P(num_landmarks, 3);
    int index = 0;
    for (Landmark landmark: landmarks) {
        P.row(index) << landmark.get_cartesian_coordinates(V, F);
        index++;
    }
    return P;
}

