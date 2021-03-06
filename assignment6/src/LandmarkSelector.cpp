#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>
#include <iostream>
#include "LandmarkSelector.h"
#include <string>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;
using Landmark = LandmarkSelector::Landmark;

void LandmarkSelector::display_landmarks(const vector<Landmark>& landmarks, const MatrixXd& V, const MatrixXi& F, Viewer& viewer) {
    viewer.data().labels_positions = MatrixXd(0,3);
    viewer.data().labels_strings.clear();
    int num_landmarks = landmarks.size();
    MatrixXd P(num_landmarks, 3);
    MatrixXd C(num_landmarks, 3);
    int index = 0;
    for (Landmark landmark: landmarks) {

        RowVector3d cartesian_point = landmark.get_cartesian_coordinates(V, F);

        P.row(index) << cartesian_point;
        C.row(index) << RowVector3d(0, 0, 1);
        viewer.data().add_label(cartesian_point,to_string(index + 1));

        index++;
    }
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
        new_landmark.bary0 = bc(0);
        new_landmark.bary1 = bc(1);
        new_landmark.bary2 = bc(2);
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
    ofstream myfile (filename);
    if (myfile.is_open()) {
        for (int i = 0; i < landmarks.size(); ++i) {
            Landmark landmark = landmarks[i];
            myfile << landmark.face_index << " " << landmark.bary0 << " " << landmark.bary1 << " " << landmark.bary2 << "\n";
        }
        myfile.close();
    } else {
        cout << "Error: Unable to open file" << endl;
    }
}

vector<Landmark> LandmarkSelector::get_landmarks_from_file(string filename) {
    vector<Landmark> deserialized_landmarks;

    string line;
    ifstream myfile (filename);
    if (myfile.is_open()) {
        int faceIndex;
        float bary0;
        float bary1;
        float bary2;
        while (myfile >> faceIndex >> bary0 >> bary1 >> bary2) {
            Landmark deserialized_landmark = Landmark();
            deserialized_landmark.face_index = faceIndex;
            deserialized_landmark.bary0 = bary0;
            deserialized_landmark.bary1 = bary1;
            deserialized_landmark.bary2 = bary2;

            deserialized_landmarks.push_back(deserialized_landmark);
        }
        myfile.close();
    } else {
        cout << "Error: Unable to open file" << endl;
    }
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

