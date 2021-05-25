#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>
#include<string>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);
bool is_selection_enabled = false;

struct Landmark: public igl::Serializable {
    int face_index;
    Vector3f bary_coords;

    void InitSerialization(){
        this->Add(face_index  , "face_index");
        this->Add(bary_coords  , "bary_coords");
    }

    RowVector3d get_cartesian_coordinates(const MatrixXd& V, const MatrixXi& F) {
        RowVector3i vertex_indices = F.row(face_index);
        RowVector3d p0 = V.row(vertex_indices(0));
        RowVector3d p1 = V.row(vertex_indices(1));
        RowVector3d p2 = V.row(vertex_indices(2));
        return bary_coords(0) * p0 + bary_coords(1) * p1 + bary_coords(2) * p2;
    }
};

vector<Landmark> landmarks;



void display_landmarks(const vector<Landmark>& landmarks) {
    cout << "Display landmarks" << endl;

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

void add_landmark_at_mouse_position() {

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

        // Add landmark to landmarks
        Landmark new_landmark = Landmark();
        new_landmark.face_index = fid;
        new_landmark.bary_coords = Vector3f(bc(0), bc(1), bc(2));
        landmarks.push_back(new_landmark);

        // Display new landmark
        display_landmarks(landmarks);
    }
}

bool callback_mouse_down(Viewer &viewer, int button, int modifier) {

    if (button == (int) Viewer::MouseButton::Right)
        return false;

    if (!is_selection_enabled) {
        return false;
    }

    add_landmark_at_mouse_position();

    return true;
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        display_landmarks(landmarks);
    }

    if (key == '2') {
        cout << "Delete last added landmark" << endl;
        landmarks.pop_back();
        display_landmarks(landmarks);
    }

    return true;
}

bool load_mesh(string filename) {
    igl::read_triangle_mesh(filename, V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core.align_camera_center(V);
    return true;
}

void save_landmarks_to_file(vector<Landmark> landmarks, string filename) {
    igl::serialize(landmarks, "landmarks", filename, true);
    cout << "landmarks saved to my-landmarks" << endl;
}

vector<Landmark> get_landmarks_from_file(string filename) {
    vector<Landmark> deserialized_landmarks;
    igl::deserialize(deserialized_landmarks, "landmarks", filename);
    return deserialized_landmarks;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        string folder_path = "../data/aligned_faces_example/example2/";
        string file_name = "alain_normal.obj";
        load_mesh(folder_path + file_name);
    } else {
        load_mesh(argv[1]);
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Instructions", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Checkbox("Enable Selection", &is_selection_enabled)) {
                cout << "Enable Selection = " << is_selection_enabled << endl;
            }

            if (ImGui::Button("Save Landmarks", ImVec2(-1, 0))) {
                save_landmarks_to_file(landmarks, "my-landmarks");
            }

            if (ImGui::Button("Load Landmarks", ImVec2(-1, 0))) {
                landmarks = get_landmarks_from_file("my-landmarks");
            }
        }
    };

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;

    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();

    return 0;

}