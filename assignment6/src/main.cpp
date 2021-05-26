#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include<string>
#include "LandmarkSelector.cpp"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);
bool is_selection_enabled = false;
LandmarkSelector landmarkSelector = LandmarkSelector();
string folder_path = "../data/aligned_faces_example/example2/";
string filename = "alain_normal";
string file_extension = ".obj";

bool callback_mouse_down(Viewer &viewer, int button, int modifier) {

    if (button == (int) Viewer::MouseButton::Right)
        return false;

    if (!is_selection_enabled) {
        return false;
    }

    landmarkSelector.add_landmark_at_mouse_position(V, F, viewer);

    return true;
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    return true;
}

bool load_mesh(string filename) {
    igl::read_triangle_mesh(filename, V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core.align_camera_center(V);
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        load_mesh(folder_path + filename + file_extension);
    } else {
        load_mesh(argv[1]);
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Landmark Selection", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Checkbox("Enable Selection", &is_selection_enabled)) {
                string message = (is_selection_enabled) ? "Selection Enabled" : "Selection Disabled";
                cout << message << endl;
            }

            if (ImGui::Button("Save Landmarks", ImVec2(-1, 0))) {
                string file_path = folder_path + filename + "_landmarks";
                landmarkSelector.save_landmarks_to_file(landmarkSelector.current_landmarks, file_path);
                cout << landmarkSelector.current_landmarks.size() << " landmarks saved to " << file_path << endl;
            }

            if (ImGui::Button("Delete last landmark", ImVec2(-1, 0))) {
                cout << "Delete last added landmark" << endl;
                landmarkSelector.delete_last_landmark();
                landmarkSelector.display_landmarks(landmarkSelector.current_landmarks, V, F, viewer);
            }

            if (ImGui::Button("Load Landmarks", ImVec2(-1, 0))) {
                string file_path = folder_path + filename + "_landmarks";
                landmarkSelector.current_landmarks = landmarkSelector.get_landmarks_from_file(file_path);
                cout << landmarkSelector.current_landmarks.size() << " landmarks loaded from " << file_path << endl;
            }

            if (ImGui::Button("Display Landmarks", ImVec2(-1, 0))) {
                landmarkSelector.display_landmarks(landmarkSelector.current_landmarks, V, F, viewer);
                cout << "Display landmarks" << endl;
            }

            if (ImGui::Button("Clear Landmarks From Viewer", ImVec2(-1, 0))) {
                landmarkSelector.clear_landmarks_from_viewer(viewer);
                cout << "Clear landmarks from viewer" << endl;
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