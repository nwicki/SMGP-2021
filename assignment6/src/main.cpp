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

// Vertex array, #V x3
Eigen::MatrixXd V(0, 3);
// Face array, #F x3
Eigen::MatrixXi F(0, 3);

// Landmark Selection
bool is_selection_enabled = false;
LandmarkSelector landmarkSelector = LandmarkSelector();

// Mesh Loading
string folder_path = "../data/face_template/";
string filename = "headtemplate";
string file_extension = ".obj";

const string filenames[] = {
        "alain_normal",
        "michi-smile",
        "moyuan_duckface",
        "alain_smile",
        "moyuan_neutral",
        "nici-brille",
        "alex neutral",
        "nici-neutral",
        "nici-smile",
        "alex smile",
        "nick neutral",
        "alex wacky",
        "nick smile",
        "ali_neutral_corrected",
        "nick wacky",
        "ali_smile_corrected",
        "nihat neutral",
        "arda_neutral_corrected",
        "nihat smile",
        "arda_smile_corrected",
        "nihat wacky",
        "bjarni_neutral",
        "patrick_neutral_corrected",
        "bjarni_smile",
        "patrick_smile_corrected",
        "chrisk_neutral",
        "person1_normal",
        "chrisk_neutral2",
        "person1_smile ",
        "chrisk_smile",
        "person2_normal ",
        "chriss_glasses",
        "person2_smile ",
        "chriss_neutral",
        "person3_normal",
        "chriss_smile",
        "person3_smile",
        "christian_neutral_corrected",
        "person4_normal",
        "christian_smile_corrected",
        "person4_smile",
        "daniel_normal",
        "person5_normal",
        "daniel_smile",
        "person5_smile",
        "dingguang_normal",
        "person6_normal",
        "dingguang_smile",
        "person6_smile",
        "fabian-brille",
        "pietro_normal",
        "fabian-neutral",
        "pietro_smile",
        "fabian-smile",
        "qais_neutral_corrected",
        "felix neutral smoothed",
        "qais_smile_corrected",
        "felix smile smoothed",
        "ryan neutral smoothed",
        "felix wacky smoothed",
        "ryan smiloe smoothed",
        "gleb_neutral",
        "selina-brille",
        "gleb_smile",
        "selina-neutral",
        "ho_neutral",
        "selina-smile",
        "ho_smile",
        "shanshan_neutral_corrected",
        "jan-brille",
        "shanshan_smile_corrected",
        "jan-neutral",
        "simon-brille",
        "jan-smile",
        "simon-neutral",
        "julian_normal",
        "simon-smile",
        "julian_smile",
        "simonh_neutral_corrected",
        "karlis_neutral_corrected",
        "simonh_smile_corrected",
        "karlis_smile_corrected",
        "simonw_neutral_corrected",
        "krispin_normal",
        "simonw_smile_corrected",
        "krispin_smile",
        "stewart neutral smoothed",
        "lea_neutral",
        "stewart smile smoothed",
        "lea_smile",
        "till_neutral",
        "livio-neutral",
        "till_smile",
        "livio-smile",
        "ulla neutral smoothed",
        "mark neutral snoothed",
        "ulla smile smoothed",
        "mark smile smoothed",
        "ulla wacky smoothed",
        "mark wacky smoothed",
        "virginia-brille",
        "markus_normal",
        "virginia-neutral",
        "markus_smile",
        "virginia-smile",
        "michael_normal",
        "zsombor-brille",
        "michael_smile",
        "zsombor-neutral",
        "michi-brille",
        "zsombor-smile",
        "michi-neutral"
};

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

    if(key == '1') {
        // is_selection_enabled = !is_selection_enabled;
    }
    return true;
}

bool load_mesh(string filename) {
    igl::read_triangle_mesh(filename, V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core.align_camera_center(V);
    cout << "Loaded file: " << filename << endl;
    return true;
}

int main(int argc, char *argv[]) {
    string file_path = folder_path + filename + file_extension;
    load_mesh(file_path);

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

            if (ImGui::Button("Delete All Landmarks", ImVec2(-1, 0))) {
                landmarkSelector.delete_all_landmarks();
                cout << "Delete All Landmarks" << endl;
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