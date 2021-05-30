#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include <string>
#include "LandmarkSelector.h"
#include "FaceRegistor.h"
#include <boost/filesystem.hpp>
#include <iostream>


using namespace std;
using namespace Eigen;
namespace fs = boost::filesystem;
using Viewer = igl::opengl::glfw::Viewer;
using ImGuiMenu = igl::opengl::glfw::imgui::ImGuiMenu;

Viewer viewer;

// Vertex array, #V x3
Eigen::MatrixXd V(0, 3);
// Face array, #F x3
Eigen::MatrixXi F(0, 3);

// Menu Mode
int current_mode = 0;

// Landmark Selection
bool is_selection_enabled = false;
LandmarkSelector landmarkSelector = LandmarkSelector();
FaceRegistor faceRegistor = FaceRegistor();

// Mesh Loading
string landmark_folder_path = "../data/face_template/";
string landmark_filename = "headtemplate";
string landmark_file_extension = ".obj";

// Face registration
Eigen::MatrixXd V_tmpl(0, 3);
Eigen::MatrixXi F_tmpl(0, 3);

vector<string> landmarked_face_names;
static int selected_face_id = 0;
string face_folder_path = "../data/scanned_faces_cleaned/";

vector<string> face_template_names;
static int selected_template_id = 2;
string tmpl_folder_path = "../data/face_template/";

bool hide_scan_face = false;
float non_rigid_lambda = 1.0f;

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

bool set_mesh(const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F, int id = 0) {
    if (id > viewer.data_list.size())
        return false;
    if (id == viewer.data_list.size()) {
        viewer.append_mesh();
        viewer.data().clear();
        viewer.data().set_mesh(_V, _F);
    } else {
        viewer.selected_data_index = id;
        viewer.data().clear();
        viewer.data().set_mesh(_V, _F);
    }

    viewer.core.align_camera_center(_V);
    return true;
}

bool load_mesh(string filename, Eigen::MatrixXd& _V = V, Eigen::MatrixXi& _F = F, int id = 0) {
    igl::read_triangle_mesh(filename, _V, _F);
    set_mesh(_V, _F, id);
    cout << "Loaded file: " << filename << endl;
    return true;
}

void draw_full_viewer_window(ImGuiMenu &menu) {
    float menu_width = 200.f * menu.menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 20.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
    ImGui::Begin(
        "Viewer", nullptr,
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
    );
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);

    // Viewing options
    if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Center object", ImVec2(-1, 0)))
        {
        viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
        }
        if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
        {
        viewer.snap_to_canonical_quaternion();
        }

        // Zoom
        ImGui::PushItemWidth(80 * menu.menu_scaling());
        ImGui::DragFloat("Zoom", &(viewer.core.camera_zoom), 0.05f, 0.1f, 20.0f);

        // Select rotation type
        int rotation_type = static_cast<int>(viewer.core.rotation_type);
        static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
        static bool orthographic = true;
        if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\0002D Mode\0\0"))
        {
        using RT = igl::opengl::ViewerCore::RotationType;
        auto new_type = static_cast<RT>(rotation_type);
        if (new_type != viewer.core.rotation_type)
        {
            if (new_type == RT::ROTATION_TYPE_NO_ROTATION)
            {
            trackball_angle = viewer.core.trackball_angle;
            orthographic = viewer.core.orthographic;
            viewer.core.trackball_angle = Eigen::Quaternionf::Identity();
            viewer.core.orthographic = true;
            }
            else if (viewer.core.rotation_type == RT::ROTATION_TYPE_NO_ROTATION)
            {
            viewer.core.trackball_angle = trackball_angle;
            viewer.core.orthographic = orthographic;
            }
            viewer.core.set_rotation_type(new_type);
        }
        }

        // Orthographic view
        ImGui::Checkbox("Orthographic view", &(viewer.core.orthographic));
        ImGui::PopItemWidth();
    }

    // Draw options
    if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Checkbox("Face-based", &(viewer.data().face_based)))
        {
        viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
        }
        ImGui::Checkbox("Show texture", &(viewer.data().show_texture));
        if (ImGui::Checkbox("Invert normals", &(viewer.data().invert_normals)))
        {
        viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
        }
        ImGui::Checkbox("Show overlay", &(viewer.data().show_overlay));
        ImGui::Checkbox("Show overlay depth", &(viewer.data().show_overlay_depth));
        ImGui::ColorEdit4("Background", viewer.core.background_color.data(),
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
        ImGui::ColorEdit4("Line color", viewer.data().line_color.data(),
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
        ImGui::DragFloat("Shininess", &(viewer.data().shininess), 0.05f, 0.0f, 100.0f);
        ImGui::PopItemWidth();
    }

    // Overlays
    if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Wireframe", &(viewer.data().show_lines));
        ImGui::Checkbox("Fill", &(viewer.data().show_faces));
        ImGui::Checkbox("Show vertex labels", &(viewer.data().show_vertid));
        ImGui::Checkbox("Show faces labels", &(viewer.data().show_faceid));
    }

    ImGui::PopItemWidth();
    ImGui::End();
}

void draw_reduced_viewer_menu() {
    // Draw options
    if (ImGui::CollapsingHeader("Viewer"))
    {
        if (ImGui::Checkbox("Face-based", &(viewer.data().face_based)))
        {
        viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
        }
        if (ImGui::Checkbox("Invert normals", &(viewer.data().invert_normals)))
        {
        viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
        }
        ImGui::Checkbox("Show overlay", &(viewer.data().show_overlay));
        ImGui::Checkbox("Show overlay depth", &(viewer.data().show_overlay_depth));
        ImGui::Checkbox("Wireframe", &(viewer.data().show_lines));
    }
}

void draw_landmark_selection_window(ImGuiMenu &menu) {
    float menu_width = 200.f * menu.menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 20.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
    ImGui::Begin(
        "Landmark Selection", nullptr,
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
    );

    draw_reduced_viewer_menu();
    ImGui::Separator();

    if (ImGui::Checkbox("Enable Selection", &is_selection_enabled)) {
        string message = (is_selection_enabled) ? "Selection Enabled" : "Selection Disabled";
        cout << message << endl;
    }

    if (ImGui::Button("Save Landmarks", ImVec2(-1, 0))) {
        string file_path = landmark_folder_path + landmark_filename + "_landmarks.txt";
        landmarkSelector.save_landmarks_to_file(landmarkSelector.current_landmarks, file_path);
        cout << landmarkSelector.current_landmarks.size() << " landmarks saved to " << file_path << endl;
    }

    if (ImGui::Button("Delete last landmark", ImVec2(-1, 0))) {
        cout << "Delete last added landmark" << endl;
        landmarkSelector.delete_last_landmark();
        landmarkSelector.display_landmarks(landmarkSelector.current_landmarks, V, F, viewer);
    }

    if (ImGui::Button("Load Landmarks", ImVec2(-1, 0))) {
        string file_path = landmark_folder_path + landmark_filename + "_landmarks.txt";
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

    ImGui::End();
}

void draw_face_registration_window(ImGuiMenu &menu) {
    float menu_width = 200.f * menu.menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 20.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
    ImGui::Begin(
        "Face registration", nullptr,
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
    );

    draw_reduced_viewer_menu();

    ImGui::Separator();

    if (ImGui::Button("Center & Scale face", ImVec2(-1, 0))) {
        string face_file_path = face_folder_path + landmarked_face_names[selected_face_id] + "_landmarks.txt";
        MatrixXd P = landmarkSelector.get_landmarks_from_file(face_file_path, V, F);
        faceRegistor.center_and_rescale_mesh(V, P);
        set_mesh(V, F, 1);
        cout << "Center & Scale face" << endl;
    }

    if (ImGui::Button("Center & Scale template", ImVec2(-1, 0))) {
        string tmpl_file_path = tmpl_folder_path + face_template_names[selected_template_id] + "_landmarks.txt";
        string face_file_path = face_folder_path + landmarked_face_names[selected_face_id] + "_landmarks.txt";
        MatrixXd P_tmpl = landmarkSelector.get_landmarks_from_file(tmpl_file_path, V_tmpl, F_tmpl);
        MatrixXd P = landmarkSelector.get_landmarks_from_file(face_file_path, V, F);
        faceRegistor.center_and_rescale_template(V_tmpl, P_tmpl, P);
        set_mesh(V_tmpl, F_tmpl, 0);
        cout << "Center & Scale template" << endl;
    }

    if (ImGui::Button("Align Rigid", ImVec2(-1, 0))) {
        string tmpl_file_path = tmpl_folder_path + face_template_names[selected_template_id] + "_landmarks.txt";
        string face_file_path = face_folder_path + landmarked_face_names[selected_face_id] + "_landmarks.txt";
        MatrixXd P_tmpl = landmarkSelector.get_landmarks_from_file(tmpl_file_path, V_tmpl, F_tmpl);
        MatrixXd P = landmarkSelector.get_landmarks_from_file(face_file_path, V, F);
        faceRegistor.align_rigid(V_tmpl, P_tmpl, P);
        set_mesh(V_tmpl, F_tmpl, 0);
        cout << "Align Rigid" << endl;
    }
    if (ImGui::Button("Align Non-Rigid", ImVec2(-1, 0))) {
        string tmpl_file_path = tmpl_folder_path + face_template_names[selected_template_id] + "_landmarks.txt";
        string face_file_path = face_folder_path + landmarked_face_names[selected_face_id] + "_landmarks.txt";
        vector<LandmarkSelector::Landmark> landmarks_tmpl = landmarkSelector.get_landmarks_from_file(tmpl_file_path);
        MatrixXd P = landmarkSelector.get_landmarks_from_file(face_file_path, V, F);
        faceRegistor.build_octree(V);
        faceRegistor.align_non_rigid_step(V_tmpl, F_tmpl, landmarks_tmpl, V, P, non_rigid_lambda);
        set_mesh(V_tmpl, F_tmpl, 0);
        cout << "Align Non-Rigid" << endl;
    }

    if (ImGui::InputFloat("lambda", &non_rigid_lambda))
    {
        non_rigid_lambda = std::max(0.0f, std::min(1000.0f, non_rigid_lambda));
    }

    if (ImGui::Checkbox("Hide scan mesh", &hide_scan_face)) {
        viewer.data_list[1].show_faces = !hide_scan_face;
        viewer.data_list[1].show_lines = !hide_scan_face;
    }

    ImGui::Text("Template Face");
    if (ImGui::BeginCombo("Template Combo", &face_template_names[selected_template_id][0])) // The second parameter is the label previewed before opening the combo.
    {
        for (int n = 0; n < face_template_names.size(); n++)
        {
            bool is_selected = (selected_template_id == n); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(&face_template_names[n][0], is_selected)) {
                selected_template_id = n;
                string template_file_path = tmpl_folder_path + face_template_names[selected_template_id]+".obj";
                load_mesh(template_file_path, V_tmpl, F_tmpl, 0);
            }

            if (is_selected)
                ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
        }
        ImGui::EndCombo();
    }

    ImGui::Text("Face to register");
    ImGui::BeginChild("Faces", ImVec2(180, 400), true);
    for (int i = 0; i < landmarked_face_names.size(); i++) {
        if (ImGui::Selectable(&landmarked_face_names[i][0], selected_face_id == i)) {
            selected_face_id = i;
            string face_file_path = face_folder_path + landmarked_face_names[selected_face_id]+".obj";
            load_mesh(face_file_path, V, F, 1);
        }
    }
    ImGui::EndChild();

    ImGui::End();
}

void draw_pca_computation_window(ImGuiMenu &menu) {
    float menu_width = 200.f * menu.menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 20.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
    ImGui::Begin(
        "PCA Computation", nullptr,
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
    );

    draw_reduced_viewer_menu();

    ImGui::Separator();

    // add pca computation menu options here

    ImGui::End();
}

void setup_gui(ImGuiMenu &menu) {
    menu.callback_draw_viewer_window = [&](){};
    menu.callback_draw_custom_window = [&]()
    {
        // Draw menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Viewer")) {
                    current_mode = 0;
                }
                if (ImGui::MenuItem("Landmark Selection")) {
                    current_mode = 1;
                    viewer.data().clear();
                    string landmark_file_path = landmark_folder_path + landmark_filename + landmark_file_extension;
                    load_mesh(landmark_file_path);
                }
                if (ImGui::MenuItem("Face Registration")) {
                    current_mode = 2;
                    viewer.data().clear();
                    string template_file_path = tmpl_folder_path + face_template_names[selected_template_id]+".obj";
                    string face_file_path = face_folder_path + landmarked_face_names[selected_face_id]+".obj";
                    load_mesh(template_file_path, V_tmpl, F_tmpl, 0);
                    load_mesh(face_file_path, V, F, 1);
                }
                if (ImGui::MenuItem("PCA Computation")) {
                    current_mode = 3;
                }
                ImGui::EndMenu();
            }
            ImGui::SameLine(ImGui::GetWindowWidth()-70);
            ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
            ImGui::EndMainMenuBar();
        }

        // Show corresponding menu
        switch(current_mode) {
            case 0: draw_full_viewer_window(menu); break;
            case 1: draw_landmark_selection_window(menu); break;
            case 2: draw_face_registration_window(menu); break;
            case 3: draw_pca_computation_window(menu); break;
            default: break;
        }
    };
}

void fill_filenames(vector<string> &names,  string path, string extension) {
    names.clear();
    for (auto const & file : fs::recursive_directory_iterator(path)) {
        if (fs::is_regular_file(file) && file.path().extension() == extension)
            names.emplace_back(file.path().stem().string());
    }
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        string file_path = landmark_folder_path + landmark_filename + landmark_file_extension;
        load_mesh(file_path);
    } else {
        load_mesh(argv[1]);
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    setup_gui(menu);

    // face registration
    fill_filenames(landmarked_face_names, "../data/scanned_faces_cleaned/", ".obj");
    fill_filenames(face_template_names, "../data/face_template/", ".obj");

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;

    viewer.data().point_size = 15;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();

    return 0;

}