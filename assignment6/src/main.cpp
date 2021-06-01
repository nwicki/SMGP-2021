#include <boost/filesystem.hpp>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include <string>
#include <iostream>

#include "LandmarkSelector.h"
#include "FaceRegistor.h"
#include "PCA.h"

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
int prev_mode = 0;

// Landmark Selection
bool is_selection_enabled = false;
LandmarkSelector landmarkSelector = LandmarkSelector();

// Mesh Loading
vector<string> landmark_folder_names = {"../data/face_template/", "../data/scanned_faces_cleaned/"};
vector<vector<string>> landmark_filenames = vector<vector<string>>(2, vector<string>());
string landmark_folder_path = "../data/face_template/";
string landmark_filename = "headtemplate";
string landmark_file_extension = ".obj";
static int selected_landmark_folder_id = 0;
static int selected_landmark_file_id = 0;

// Face registration
Eigen::MatrixXd V_tmpl(0, 3);
Eigen::MatrixXi F_tmpl(0, 3);
bool hide_scan_face = false;
bool has_subdivided = false;
FaceRegistor faceRegistor = FaceRegistor(&landmarkSelector);

// PCA computation
PCA *pca = new PCA();

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
    if(current_mode == 2) { // face registration
        if(key == '1') {
            viewer.selected_data_index = 0;
        }
        if(key == '2') {
            viewer.selected_data_index = 1;
        }
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

    ImGui::PushItemWidth(0.9*menu_width);
    ImGui::Text("Choose folder");
    if (ImGui::BeginCombo("", &landmark_folder_names[selected_landmark_folder_id][0]))
    {
        for (int n = 0; n < landmark_folder_names.size(); n++)
        {
            bool is_selected = (selected_landmark_folder_id == n);
            if (ImGui::Selectable(&landmark_folder_names[n][0], is_selected)) {
                selected_landmark_folder_id = n;
                landmark_folder_path = landmark_folder_names[selected_landmark_folder_id];
            }

            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::BeginChild("Faces", ImVec2(180, 400), true);
    for (int i = 0; i < landmark_filenames[selected_landmark_folder_id].size(); i++) {
        if (ImGui::Selectable(&landmark_filenames[selected_landmark_folder_id][i][0], selected_landmark_file_id == i)) {
            selected_landmark_file_id = i;
            landmark_filename = landmark_filenames[selected_landmark_folder_id][selected_landmark_file_id];
            string face_file_path = landmark_folder_path + landmark_filename + landmark_file_extension;
            load_mesh(face_file_path, V, F, 0);
            landmarkSelector.current_landmarks.clear();
        }
    }
    ImGui::EndChild();
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
        faceRegistor.center_and_rescale_scan(V, F);
        set_mesh(V, F, 1);
        cout << "Center & Scale face" << endl;
    }

    if (ImGui::Button("Center & Scale template", ImVec2(-1, 0))) {
        faceRegistor.center_and_rescale_template(V_tmpl, F_tmpl, V, F);
        set_mesh(V_tmpl, F_tmpl, 0);
        cout << "Center & Scale template" << endl;
    }

    if (ImGui::Button("Align Rigid", ImVec2(-1, 0))) {
        faceRegistor.align_rigid(V_tmpl, F_tmpl, V, F);
        set_mesh(V, F, 1);
        cout << "Align Rigid" << endl;
    }
    if (ImGui::Button("Align Non-Rigid", ImVec2(-1, 0))) {
        faceRegistor.build_octree(V);
        faceRegistor.align_non_rigid_step(V_tmpl, F_tmpl, V, F);
        set_mesh(V_tmpl, F_tmpl, 0);
        cout << "Align Non-Rigid" << endl;
    }
    if (ImGui::Button("Register", ImVec2(-1, 0))) {
        faceRegistor.register_face(V_tmpl, F_tmpl, V, F);
        set_mesh(V_tmpl, F_tmpl, 0);
        set_mesh(V, F, 1);
        cout << "Register face" << endl;
    }
    if (ImGui::Button("Subdivide template", ImVec2(-1, 0))) {
        faceRegistor.subdivide_template(V_tmpl, F_tmpl);
        set_mesh(V_tmpl, F_tmpl, 0);
        has_subdivided = true;
        cout << "Subdivide template" << endl;
    }
    if (ImGui::Button("Save registered face", ImVec2(-1, 0))) {
        string save_path = faceRegistor.save_registered_scan(V_tmpl, F_tmpl);
        cout << "Saved registered face to " << save_path << endl;
    }
    ImGui::PushItemWidth(0.4*menu_width);
    if (ImGui::InputFloat("lambda", &faceRegistor.m_lambda))
    {
        faceRegistor.m_lambda = std::max(0.0f, std::min(1000.0f, faceRegistor.m_lambda));
    }

    if (ImGui::InputFloat("epsilon", &faceRegistor.m_epsilon))
    {
        faceRegistor.m_epsilon = std::max(0.0f, std::min(1000.0f, faceRegistor.m_epsilon));
    }
    ImGui::PopItemWidth();

    if (ImGui::Checkbox("Hide scan mesh", &hide_scan_face)) {
        viewer.data_list[1].show_faces = !hide_scan_face;
        viewer.data_list[1].show_lines = !hide_scan_face;
    }

    ImGui::PushItemWidth(0.9*menu_width);
    ImGui::Text("Template Face");
    if (ImGui::BeginCombo("", &faceRegistor.tmpl_names[faceRegistor.tmpl_id][0])) // The second parameter is the label previewed before opening the combo. 
    {
        for (int n = 0; n < faceRegistor.tmpl_names.size(); n++)
        {
            bool is_selected = (faceRegistor.tmpl_id == n); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(&faceRegistor.tmpl_names[n][0], is_selected)) {
                faceRegistor.tmpl_id = n;
                string tmpl_file_path = faceRegistor.tmpl_folder_path + faceRegistor.tmpl_names[faceRegistor.tmpl_id]+".obj";
                load_mesh(tmpl_file_path, V_tmpl, F_tmpl, 0);
                has_subdivided = false;
            }

            if (is_selected)
                ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::Text("Face to register");
    ImGui::BeginChild("Faces", ImVec2(180, 300), true);
    for (int i = 0; i < faceRegistor.scan_names.size(); i++) {
        if (ImGui::Selectable(&faceRegistor.scan_names[i][0], faceRegistor.scan_id == i)) {
            faceRegistor.scan_id = i;
            string scan_file_path = faceRegistor.scan_folder_path + faceRegistor.scan_names[faceRegistor.scan_id]+".obj";
            load_mesh(scan_file_path, V, F, 1);
        }
    }
    ImGui::EndChild();

    if (ImGui::Button("Register all", ImVec2(-1, 0))) {
        int prev_id = faceRegistor.scan_id;
        for (int i = 0; i < faceRegistor.scan_names.size(); i++) {
            faceRegistor.scan_id = i;
            // load a scanned face
            string scan_file_path = faceRegistor.scan_folder_path + faceRegistor.scan_names[i]+".obj";
            load_mesh(scan_file_path, V, F, 1);
            // reload template face
            string tmpl_file_path = faceRegistor.tmpl_folder_path + faceRegistor.tmpl_names[faceRegistor.tmpl_id]+".obj";
            load_mesh(tmpl_file_path, V_tmpl, F_tmpl, 0);
            // register it (same code as register)
            faceRegistor.register_face(V_tmpl, F_tmpl, V, F);
            set_mesh(V_tmpl, F_tmpl, 0);
            set_mesh(V, F, 1);
            // save mesh
            string save_path = faceRegistor.save_registered_scan(V_tmpl, F_tmpl);
            cout << "Saved registered face to " << save_path << endl;
        }
        faceRegistor.scan_id = prev_id;
        cout << "Registered all faces using selected template" << endl;
    }

    ImGui::End();
}

void draw_pca_computation_window(ImGuiMenu &menu) {
    float menu_width = 200.f * menu.menu_scaling();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 20.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
    //ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
    ImGui::Begin(
        "PCA Computation", nullptr,ImGuiWindowFlags_AlwaysAutoResize
    );

    draw_reduced_viewer_menu();

    ImGui::Separator();

    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Combo("",&pca->_currentData,pca->_dataExamples.data(),pca->_dataExamples.size())) {
        pca->loadFaces(viewer, F, false);
    }

    if (ImGui::Button("Show average face", ImVec2(-1,0))) {
        pca->showAverageFace(viewer, F);
    }

    if(ImGui::InputInt("Face index", &pca->_faceIndex)) {
        pca->updateFaceIndex(viewer, F);
    }

    if (ImGui::Button("Show face", ImVec2(-1,0))) {
        pca->showFace(viewer, F);
    }

    ImGui::Separator();

    if(ImGui::InputInt("#Eigen faces", &pca->_nEigenFaces)) {
        pca->_nEigenFaces = min(max(1,pca->_nEigenFaces), pca->_maxEigenFaces);
        pca->initializeParameters();
        pca->recomputeAll();
        pca->updateWeightEigenFaces();
        pca->showEigenFaceOffset(viewer, F);
    }

    for(int i = 0; i < pca->_nEigenFaces; i++) {
        if(ImGui::SliderFloat(("Eigen face " + to_string(i)).c_str(), &pca->_weightEigenFaces(i),0.0,1.0,"%.3f")) {
            pca->computeEigenFaceOffsetIndex();
            viewer.data().clear();
            viewer.data().set_mesh(pca->_meanFace + pca->_faceOffset, F);
        }
    }

    if (ImGui::Button("Show face with Eigen face offsets", ImVec2(-1,0))) {
        pca->showEigenFaceOffset(viewer, F);
    }

    if (ImGui::Button("Show error", ImVec2(-1,0))) {
        pca->showEigenFaceOffset(viewer, F);
        pca->showError(viewer, F);
    }

    if(ImGui::InputInt("Morph face index", &pca->_morphIndex)) {
        pca->_morphIndex = min(max(0, pca->_morphIndex), (int) (pca->_faceList.size() - 1));
        pca->showMorphedFace(viewer, F);
    }

    ImGui::Text("Morphing of the following indices: \nFace index: %d \nMorph face index: %d", pca->_faceIndex, pca->_morphIndex);

    if(ImGui::SliderFloat("Morph rate", &pca->_morphLambda,0,1)) {
        pca->showMorphedFace(viewer, F);
    }

    if (ImGui::Button("Show morphed face", ImVec2(-1,0))) {
        pca->showMorphedFace(viewer, F);
    }

    ImGui::End();
}

void clear_all_data(){
    for(size_t i=0; i<viewer.data_list.size(); i++){
        viewer.selected_data_index = i;
        viewer.data().clear();
    } 
}

void setup_gui(ImGuiMenu &menu) {
    menu.callback_draw_viewer_window = [&](){};
    menu.callback_draw_custom_window = [&]()
    {
        // Draw menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Viewer")) {
                    prev_mode = current_mode;
                    current_mode = 0;
                }
                if (ImGui::MenuItem("Landmark Selection")) {
                    if(current_mode != 0 || prev_mode != 1){
                    clear_all_data();
                    string landmark_file_path = landmark_folder_path + landmark_filename + landmark_file_extension;
                    load_mesh(landmark_file_path);
                }
                    prev_mode = current_mode;
                    current_mode = 1;
                }
                if (ImGui::MenuItem("Face Registration")) {
                    if(current_mode != 0 || prev_mode != 2){
                        clear_all_data();
                        string tmpl_file_path = faceRegistor.tmpl_folder_path + faceRegistor.tmpl_names[faceRegistor.tmpl_id]+".obj";
                        load_mesh(tmpl_file_path, V_tmpl, F_tmpl, 0);
                    }
                    prev_mode = current_mode;
                    current_mode = 2;
                }
                if (ImGui::MenuItem("PCA Computation")) {
                    if(current_mode != 0 || prev_mode != 3){
                    clear_all_data();
                }
                    prev_mode = current_mode;
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
    sort(names.begin(), names.end());
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
    fill_filenames(landmark_filenames[0], "../data/face_template/", ".obj");
    fill_filenames(landmark_filenames[1], "../data/scanned_faces_cleaned/", ".obj");

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;

    viewer.data().point_size = 15;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    // Moves face to look towards camera
    igl::trackball(viewer.core.viewport(2),viewer.core.viewport(3),2.0f,viewer.down_rotation,0.0,0.0,0.0,2.5,viewer.core.trackball_angle);
    viewer.snap_to_canonical_quaternion();
    viewer.launch();

    return 0;

}
