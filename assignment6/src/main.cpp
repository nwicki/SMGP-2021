#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>
#include "PCA.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0,0);
//face array, #F x3
Eigen::MatrixXi F(0,0);

int pickVertex(int mouse_x, int mouse_y) {
    int vi = -1;

    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view /* viewer.data().model*/,
                                viewer.core.proj, viewer.core.viewport, V, F, fid, bc)) {
        // paint hit red
        bc.maxCoeff(&vi);
        vi = F(fid,vi);
    }
    return vi;

}

bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int) Viewer::MouseButton::Right)
        return false;

    int vi = pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);
    cout << "Found vertex with index: " << vi << endl;

    return true;
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        printf("You pressed 1");
    }

    return true;
}

bool load_mesh(string filename) {
    igl::read_triangle_mesh(filename,V,F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core.align_camera_center(V);
    return true;
}

int main(int argc, char *argv[]) {
    // Menu initialization
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Instructions", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Button("Save Landmarks", ImVec2(-1,0))) {
                printf("not implemented yet");
            }
        }
    };

    #pragma region PCA
    PCA *pca = new PCA();
    menu.callback_draw_custom_window = [&]() {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("PCA Menu", NULL, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Data set: %s",pca->_currentData.c_str());
        if (ImGui::Button("Load faces", ImVec2(-1,0))) {
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

        if(ImGui::InputInt("Morph face index", &pca->_morphIndex)) {
            pca->_morphIndex = min(max(0, pca->_morphIndex), (int) (pca->_faceList.size() - 1));
            pca->showMorphedFace(viewer, F);
        }

        ImGui::Text("Morphing of the following indices: \nFace index: %d \nMorph face index: %d", pca->_faceIndex, pca->_morphIndex);

        if(ImGui::SliderFloat("Morphing Variable", &pca->_morphLambda,0,1)) {
            pca->showMorphedFace(viewer, F);
        }

        if (ImGui::Button("Show morphed face", ImVec2(-1,0))) {
            pca->showMorphedFace(viewer, F);
        }
        ImGui::End();
    };
    #pragma endregion

    // Viewer initialization
    viewer.callback_key_down = callback_key_down;
    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.data().show_lines = false;
    // Moves face to look towards camera
    igl::trackball(viewer.core.viewport(2),viewer.core.viewport(3),2.0f,viewer.down_rotation,0.0,0.0,0.0,2.5,viewer.core.trackball_angle);
    viewer.snap_to_canonical_quaternion();
    viewer.launch();

    return 0;

}