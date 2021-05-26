#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <vector>
#include <igl/unproject_onto_mesh.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//vertex array, #V x3
Eigen::MatrixXd V(0,3);
//face array, #F x3
Eigen::MatrixXi F(0,3);

// PCA environment
// Includes
#include <cstring>
#include <dirent.h>

// Variables
// List of faces already preprocessed for PCA
vector<string> _faceFiles;
vector<MatrixXd> _faceList;
MatrixXd _PCA_A;
MatrixXd _PCA_Covariance;
MatrixXd _eigenFaces;
int _nEigenFaces = 7;

// Constant variables
const string _dataExample1 = "../data/aligned_faces_example/example1/";
const string _dataExample2 = "../data/aligned_faces_example/example2/";
const string _dataExample3 = "../data/aligned_faces_example/example3/";

// Check variables

// Functions

void loadFaces(string path) {
    DIR *directory;
    struct dirent *entry;
    if ((directory = opendir(path.c_str())) != NULL) {
        // Get all file paths and store in _faceFiles
        while ((entry = readdir(directory)) != NULL) {
          _faceFiles.push_back(entry->d_name);
        }
        closedir (directory);
    }
    else {
        cerr << "Failed to load faces: " << path << endl;
        return;
    }

    _faceFiles.erase(remove(_faceFiles.begin(), _faceFiles.end(), ".."), _faceFiles.end());
    _faceFiles.erase(remove(_faceFiles.begin(), _faceFiles.end(), "."), _faceFiles.end());

    // Store faces in a list
    MatrixXd vertices, faces;
    for(int i = 0; i < _faceFiles.size(); i++){
        string file = _dataExample1 + _faceFiles[i];
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _faceList.push_back(vertices);
    }
}

void computePCA() {
    // Initialize PCA's A matrix
    int nVertices = _faceList[0].cols() * _faceList[0].rows();

    _PCA_A.resize(nVertices, _faceList.size());

    // Add each face in the list as a vector to the PCA matrix
    for(int i = 0; i < _faceList.size(); i++){
        MatrixXd vertices = _faceList[0];
        // Squish 3D into 1D
        vertices.resize(nVertices,1);
        _PCA_A.col(i) = vertices;
    }

    // Center all vertices in _PCA_A
    _PCA_A = (_PCA_A.colwise() - _PCA_A.rowwise().mean()).transpose();
    // Compute selfadjoint covariance matrix for more stable eigen decomposition:
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    _PCA_Covariance = _PCA_A.adjoint() * _PCA_A * (1 / _PCA_A.rows());
    // Compute selfadjoint eigendecomposition
    SelfAdjointEigenSolver<MatrixXd> eigenDecomposition(_PCA_Covariance);
    // Save transformations for user interaction
    _eigenFaces = eigenDecomposition.eigenvectors();//.block(0,0,_nEigenFaces,_PCA_A.rows());
}

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
    if(argc != 2) {
        load_mesh("../data/aligned_faces_example/example2/alain_normal.obj");
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

        if (ImGui::Button("Save Landmarks", ImVec2(-1,0))) {
            printf("not implemented yet");
        }
        if (ImGui::Button("Load faces")) {
            loadFaces(_dataExample1);
            computePCA();
        }
    }
  };

  viewer.callback_key_down = callback_key_down;
  //viewer.callback_mouse_down = callback_mouse_down;

  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.launch();

  return 0;

}