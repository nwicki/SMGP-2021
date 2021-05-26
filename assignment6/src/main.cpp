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
Eigen::MatrixXd V;
//face array, #F x3
Eigen::MatrixXi F;

// PCA environment
// Includes
#include <string>
#include <dirent.h>
#include <set>

// Variables
// List of faces already preprocessed for PCA
set<string> _faceFiles;
vector<MatrixXd> _faceList;
MatrixXd _PCA_A;
MatrixXd _PCA_Covariance;
MatrixXd _meanFace;
MatrixXd _eigenFaces;
int _nEigenFaces = 7;
string _currentData;
int _showFaceIndex;

// Constant variables
const string _dataExample1 = "../data/aligned_faces_example/example1/";
const string _dataExample2 = "../data/aligned_faces_example/example2/";
const string _dataExample3 = "../data/aligned_faces_example/example3/";
const string _fileEigenFaces = "eigenfaces.txt";

// Check variables

// Functions
void loadFaces();
void storeEigenFaces();
void loadEigenFaces();

bool endsWith(const string& str, const string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

void loadFaces() {
    DIR *directory;
    struct dirent *entry;
    if ((directory = opendir(_currentData.c_str())) != NULL) {
        // Get all file paths and store in _faceFiles
        while ((entry = readdir(directory)) != NULL) {
            string current = entry->d_name;
            if(endsWith(current, ".obj")) {
                _faceFiles.insert(current);
            }
        }
        closedir (directory);
    }
    else {
        cerr << "Failed to load faces: " << _currentData << endl;
        return;
    }

    // Store faces in a list
    MatrixXd vertices;
    MatrixXi faces;
    for(auto it = _faceFiles.begin(); it != _faceFiles.end(); it++){
        string file = _dataExample1 + *it;
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _faceList.push_back(vertices);
    }
    F = faces;
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
    int rows = _faceList[0].rows();
    // Compute selfadjoint covariance matrix for more stable eigen decomposition:
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    _PCA_Covariance = _PCA_A.adjoint() * _PCA_A * (1 / _PCA_A.rows());
    // Compute selfadjoint eigendecomposition
    SelfAdjointEigenSolver<MatrixXd> eigenDecomposition(_PCA_Covariance);
    // Save transformations for user interaction
    _eigenFaces = eigenDecomposition.eigenvectors();
    // Save computed eigen faces for later usage
    storeEigenFaces();
}

void storeEigenFaces() {
    string filePath = _currentData + _fileEigenFaces;
    cout << "Storing eigen faces in: " << filePath << endl;
    ofstream file(_currentData + _fileEigenFaces);
    for(int i = 0; i < _eigenFaces.rows(); i++) {
        file << _eigenFaces.row(i) << endl;
    }
    file.close();
    cout << endl;
}

void loadEigenFaces() {
    string filePath = _currentData + _fileEigenFaces;
    cout << "Loading eigen faces from: " << filePath << endl;
    ifstream file(filePath);
    vector<vector<double>> eigenFaces;
    double a;
    string row;
    while(getline(file, row)) {
        istringstream rowStream(row);
        eigenFaces.push_back(vector<double>());
        while(rowStream >> a) {
            eigenFaces.back().push_back(a);
        }
    }
    _eigenFaces.resize(eigenFaces.size(), eigenFaces[0].size());
    for(int i = 0; i < _eigenFaces.rows(); i++) {
        _eigenFaces.row(i) = RowVectorXd::Map(eigenFaces[i].data(), eigenFaces[i].size());
    }
    cout << endl;
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

    _currentData = _dataExample1;

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
            if (ImGui::Button("Load faces", ImVec2(-1,0))) {
                loadFaces();
            }
            if (ImGui::Button("Compute PCA (release mode only)", ImVec2(-1,0))) {
                computePCA();
            }

            if (ImGui::Button("Load Eigen faces", ImVec2(-1,0))) {
                loadEigenFaces();
            }

            if (ImGui::Button("Show average face", ImVec2(-1,0))) {
                if(_meanFace.rows() == 0) {
                    if(_faceList.size() == 0) {
                        cout << "No faces loaded" << endl;
                        return;
                    }
                    else {
                        MatrixXd sum = _faceList[0];
                        for(int i = 1; i < _faceList.size(); i++) {
                            sum += _faceList[i];
                        }
                        _meanFace = sum / _faceList.size();
                    }
                }
                viewer.data().clear();
                viewer.data().set_mesh(_meanFace, F);
            }

            if (ImGui::InputInt("Show face", &_showFaceIndex)) {
                if(_faceList.size() == 0) {
                    cout << "No faces loaded" << endl;
                    return;
                }
                if(_showFaceIndex < 0 || _faceList.size() <= _showFaceIndex) {
                    cout << "Face index out of bound" << endl;
                    return;
                }
                viewer.data().clear();
                viewer.data().set_mesh(_faceList[_showFaceIndex], F);
            }
        }
    };

  viewer.callback_key_down = callback_key_down;


  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.launch();

  return 0;

}