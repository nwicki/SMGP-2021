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
vector<VectorXd> _faceListDeviation;
vector<MatrixXd> _faceOffsets;
MatrixXd _PCA_A;
MatrixXd _PCA_Covariance;
MatrixXd _meanFace;
MatrixXd _eigenFaces;
int _nEigenFaces = 10;
string _currentData;
int _faceIndex;
vector<float> _weightEigenFaces;
vector<vector<float>> _weightEigenFacesPerFace;

// Constant variables
const string _dataExample1 = "../data/aligned_faces_example/example1/";
const string _dataExample2 = "../data/aligned_faces_example/example2/";
const string _dataExample3 = "../data/aligned_faces_example/example3/";
const string _fileEigenFaces = "eigenfaces.txt";
const int _maxEigenFaces = 20;

// Check variables

// Functions
void loadFaces();
void storeEigenFaces();
void loadEigenFaces();
void computeMeanFace();
void initializeParameters();
void convert3Dto1D(MatrixXd& m);
void computeDeviation();

void convert3Dto1D(MatrixXd& m) {
    if(m.cols() != 3) {
        cout << "Cannot convert non 3-dimensional matrices" << endl;
        exit(1);
    }
    MatrixXd test = m;
    m.resize(m.cols() * m.rows(), 1);
    for(int i = 0; i < test.cols(); i++) {
        if(!(test.col(i) - m.block(i * test.rows(),0,test.rows(),1)).isZero(1e-6)) {
            cout << "3D to 1D conversion failed" << endl;
            exit(1);
        }
    }
}

void convert1Dto3D(MatrixXd& m) {
    if(m.cols() != 1) {
        cout << "Cannot convert non 1-dimensional matrices" << endl;
        exit(1);
    }
    MatrixXd test = m;
    m.resize(m.rows() / 3, 3);
    for(int i = 0; i < m.cols(); i++) {
        if(!(m.col(i) - test.block(i * m.rows(),0,m.rows(),1)).isZero(1e-10)) {
            cout << "1D to 3D conversion failed" << endl;
            cout << "m.rows(): " << m.rows() << endl;
            cout << "test.block(i * m.rows(),0,m.rows(),1).rows(): " << test.block(i * m.rows(),0,m.rows(),1).rows() << endl;
            cout << "Column " << i << " of m: " << m.col(i).transpose() << endl;
            cout << "Segment " << i << " of test: " << test.block(i * m.rows(),0,m.rows(),1).col(0).transpose() << endl;
            exit(1);
        }
    }
}

void computeDeviation() {
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute deviation to mean face for each face" << endl;
    computeMeanFace();
    _faceListDeviation = vector<VectorXd>(_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        MatrixXd diff = _faceList[i] - _meanFace;
        convert3Dto1D(diff);
        _faceListDeviation[i] = diff;
    }
    cout << endl;
}

void initializeParameters() {
    _currentData = _dataExample1;
    _weightEigenFaces = vector<float>(_nEigenFaces);
}

bool endsWith(const string& str, const string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

void loadFaces() {
    cout << "Load faces from " << _currentData << endl;
    _faceFiles = set<string>();
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
    _faceList = vector<MatrixXd>(_faceFiles.size());
    int i = 0;
    for(auto it = _faceFiles.begin(); it != _faceFiles.end(); it++){
        string file = _dataExample1 + *it;
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _faceList[i++] = vertices;
    }
    F = faces;
    cout << endl;
}

void computePCA() {
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute PCA of faces in " << _currentData << endl;
    // Measure runtime
    auto start = chrono::high_resolution_clock::now();
    // Initialize PCA's A matrix
    int nVertices = _faceList[0].cols() * _faceList[0].rows();

    _PCA_A.resize(nVertices, _faceList.size());
    computeMeanFace();
    computeDeviation();
    // Add each face in the list as a vector to the PCA matrix
    for(int i = 0; i < _faceList.size(); i++){
        _PCA_A.col(i) = _faceListDeviation[i];
    }

    // Compute selfadjoint covariance matrix for more stable eigen decomposition:
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    _PCA_Covariance = _PCA_A.adjoint() * _PCA_A * (1.0 / _PCA_A.cols());
    // Compute selfadjoint eigendecomposition
    SelfAdjointEigenSolver<MatrixXd> eigenDecomposition(_PCA_Covariance);
    // Get Eigen vectors
    MatrixXd eigenVectors = eigenDecomposition.eigenvectors();
    // Compute dominant Eigen faces using the approach described in https://www.face-rec.org/algorithms/PCA/jcn.pdf - page 5
    _eigenFaces.resize(_PCA_A.rows(),eigenVectors.cols());
    for(int i = 0; i < eigenVectors.cols(); i++) {
        for(int j = 0; j < eigenVectors.rows(); j++) {
            _eigenFaces.col(i) += eigenVectors.col(i)(j) * _faceListDeviation[j];
        }
    }
    cout << _eigenFaces.col(0).transpose() << endl;
    // Result runtime
    auto end = chrono::high_resolution_clock::now();
    cout << "PCA execution time: " << chrono::duration_cast<chrono::milliseconds> (end-start).count() << " ms" << endl;
    // Save computed eigen faces for later usage
    storeEigenFaces();
    cout << endl;
}

void storeEigenFaces() {
    string filePath = _currentData + _fileEigenFaces;
    cout << "Storing " << _maxEigenFaces << " eigen faces in: " << filePath << endl;
    ofstream file(_currentData + _fileEigenFaces);
    for(int i = 0; i < _maxEigenFaces; i++) {
        file << _eigenFaces.col(i).transpose() << endl;
    }
    file.close();
    cout << endl;
}

void loadEigenFaces() {
    string filePath = _currentData + _fileEigenFaces;
    cout << "Loading " << _maxEigenFaces << " eigen faces from: " << filePath << endl;
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
    _eigenFaces.resize(eigenFaces[0].size(), eigenFaces.size());
    for(int i = 0; i < _eigenFaces.cols(); i++) {
        _eigenFaces.col(i) = VectorXd::Map(eigenFaces[i].data(), eigenFaces[i].size());
    }
    cout << endl;
}

void computeMeanFace() {
    MatrixXd sum = _faceList[0];
    for(int i = 1; i < _faceList.size(); i++) {
        sum += _faceList[i];
    }
    _meanFace = sum / _faceList.size();
}

void computeEigenFaceWeights() {
    if(_eigenFaces.size() < _nEigenFaces) {
        cout << "Not enough eigen faces available" << endl;
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    computeMeanFace();
    cout << "Compute weights for each face and corresponding Eigenfaces" << endl;
    _weightEigenFacesPerFace = vector<vector<float>>(_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        _weightEigenFacesPerFace[i] = vector<float>(_nEigenFaces);
        for(int j = 0; j < _nEigenFaces; j++) {
            _weightEigenFacesPerFace[i][j] = _faceListDeviation[i].dot(_eigenFaces.col(j));
            cout << "Weight " << j << " of face " << i << ": " << _weightEigenFacesPerFace[i][j] << endl;
            cout << "Deviation " << i << ": " << _faceListDeviation[i].transpose() << endl;
            cout << "Eigen face " << j << ": " << _eigenFaces.col(j).transpose() << endl;
        }
    }
    cout << endl;
}

void computeEigenFaceOffsets() {
    if(_weightEigenFacesPerFace.size() < _faceList.size()) {
        cout << "Not enough weights available" << endl;
    }
    cout << "Compute eigen face offsets" << endl;
    _faceOffsets = vector<MatrixXd>(_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
         MatrixXd sum(_faceList[i].rows(),_faceList[i].cols());
        for(int j = 0; j < _nEigenFaces; j++) {
            MatrixXd weighted = _weightEigenFacesPerFace[i][j] * _eigenFaces.col(j);
            cout << "Computation for Eigen face offset: " << i << endl;
            cout << "Weight: " << _weightEigenFacesPerFace[i][j] << endl;
            cout << "Eigen face: " << _eigenFaces.col(j).transpose() << endl;
            convert1Dto3D(weighted);
            sum += weighted;
        }
        _faceOffsets[i] = sum / (double) _nEigenFaces;
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

    initializeParameters();

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

    menu.callback_draw_custom_window = [&]() {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("PCA Menu", NULL, ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::Button("Load faces", ImVec2(-1,0))) {
            loadFaces();
        }
        if (ImGui::Button("Compute PCA (release mode only)", ImVec2(-1,0))) {
            computePCA();
        }

        if (ImGui::Button("Load Eigen faces", ImVec2(-1,0))) {
            loadEigenFaces();
        }

        ImGui::InputInt("Face index", &_faceIndex);

        if (ImGui::Button("Show face", ImVec2(-1,0))) {
            if(_faceList.empty()) {
                cout << "No faces loaded" << endl;
            }
            else {
                if(_faceIndex < 0 || _faceList.size() <= _faceIndex) {
                    cout << "Face index out of bound" << endl;
                }
                else {
                    viewer.data().clear();
                    viewer.data().set_mesh(_faceList[_faceIndex], F);
                }
            }
        }

        if (ImGui::Button("Show average face", ImVec2(-1,0))) {
            if(_meanFace.rows() == 0) {
                if(_faceList.empty()) {
                    cout << "No faces loaded" << endl;
                }
                else {
                    computeMeanFace();
                    viewer.data().clear();
                    viewer.data().set_mesh(_meanFace, F);
                }
            }
            else {
                viewer.data().clear();
                viewer.data().set_mesh(_meanFace, F);
            }
        }

        if (ImGui::Button("Compute face weights", ImVec2(-1,0))) {
            computeEigenFaceWeights();
        }

        if (ImGui::Button("Set face weights", ImVec2(-1,0))) {
            if(_weightEigenFacesPerFace.empty()) {
                cout << "No weights computed" << endl;
            }
            else {
                _weightEigenFaces = _weightEigenFacesPerFace[_faceIndex];
            }
        }

        for(int i = 0; i < _nEigenFaces; i++) {
            ImGui::SliderFloat(("Eigen face " + to_string(i)).c_str(), &_weightEigenFaces[i],-20.0,20.0);
        }

        if (ImGui::Button("Compute Eigen face offset", ImVec2(-1,0))) {
            computeEigenFaceOffsets();
        }

        if (ImGui::Button("Show face with Eigen face offsets", ImVec2(-1,0))) {
            if(_faceOffsets.empty()) {
                cout << "No faces with Eigen offsets computed" << endl;
            }
            else {
                viewer.data().clear();
                viewer.data().set_mesh(_meanFace + _faceOffsets[_faceIndex], F);
            }
        }
        ImGui::End();
    };

    viewer.callback_key_down = callback_key_down;
    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();

    return 0;

}