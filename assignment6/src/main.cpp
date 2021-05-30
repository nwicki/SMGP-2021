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
Eigen::MatrixXd V(0,0);
//face array, #F x3
Eigen::MatrixXi F(0,0);

// PCA environment
// Includes
#include <string>
#include <dirent.h>
#include <set>
#include <sys/stat.h>

// Variables
// List of faces already preprocessed for PCA
set<string> _faceFiles;
vector<MatrixXd> _faceList;
vector<MatrixXd> _faceListDeviation;

// Offset of all faces
vector<MatrixXd> _faceOffsets;
// Face index offset
MatrixXd _faceOffset;
// cov = A * A^T
MatrixXd _PCA_A(0,0);
// Covariance matrix for Eigen decomposition
MatrixXd _PCA_Covariance(0,0);
// Average face
MatrixXd _meanFace(0,0);
// Eigen faces #3*vertices x #faces
MatrixXd _eigenFaces(0,0);
// Amount of Eigen faces considered
int _nEigenFaces = 10;
// Data folder
string _currentData;
// Face index chosen by user
int _faceIndex;
// Weights of the face displayed
VectorXf _weightEigenFaces(0);
// Columns hold weight per Eigen face #Eigen faces x #faces
MatrixXd _weightEigenFacesPerFace(0,0);
// Face index to morph the current one with
int _morphIndex;
// Morph factor
float _morphLambda;

// Booleans to reduce recomputation
bool _recomputeMeanFace = false;
bool _recomputeDeviation = false;
bool _recomputePCA = false;
bool _recomputeEigenFaceWeights = false;
bool _recomputeEigenFaceOffsets = false;


// Constant variables
const set<string> _dataExamples = {
        "../data/aligned_faces_example/example1/",
        "../data/aligned_faces_example/example2/",
        "../data/aligned_faces_example/example3/"
};
const string _fileEigenFaces = "eigenfaces.txt";
const int _maxEigenFaces = 20;

// Check variables

// Functions
void storeEigenFaces();
void loadEigenFaces();
bool endsWith(const string& str, const string& suffix);
void convert3Dto1D(MatrixXd& m);
void convert1Dto3D(MatrixXd& m);
void initializeParameters();
void loadFaces(Viewer& viewer, bool init);
void computeMeanFace();
void computeDeviation();
void computePCA();
void computeEigenFaceWeights();
void computeEigenFaceOffsets();
void computeEigenFaceOffsetIndex();
void recomputeAll();
void showAverageFace(Viewer& viewer);
void updateWeightEigenFaces();
void updateFaceIndex(Viewer& viewer);
void showFace(Viewer& viewer);
void showEigenFaceOffset(Viewer& viewer);
void showMorphedFace(Viewer& viewer);

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

bool endsWith(const string& str, const string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

void initializeParameters() {
    _weightEigenFaces.resize(_nEigenFaces);
    _weightEigenFaces.setZero();
    _recomputeMeanFace = true;
    _recomputeDeviation = true;
    _recomputePCA = true;
    _recomputeEigenFaceWeights = true;
    _recomputeEigenFaceOffsets = true;
}

void loadFaces(Viewer& viewer, bool init) {
    _weightEigenFaces.resize(_nEigenFaces);
    _weightEigenFaces.setZero();
    if(!init) {
        string file = igl:: file_dialog_open();
        struct stat buffer;
        if(stat (file.c_str(), &buffer) != 0) {
            return;
        }
        _currentData = file.substr(0,file.find_last_of("/") + 1);
    }
    if(_dataExamples.find(_currentData) == _dataExamples.end()) {
        // Get file path into relational format
        _currentData = "../" + _currentData.substr(_currentData.length() - (*_dataExamples.begin()).length() + 3);
        if(_dataExamples.find(_currentData) == _dataExamples.end()) {
            cout << "No viable data set chosen: " << _currentData << endl;
            return;
        }
    }
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
        string file = _currentData + *it;
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _faceList[i++] = vertices;
    }
    F = faces;
    initializeParameters();
    recomputeAll();
    updateFaceIndex(viewer);
    showFace(viewer);
    viewer.snap_to_canonical_quaternion();
    viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
    cout << endl;
}

void computeMeanFace() {
    if(!_recomputeMeanFace) {
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute mean face" << endl;
    MatrixXd sum = _faceList[0];
    for(int i = 1; i < _faceList.size(); i++) {
        sum += _faceList[i];
    }
    _meanFace = sum / _faceList.size();
    _recomputeMeanFace = false;
    cout << endl;
}

void computeDeviation() {
    if(!_recomputeDeviation) {
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute deviation to mean face for each face" << endl;
    _faceListDeviation = vector<MatrixXd>(_faceList.size());
    _PCA_A.resize(_faceList[0].cols() * _faceList[0].rows(), _faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        MatrixXd diff = _faceList[i] - _meanFace;
        _faceListDeviation[i] = diff;
        convert3Dto1D(diff);
        _PCA_A.col(i) = diff;
    }
    _recomputeDeviation = false;
    cout << endl;
}

void computePCA() {
    if(!_recomputePCA) {
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute PCA of faces" << endl;
    // Measure runtime
    auto start = chrono::high_resolution_clock::now();
    // Initialize PCA's A matrix
    int nVertices = _faceList[0].cols() * _faceList[0].rows();

    // Compute selfadjoint covariance matrix for more stable eigen decomposition:
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    _PCA_Covariance = _PCA_A.adjoint() * _PCA_A * (1.0 / _PCA_A.cols());
    // Compute selfadjoint eigendecomposition
    SelfAdjointEigenSolver<MatrixXd> eigenDecomposition(_PCA_Covariance);
    // Get Eigen vectors
    MatrixXd eigenVectors = eigenDecomposition.eigenvectors();
    // Compute dominant Eigen faces using the approach described in https://www.face-rec.org/algorithms/PCA/jcn.pdf - page 5
    MatrixXd decreasing(_PCA_A.rows(),eigenVectors.cols());
    for(int i = 0; i < eigenVectors.cols(); i++) {
        for(int j = 0; j < eigenVectors.rows(); j++) {
            decreasing.col(i) += eigenVectors.col(i)(j) * _PCA_A.col(j);
        }
    }
    decreasing.colwise().normalize();
    // Order Eigen faces in decreasing order
    _eigenFaces.resize(decreasing.rows(),_maxEigenFaces);
    for(int i = 0; i < _maxEigenFaces; i++) {
        _eigenFaces.col(i) = decreasing.col(decreasing.cols() - 1 - i);
    }
    // Result runtime
    auto end = chrono::high_resolution_clock::now();
    cout << "PCA execution time: " << chrono::duration_cast<chrono::milliseconds> (end-start).count() << " ms" << endl;
    // PCA for current faces computed
    _recomputePCA = false;
    cout << endl;
}

void computeEigenFaceWeights() {
    if(!_recomputeEigenFaceWeights) {
        return;
    }
    if(_eigenFaces.size() < _nEigenFaces) {
        cout << "Not enough eigen faces available" << endl;
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute weights for each face and corresponding Eigenfaces" << endl;
    _weightEigenFacesPerFace.resize(_nEigenFaces,_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        for(int j = 0; j < _nEigenFaces; j++) {
            _weightEigenFacesPerFace(j,i) = _PCA_A.col(i).dot(_eigenFaces.col(j));
        }
    }
    // Eigen face weights for current faces computed
    _recomputeEigenFaceWeights = false;
    cout << endl;
}

void computeEigenFaceOffsets() {
    if(!_recomputeEigenFaceOffsets) {
        return;
    }
    if(_weightEigenFacesPerFace.size() < _faceList.size()) {
        cout << "Not enough weights available" << endl;
        return;
    }
    cout << "Compute eigen face offsets" << endl;
    _faceOffsets = vector<MatrixXd>(_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        MatrixXd sum(_faceList[i].rows(),_faceList[i].cols());
        sum.setZero();
        for(int j = 0; j < _nEigenFaces; j++) {
            MatrixXd weighted = _weightEigenFacesPerFace(j,i) * _eigenFaces.col(j);
            convert1Dto3D(weighted);
            sum += weighted;
        }
        _faceOffsets[i] = sum;
    }
    _recomputeEigenFaceOffsets = false;
    cout << endl;
}

void computeEigenFaceOffsetIndex() {
    if(_weightEigenFacesPerFace.size() < _faceList.size()) {
        cout << "Not enough weights available" << endl;
        return;
    }
    MatrixXd sum(_faceList[0].rows(),_faceList[0].cols());
    sum.setZero();
    for(int j = 0; j < _nEigenFaces; j++) {
        MatrixXd weighted = _weightEigenFaces(j) * _eigenFaces.col(j);
        convert1Dto3D(weighted);
        sum += weighted;
    }
    _faceOffset = sum;
}

void recomputeAll() {
    computeMeanFace();
    computeDeviation();
    computePCA();
    computeEigenFaceWeights();
    computeEigenFaceOffsets();
    computeEigenFaceOffsetIndex();
}

void showAverageFace(Viewer& viewer) {
    if(_meanFace.rows() == 0) {
        cout << "No mean face computed" << endl;
    }
    else {
        _faceIndex = -1;
        _weightEigenFaces.setZero();
        computeEigenFaceOffsetIndex();
        viewer.data().clear();
        viewer.data().set_mesh(_meanFace, F);
    }
}

void updateWeightEigenFaces() {
    _weightEigenFaces.resize(_nEigenFaces);
    if(_faceIndex == -1) {
        _weightEigenFaces.setZero();
    }
    else {
        for(int i = 0; i < _nEigenFaces; i++) {
            _weightEigenFaces(i) = _weightEigenFacesPerFace(i,_faceIndex);
        }
    }
    computeEigenFaceOffsetIndex();
}

void updateFaceIndex(Viewer& viewer) {
    _faceIndex = min(max(0,_faceIndex),(int) (_faceList.size() - 1));
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
    }
    else {
        viewer.data().clear();
        viewer.data().set_mesh(_faceList[_faceIndex], F);
    }

    if(_weightEigenFacesPerFace.rows() == 0) {
        cout << "No weights computed" << endl;
    }
    else {
        updateWeightEigenFaces();
    }
}

void showFace(Viewer& viewer) {
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
    }
    else {
        viewer.data().clear();
        if(_faceIndex == -1) {
            viewer.data().set_mesh(_meanFace, F);
        }
        else {
            viewer.data().set_mesh(_faceList[_faceIndex], F);
        }
    }
}

void showEigenFaceOffset(Viewer& viewer) {
    if(_faceOffsets.empty()) {
        cout << "No faces with Eigen offsets computed" << endl;
    }
    else if(_meanFace.rows() == 0) {
        cout << "No mean face computed" << endl;
    }
    else {
        viewer.data().clear();
        viewer.data().set_mesh(_meanFace + _faceOffset, F);
    }
}

void showMorphedFace(Viewer& viewer) {
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
    }
    else if(_meanFace.rows() == 0) {
        cout << "No mean face computed" << endl;
    }
    else if(_faceOffsets.empty()) {
        cout << "No faces with Eigen offsets computed" << endl;
    }
    else {
        viewer.data().clear();
        if(_faceIndex == -1) {
            viewer.data().set_mesh(_meanFace + _morphLambda * _faceOffsets[_morphIndex], F);
        }
        else {
            viewer.data().set_mesh(_meanFace + _morphLambda * _faceOffsets[_morphIndex] + (1 - _morphLambda) * _faceOffset, F);
        }
    }
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

    menu.callback_draw_custom_window = [&]() {
        computeMeanFace();
        computeDeviation();
        computePCA();
        computeEigenFaceWeights();
        computeEigenFaceOffsets();
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("PCA Menu", NULL, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Data set: %s",_currentData.c_str());
        if (ImGui::Button("Load faces", ImVec2(-1,0))) {
            loadFaces(viewer, false);
        }

        if (ImGui::Button("Show average face", ImVec2(-1,0))) {
            showAverageFace(viewer);
        }

        if(ImGui::InputInt("Face index", &_faceIndex)) {
            updateFaceIndex(viewer);
        }

        if (ImGui::Button("Show face", ImVec2(-1,0))) {
            showFace(viewer);
        }

        ImGui::Separator();

        if(ImGui::InputInt("#Eigen faces", &_nEigenFaces)) {
            _nEigenFaces = min(max(1,_nEigenFaces), _maxEigenFaces);
            initializeParameters();
            recomputeAll();
            updateWeightEigenFaces();
            showEigenFaceOffset(viewer);
        }

        for(int i = 0; i < _nEigenFaces; i++) {
            if(ImGui::SliderFloat(("Eigen face " + to_string(i)).c_str(), &_weightEigenFaces(i),-1000.0,1000.0)) {
                computeEigenFaceOffsetIndex();
                viewer.data().clear();
                viewer.data().set_mesh(_meanFace + _faceOffset, F);
            }
        }

        if (ImGui::Button("Show face with Eigen face offsets", ImVec2(-1,0))) {
            showEigenFaceOffset(viewer);
        }

        if(ImGui::InputInt("Morph Face Index", &_morphIndex)) {
            _morphIndex = min(max(0, _morphIndex), (int) (_faceList.size() - 1));
            showMorphedFace(viewer);
        }

        if(ImGui::SliderFloat("Morphing Variable", &_morphLambda,0,1)) {
            showMorphedFace(viewer);
        }

        if (ImGui::Button("Show morphed face", ImVec2(-1,0))) {
            showMorphedFace(viewer);
        }
        ImGui::End();
    };

    // PCA initialization
    _currentData = *_dataExamples.begin();
    loadFaces(viewer, true);

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