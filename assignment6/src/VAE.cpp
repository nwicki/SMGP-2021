#include "VAE.h"
#include <fstream>


bool VAE::endsWith(const string& str, const string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

// Load weights and biases of saved VAE model
void VAE::initializeParameters() {
    cout << "Initialize parameters: Load VAE model weights and biases" << endl;
    _modelWeights.clear();
    _modelBiases.clear();

    // Load weights
    for(string file : _modelWeightFiles){
        cout << "Read file: " << file << "\n";
        ifstream infile(file);
        int rows, cols;
        infile >> rows >> cols;
        MatrixXd W(rows, cols);
        double value;
        int i = 0;
        int j = 0;
        while (infile >> value){
            if(j == cols){
                j = 0;
                i++;
            }
            W(i, j) = value;
            j++;
        }
        _modelWeights.push_back(W);
    }

    // Load biases
    for(string file : _modelBiasFiles){
        cout << "Read file: " << file << "\n";
        ifstream infile(file);
        int rows, cols;
        infile >> rows >> cols;
        VectorXd B(rows);
        double value;
        int i = 0;
        while (infile >> value){
            B(i) = value;
            i++;
        }
        _modelBiases.push_back(B);
    }
}

// Load real faces, reconstructed faces and corresponding features (latent variable)
void VAE::loadFaces(Viewer& viewer, MatrixXi& F) {
    cout << "Load faces from " << _dataExamples[_currentData] << endl;
    _realFaceFiles = set<string>();
    _vaeFeatureFiles = set<string>();
    DIR *directory;
    struct dirent *entry;
    if ((directory = opendir(_dataExamples[_currentData])) != NULL) {
        // Get all file paths and store in _faceFiles
        while ((entry = readdir(directory)) != NULL) {
            string current = entry->d_name;
            if(endsWith(current, ".obj")) {
                _realFaceFiles.insert(current);
            } else if(endsWith(current, "_features.txt")) {
                _vaeFeatureFiles.insert(current);
            }
        }
        closedir (directory);
    }
    else {
        cerr << "Failed to load faces: " << _dataExamples[_currentData] << endl;
        return;
    }

    // Store faces in a list
    MatrixXd vertices;
    MatrixXi faces;
    _realFaceList = vector<MatrixXd>(_realFaceFiles.size());
    int i = 0;
    for(auto it = _realFaceFiles.begin(); it != _realFaceFiles.end(); it++){
        string file = _dataExamples[_currentData] + *it;
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _realFaceList[i++] = vertices;
    }
    F = faces;

    //load the feature weights
    _weightFeaturesPerFace.resize((int) _nFeatures, (int) _realFaceList.size());
    _weightFeaturesMinMax.resize((int) _nFeatures, 2);
    _weightFeaturesMinMax.setZero();

    int face_idx = 0;
    for(string file : _vaeFeatureFiles){
        cout << "Read file: " << file << "\n";
        ifstream infile(_dataExamples[_currentData] + file);
        double value;
        int feature_idx = 0;
        while (infile >> value){
            _weightFeaturesPerFace(feature_idx, face_idx) = value;
            feature_idx++;
        }
        face_idx++;
    }

    _weightFeaturesMinMax.col(0) = _weightFeaturesPerFace.rowwise().minCoeff();
    _weightFeaturesMinMax.col(1) = _weightFeaturesPerFace.rowwise().maxCoeff();

    // normalize weight features
    for(int i=0; i<_weightFeaturesPerFace.rows(); i++){
        for(int j=0; j<_weightFeaturesPerFace.cols(); j++){
            _weightFeaturesPerFace(i,j) = 2 * (_weightFeaturesPerFace(i,j) - _weightFeaturesMinMax(i,0)) / (_weightFeaturesMinMax(i,1) - _weightFeaturesMinMax(i,0)) - 1;
        }
    }

    updateFaceIndex(viewer, F);
    showFace(viewer, F);
    viewer.snap_to_canonical_quaternion();
    viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
    cout << endl;
}

void VAE::updateFaceIndex(Viewer& viewer, MatrixXi& F) {
    _faceIndex = min(max(0,_faceIndex),(int) (_realFaceList.size() - 1));
    if(_realFaceList.empty()) {
        cout << "No faces loaded" << endl;
    }
    else {
        viewer.data().clear();
        viewer.data().set_mesh(_realFaceList[_faceIndex], F);
    }

    if(_weightFeaturesPerFace.rows() == 0) {
        cout << "No weights computed" << endl;
    }
    else {
        _weightFeatures = _weightFeaturesPerFace.col(_faceIndex).cast<float>();
    }
}

// Show real registered ground truth face
void VAE::showFace(Viewer& viewer, MatrixXi& F) {
    if(_realFaceList.empty()) {
        cout << "No faces loaded" << endl;
    }
    else {
        viewer.data().clear();
        viewer.data().set_mesh(_realFaceList[_faceIndex], F);
    }
}

// Show reconstructed face computed using the saved latent variable
// and passing it through the VAE model decoder
void VAE::showReconstructedFace(Viewer &viewer, MatrixXi &F) {
    VectorXd z = denormalizeWeight(_weightFeatures.cast<double>());
    MatrixXd V;
    forwardDecoder(z, V);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
}

// Set the feature weights corresponding to the face
void VAE::setWeightsReconstructedFace() {
    if(_faceIndex == -1) {
        _weightFeatures.setZero();
    }
    else {
        for(int i = 0; i < _weightFeatures.rows(); i++) {
            _weightFeatures(i) = _weightFeaturesPerFace(i,_faceIndex);
        }
    }
}

// Show L1 error between original and reconstructed mesh
void VAE::showError(Viewer& viewer) {
    MatrixXd face;
    face = _realFaceList[_faceIndex];
    MatrixXd V = viewer.data().V;
    if(V.rows() != face.rows() || V.cols() != face.cols()) {
        cout << "Cannot compare two faces of different sizes" << endl;
        cout << "Size of currently shown face: " << V.rows() << " rows, " << V.cols() << " cols" << endl;
        cout << "Size of face compared to: " << face.rows() << " rows, " << face.cols() << " cols" << endl;
        return;
    }
    VectorXd error = (V - face).rowwise().norm();
    MatrixXd C;
    igl::colormap(igl::COLOR_MAP_TYPE_JET, error, 0.0, 4.0, C); //empirical range
    viewer.data().set_colors(C);
    cout << "Mean error: " << error.mean() << endl;
}

// Pass a latent variable through the decoder and outputs a set of vertex positions
void VAE::forwardDecoder(VectorXd z, MatrixXd& out) {
    VectorXd x1 = (_modelWeights[4] * z + _modelBiases[4]).cwiseMax(0); // x1 = relu(dec1(z))
    VectorXd x2 = (_modelWeights[5] * x1 + _modelBiases[5]).cwiseMax(0); // x2 = relu(dec2(x1))
    VectorXd x = _modelWeights[6] * x2 + _modelBiases[6]; // x = dec3(x2)
    out.resize(x.rows()/3, 3);
    for(int i=0; i<out.rows(); i++){
        for(int j=0; j<out.cols(); j++){
            out(i,j) = x(3*i+j);
        }
    }
}

// Convert weights in range (-1, 1) to real weight values
VectorXd VAE::denormalizeWeight(VectorXd w) {
    VectorXd r = (0.5 * (w.array() + 1.0) * (_weightFeaturesMinMax.col(1) - _weightFeaturesMinMax.col(0)).array() +  _weightFeaturesMinMax.col(0).array()).matrix();
    return r;
}