//
// Created by Nicolas Wicki on 30.05.21.
//

#include "PCA.h"

void PCA::convert3Dto1D(MatrixXd& m) {
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

void PCA::convert1Dto3D(MatrixXd& m) {
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

bool PCA::endsWith(const string& str, const string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

void PCA::initializeParameters() {
    _weightEigenFaces.resize(max((int) _faceList.size(),_nEigenFaces));
    _weightEigenFaces.setZero();
}

void PCA::loadFaces(Viewer& viewer, MatrixXi& F, bool init) {
    cout << "Load faces from " << _dataExamples[_currentData] << endl;
    _faceFiles = set<string>();
    DIR *directory;
    struct dirent *entry;
    if ((directory = opendir(_dataExamples[_currentData])) != NULL) {
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
        cerr << "Failed to load faces: " << _dataExamples[_currentData] << endl;
        return;
    }

    // Store faces in a list
    MatrixXd vertices;
    MatrixXi faces;
    _faceList = vector<MatrixXd>(_faceFiles.size());
    int i = 0;
    for(auto it = _faceFiles.begin(); it != _faceFiles.end(); it++){
        string file = _dataExamples[_currentData] + *it;
        cout << "Read file: " << file << "\n";
        igl::read_triangle_mesh(file,vertices,faces);
        _faceList[i++] = vertices;
    }
    F = faces;
    initializeParameters();
    recomputeAll();
    updateFaceIndex(viewer, F);
    showFace(viewer, F);
    viewer.snap_to_canonical_quaternion();
    viewer.core.align_camera_center(viewer.data().V, viewer.data().F);
    cout << endl;
}

void PCA::computeMeanFace() {
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
    cout << endl;
}

void PCA::computeDeviation() {
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
    cout << endl;
}

void PCA::computePCA() {
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
    _eigenFaces.resize(decreasing.rows(), _faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        _eigenFaces.col(i) = decreasing.col(decreasing.cols() - 1 - i);
    }
    // Result runtime
    auto end = chrono::high_resolution_clock::now();
    cout << "PCA execution time: " << chrono::duration_cast<chrono::milliseconds> (end-start).count() << " ms" << endl;
    cout << endl;
}

void PCA::computeEigenFaceWeights() {
    if(_eigenFaces.cols() != _faceList.size()) {
        cout << "Not enough eigen faces available" << endl;
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute weights for each face and corresponding Eigenfaces" << endl;
    _weightEigenFacesPerFace.resize((int) _faceList.size(),(int) _faceList.size());
    _weightEigenFacesMinMax.resize((int) _faceList.size(),2);
    _weightEigenFacesMinMax.setZero();
    for(int i = 0; i < _faceList.size(); i++) {
        for(int j = 0; j < _faceList.size(); j++) {
            double weight = _PCA_A.col(i).dot(_eigenFaces.col(j));
            _weightEigenFacesMinMax(j,0) = min(_weightEigenFacesMinMax(j,0), weight);
            _weightEigenFacesMinMax(j,1) = max(_weightEigenFacesMinMax(j,1), weight);
            _weightEigenFacesPerFace(j,i) = weight;
        }
    }
    _weightEigenFacesMinMax.col(0) = _weightEigenFacesMinMax.col(0).cwiseAbs();
    for(int i = 0; i < _faceList.size(); i++) {
        for(int j = 0; j < _faceList.size(); j++) {
            double weight = _weightEigenFacesPerFace(j,i);
            if(weight < 0) {
                _weightEigenFacesPerFace(j,i) = weight / _weightEigenFacesMinMax(j,0);
            }
            else if(0 < weight) {
                _weightEigenFacesPerFace(j,i) = weight / _weightEigenFacesMinMax(j,1);
            }
            else {
                _weightEigenFacesPerFace(j,i) = 0.0;
            }
        }
    }
    cout << endl;
}

void PCA::computeEigenFaceOffsets() {
    if(_weightEigenFacesPerFace.cols() < _faceList.size()) {
        cout << "Not enough weights available" << endl;
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    cout << "Compute eigen face offsets" << endl;
    _faceOffsets = vector<MatrixXd>(_faceList.size());
    for(int i = 0; i < _faceList.size(); i++) {
        MatrixXd sum(_faceList[i].rows(),_faceList[i].cols());
        sum.setZero();
        for(int j = 0; j < _nEigenFaces; j++) {
            double weight = _weightEigenFacesPerFace(j,i);
            MatrixXd weighted;
            if(weight < 0) {
                weighted = weight * _weightEigenFacesMinMax(j,0) * _eigenFaces.col(j);
            }
            else if(0 < weight) {
                weighted = weight * _weightEigenFacesMinMax(j,1) * _eigenFaces.col(j);
            }
            else {
                weighted.resize(_eigenFaces.rows(),1);
                weighted.setZero();
            }
            convert1Dto3D(weighted);
            sum += weighted;
        }
        _faceOffsets[i] = sum;
    }
    cout << endl;
}

void PCA::computeEigenFaceOffsetIndex() {
    if(_weightEigenFacesPerFace.cols() < _faceList.size()) {
        cout << "Not enough weights available" << endl;
        return;
    }
    if(_faceList.empty()) {
        cout << "No faces loaded" << endl;
        return;
    }
    MatrixXd sum(_faceList[0].rows(),_faceList[0].cols());
    sum.setZero();
    if(_faceIndex == -1) {
        _faceOffset = sum;
        return;
    }
    for(int j = 0; j < _nEigenFaces; j++) {
        double weight = _weightEigenFaces(j);
        MatrixXd weighted;
        if(weight < 0) {
            weighted = weight * _weightEigenFacesMinMax(j,0) * _eigenFaces.col(j);
        }
        else if(0 < weight) {
            weighted = weight * _weightEigenFacesMinMax(j,1) * _eigenFaces.col(j);
        }
        else {
            weighted.resize(_eigenFaces.rows(),1);
            weighted.setZero();
        }
        convert1Dto3D(weighted);
        sum += weighted;
    }
    _faceOffset = sum;
}

void PCA::recomputeAll() {
    computeMeanFace();
    computeDeviation();
    computePCA();
    computeEigenFaceWeights();
    computeEigenFaceOffsets();
    computeEigenFaceOffsetIndex();
}

void PCA::showAverageFace(Viewer& viewer, MatrixXi& F) {
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

void PCA::updateWeightEigenFaces() {
    if(_faceIndex == -1) {
        _weightEigenFaces.setZero();
    }
    computeEigenFaceOffsetIndex();
}

void PCA::updateFaceIndex(Viewer& viewer, MatrixXi& F) {
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
        _weightEigenFaces.setZero();
        updateWeightEigenFaces();
    }
}

void PCA::showFace(Viewer& viewer, MatrixXi& F) {
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

void PCA::showApproximatedFace(Viewer &viewer, MatrixXi &F) {
    if(_faceOffsets.empty()) {
        cout << "No faces with Eigen offsets computed" << endl;
    }
    else if(_meanFace.rows() == 0) {
        cout << "No mean face computed" << endl;
    }
    else {
        viewer.data().clear();
        if(_faceIndex == -1) {
            viewer.data().set_mesh(_meanFace, F);
            _weightEigenFaces.setZero();
        }
        else {
            viewer.data().set_mesh(_meanFace + _faceOffsets[_faceIndex], F);
        }
    }
}

void PCA::setWeightsApproximatedFace() {
    if(_faceIndex == -1) {
        _weightEigenFaces.setZero();
    }
    else {
        for(int i = 0; i < _weightEigenFaces.rows(); i++) {
            _weightEigenFaces(i) = _weightEigenFacesPerFace(i,_faceIndex);
        }
    }
}

void PCA::showEigenFaceOffset(Viewer& viewer, MatrixXi& F) {
    if(_meanFace.rows() == 0) {
        cout << "No mean face computed" << endl;
    }
    else {
        computeEigenFaceOffsetIndex();
        viewer.data().clear();
        viewer.data().set_mesh(_meanFace + _faceOffset, F);
    }
}

void PCA::showMorphedFace(Viewer& viewer, MatrixXi& F) {
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
        if(_faceIndex == -1) {
            viewer.data().clear();
            viewer.data().set_mesh(_meanFace + _morphLambda * _faceOffsets[_morphIndex], F);
        }
        else {
            viewer.data().clear();
            viewer.data().set_mesh(_meanFace + _morphLambda * _faceOffsets[_morphIndex] + (1 - _morphLambda) * _faceOffset, F);
        }
    }
}

void PCA::showError(Viewer& viewer) {
    MatrixXd face;
    if(_faceIndex == -1) {
        face = _meanFace;
    }
    else {
        face = _faceList[_faceIndex];
    }
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
}

void PCA::saveMesh(Viewer& viewer) {
    struct stat buffer;
    if(stat (_PCA_Results.c_str(), &buffer) != 0) {
        if(mkdir(_PCA_Results.c_str(), 0777) == -1) {
            cout << "Folder creation failed" << endl;
            return;
        }
    }
    string fileName = _PCA_Results + to_string(chrono::system_clock::now().time_since_epoch().count()) + ".obj";
    cout << "Writing current mesh to " << fileName << endl;
    igl::writeOBJ(fileName,viewer.data().V,viewer.data().F);
    cout << endl;
}