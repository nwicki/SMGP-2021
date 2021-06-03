//
// Created by Nicolas Wicki on 30.05.21.
//

#ifndef ASSIGNMENT6_PCA_H
#define ASSIGNMENT6_PCA_H
// Includes
#include <string>
#include <dirent.h>
#include <set>
#include <sys/stat.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

class PCA {
public:
    // Variables
    // List of faces already preprocessed for PCA
    set<string> _faceFiles = set<string>();
    vector<MatrixXd> _faceList = vector<MatrixXd>();
    vector<MatrixXd> _faceListDeviation = vector<MatrixXd>();

    // Offset of all faces
    vector<MatrixXd> _faceOffsets = vector<MatrixXd>();
    // Face index offset
    MatrixXd _faceOffset = MatrixXd(0,0);
    // cov = A * A^T
    MatrixXd _PCA_A = MatrixXd(0,0);
    // Covariance matrix for Eigen decomposition
    MatrixXd _PCA_Covariance = MatrixXd(0,0);
    // Average face
    MatrixXd _meanFace = MatrixXd(0,0);
    // Eigen faces #3*vertices x #faces
    MatrixXd _eigenFaces = MatrixXd(0,0);
    // Amount of Eigen faces considered
    int _nEigenFaces = 10;
    // Data folder
    int _currentData = 0;
    // Face index chosen by user
    int _faceIndex = 0;
    // Weights of the face displayed in [0,1]
    VectorXf _weightEigenFaces = VectorXf(0);
    // Columns hold weight per Eigen face #Eigen faces x #faces
    MatrixXd _weightEigenFacesPerFace = MatrixXd(0,0);
    // Column 0 -> minimal weights, Column 1 -> maximal weights
    MatrixXd _weightEigenFacesMinMax = MatrixXd(0,0);
    // Face index to morph the current one with
    int _morphIndex = 0;
    // Morph factor
    float _morphLambda = 0;
    // Show error
    bool _showError;

    // Constant variables
    const vector<const char*> _dataExamples = {
            "Choose dataset",
            "../data/aligned_faces_example/example1/",
            "../data/aligned_faces_example/example2/",
            "../data/aligned_faces_example/example3/",
            "../data/aligned_faces/"
    };
    const string _PCA_Results = "../data/pca-results/";

    // Functions
    PCA() {
        initializeParameters();
    }
    bool endsWith(const string& str, const string& suffix);
    void convert3Dto1D(MatrixXd& m);
    void convert1Dto3D(MatrixXd& m);
    void initializeParameters();
    void loadFaces(Viewer& viewer, MatrixXi& F, bool init);
    void computeMeanFace();
    void computeDeviation();
    void computePCA();
    void computeEigenFaceWeights();
    void computeEigenFaceOffsets();
    void computeEigenFaceOffsetIndex();
    void recomputeAll();
    void showAverageFace(Viewer& viewer, MatrixXi& F);
    void updateWeightEigenFaces();
    void updateFaceIndex(Viewer& viewer, MatrixXi& F);
    void showFace(Viewer& viewer, MatrixXi& F);
    void showApproximatedFace(Viewer& viewer, MatrixXi& F);
    void setWeightsApproximatedFace();
    void showEigenFaceOffset(Viewer& viewer, MatrixXi& F);
    void showMorphedFace(Viewer& viewer, MatrixXi& F);
    void showError(Viewer& viewer);
    void saveMesh(Viewer& viewer);
};


#endif //ASSIGNMENT6_PCA_H
