#ifndef ASSIGNMENT6_VAE_H
#define ASSIGNMENT6_VAE_H
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

class VAE {
public:
    // Variables
    set<string> _realFaceFiles = set<string>(); // real registered faces
    set<string> _vaeFeatureFiles = set<string>(); // latent variable from VAE
    vector<MatrixXd> _realFaceList = vector<MatrixXd>();

    // Amount of features considered
    const int _nFeatures = 16;
    // Data folder
    int _currentData = 0;
    // Face index chosen by user
    int _faceIndex = 0;
    // Weights of the face displayed in [-1,1]
    VectorXf _weightFeatures = VectorXf::Zero(_nFeatures);
    // Columns hold weight per feature #features x #faces
    MatrixXd _weightFeaturesPerFace = MatrixXd(0,0);
    // Column 0 -> minimal weights, Column 1 -> maximal weights
    MatrixXd _weightFeaturesMinMax = MatrixXd(0,0);

    //VAE Model
    vector<MatrixXd> _modelWeights;
    vector<VectorXd> _modelBiases;
    const vector<string> _modelWeightFiles = {
            "../data/vae_faces/model/enc1_weight.txt",
            "../data/vae_faces/model/enc2_weight.txt",
            "../data/vae_faces/model/enc_mu_weight.txt",
            "../data/vae_faces/model/enc_var_weight.txt",
            "../data/vae_faces/model/dec1_weight.txt",
            "../data/vae_faces/model/dec2_weight.txt",
            "../data/vae_faces/model/dec3_weight.txt"
    };

    const vector<string> _modelBiasFiles = {
            "../data/vae_faces/model/enc1_bias.txt",
            "../data/vae_faces/model/enc2_bias.txt",
            "../data/vae_faces/model/enc_mu_bias.txt",
            "../data/vae_faces/model/enc_var_bias.txt",
            "../data/vae_faces/model/dec1_bias.txt",
            "../data/vae_faces/model/dec2_bias.txt",
            "../data/vae_faces/model/dec3_bias.txt",
    };

    // Files
    const vector<const char*> _dataExamples = {
            "Choose dataset",
            "../data/vae_faces/eval/",
            "../data/vae_faces/train/"
    };

    // Functions
    bool endsWith(const string& str, const string& suffix);
    void initializeParameters();
    void loadFaces(Viewer& viewer, MatrixXi& F);
    void updateFaceIndex(Viewer& viewer, MatrixXi& F);
    void showFace(Viewer& viewer, MatrixXi& F);
    void showReconstructedFace(Viewer& viewer, MatrixXi& F);
    void setWeightsReconstructedFace();
    void showError(Viewer& viewer);
    void forwardDecoder(VectorXd z, MatrixXd& out);
    VectorXd denormalizeWeight(VectorXd w);
};


#endif //ASSIGNMENT6_VAE_H
