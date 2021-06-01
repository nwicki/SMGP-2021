#include <igl/opengl/glfw/Viewer.h>
#include <nanoflann.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
namespace fs = boost::filesystem;
using Viewer = igl::opengl::glfw::Viewer;

class Preprocessor {
public:
    string mesh_folder_path = "../data/scanned_faces_cleaned/";
    vector<string> mesh_names;
    int mesh_id = 0;

    string save_folder_path = "../data/preprocessed_faces/";

    double iso_value = 5.0;
    double max_dist = 0.0;
    VectorXd scalar_field;

    Preprocessor() {
        mesh_names.clear();
        for (auto const & file : fs::recursive_directory_iterator(mesh_folder_path)) {
            if (fs::is_regular_file(file) && file.path().extension() == ".obj")
                mesh_names.emplace_back(file.path().stem().string());
        }
        sort(mesh_names.begin(), mesh_names.end());
    }

    void clean_connected_components(Viewer& viewer, MatrixXd &V, MatrixXi &F);

    void compute_distance_to_boundary(Viewer& viewer, MatrixXd &V, MatrixXi&F);

    void smooth_distance_field(Viewer& viewer, MatrixXd &V, MatrixXi&F, int num_iter=1, float w = 0.02);

    void remesh(Viewer& viewer, MatrixXd &V, MatrixXi&F);

    void preprocess(Viewer& viewer, MatrixXd &V, MatrixXi&F, int num_iter=1, float w = 0.02);

    string save_mesh(MatrixXd &V, MatrixXi&F);

};