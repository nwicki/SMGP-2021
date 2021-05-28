#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <vector>

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

class FaceRegistor {
public:

    void center_and_rescale_mesh(MatrixXd &V, const MatrixXd &P, double factor = 1.0);

    void center_and_rescale_template(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P);

    void align_rigid(MatrixXd &V_tmpl, const MatrixXd &P_tmpl, const MatrixXd &P);

};