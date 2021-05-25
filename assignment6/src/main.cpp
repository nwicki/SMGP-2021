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


RowVector3d get_cartesian_coordinates(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3f barycentric_coordinates) {
    return barycentric_coordinates(0) * p0 + barycentric_coordinates(1) * p1 + barycentric_coordinates(2) * p2;
}

void add_vertex(int mouse_x, int mouse_y) {

    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view, viewer.core.proj, viewer.core.viewport, V, F, fid, bc)) {

        Eigen::RowVector3i face_point_indices = F.row(fid);
        Eigen::RowVector3d p0 = V.row(face_point_indices(0));
        Eigen::RowVector3d p1 = V.row(face_point_indices(1));
        Eigen::RowVector3d p2 = V.row(face_point_indices(2));

        Eigen::RowVector3d point_on_mesh;
        point_on_mesh << get_cartesian_coordinates(p0, p1, p2, bc);
        // paint hit red
        viewer.data().add_points(point_on_mesh, Eigen::RowVector3d(1, 0, 0));

        cout << "Added vertex" << endl;
    }
}

bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int) Viewer::MouseButton::Right)
        return false;

    add_vertex(viewer.current_mouse_x, viewer.current_mouse_y);

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
    }
  };

  viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;

  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.launch();

  return 0;

}