#include <iostream>
#include <eigen3/Eigen/Core>

void filter_cuboid(cv::Mat& rgb_img, Eigen::Matrix<double,4,4> transToWolrd, std::vector<cuboid *> &vcuboid, Eigen::MatrixXd all_lines_raw,
                   const Eigen::Vector4d &obj_bbox_coors);

bool check_inside_box(const Vector2d &pt, const Vector2d &box_left_top, const Vector2d &box_right_bottom);

void getVanishingPoints(const Matrix3d &KinvR, double yaw_esti, Vector2d &vp_1, Vector2d &vp_2, Vector2d &vp_3);

