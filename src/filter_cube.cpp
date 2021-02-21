// std c
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
#include <numeric>
// opencv pcl
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

//ours
#include "main.h"

using namespace std;
using namespace Eigen;


bool check_inside_box(const Vector2d &pt, const Vector2d &box_left_top, const Vector2d &box_right_bottom)
{
    return box_left_top(0) <= pt(0) && pt(0) <= box_right_bottom(0) && box_left_top(1) <= pt(1) && pt(1) <= box_right_bottom(1);
}

void fast_RemoveRow(MatrixXd &matrix, int rowToRemove, int &total_line_number)
{
    matrix.row(rowToRemove) = matrix.row(total_line_number - 1);
    total_line_number--;
}

void atan2_vector(const VectorXd &y_vec, const VectorXd &x_vec, VectorXd &all_angles)
{
    all_angles.resize(y_vec.rows());
    for (int i = 0; i < y_vec.rows(); i++)
        all_angles(i) = std::atan2(y_vec(i), x_vec(i)); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
}

void merge_break_lines(const MatrixXd &all_lines, MatrixXd &merge_lines_out, double pre_merge_dist_thre,
                       double pre_merge_angle_thre_degree, double edge_length_threshold)
{
    bool can_force_merge = true;
    merge_lines_out = all_lines;
    int total_line_number = merge_lines_out.rows(); // line_number will become smaller and smaller, merge_lines_out doesn't change
    int counter = 0;
    double pre_merge_angle_thre = pre_merge_angle_thre_degree / 180.0 * M_PI;
    while ((can_force_merge) && (counter < 500))
    {
        counter++;
        can_force_merge = false;
        MatrixXd line_vector = merge_lines_out.topRightCorner(total_line_number, 2) - merge_lines_out.topLeftCorner(total_line_number, 2);
        VectorXd all_angles;
        atan2_vector(line_vector.col(1), line_vector.col(0), all_angles); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
        for (int seg1 = 0; seg1 < total_line_number - 1; seg1++)
        {
            for (int seg2 = seg1 + 1; seg2 < total_line_number; seg2++)
            {
                double diff = std::abs(all_angles(seg1) - all_angles(seg2));
                double angle_diff = std::min(diff, M_PI - diff);
                if (angle_diff < pre_merge_angle_thre)
                {
                    double dist_1ed_to_2 = (merge_lines_out.row(seg1).tail(2) - merge_lines_out.row(seg2).head(2)).norm();
                    double dist_2ed_to_1 = (merge_lines_out.row(seg2).tail(2) - merge_lines_out.row(seg1).head(2)).norm();

                    if ((dist_1ed_to_2 < pre_merge_dist_thre) || (dist_2ed_to_1 < pre_merge_dist_thre))
                    {
                        Vector2d merge_start, merge_end;
                        if (merge_lines_out(seg1, 0) < merge_lines_out(seg2, 0))
                            merge_start = merge_lines_out.row(seg1).head(2);
                        else
                            merge_start = merge_lines_out.row(seg2).head(2);

                        if (merge_lines_out(seg1, 2) > merge_lines_out(seg2, 2))
                            merge_end = merge_lines_out.row(seg1).tail(2);
                        else
                            merge_end = merge_lines_out.row(seg2).tail(2);

                        double merged_angle = std::atan2(merge_end(1) - merge_start(1), merge_end(0) - merge_start(0));
                        double temp = std::abs(all_angles(seg1) - merged_angle);
                        double merge_angle_diff = std::min(temp, M_PI - temp);
                        if (merge_angle_diff < pre_merge_angle_thre)
                        {
                            merge_lines_out.row(seg1).head(2) = merge_start;
                            merge_lines_out.row(seg1).tail(2) = merge_end;
                            fast_RemoveRow(merge_lines_out, seg2, total_line_number); //also decrease  total_line_number
                            can_force_merge = true;
                            break;
                        }
                    }
                }
            }
            if (can_force_merge)
                break;
        }
    }
    //     std::cout<<"total_line_number after mege  "<<total_line_number<<std::endl;
    if (edge_length_threshold > 0)
    {
        MatrixXd line_vectors = merge_lines_out.topRightCorner(total_line_number, 2) - merge_lines_out.topLeftCorner(total_line_number, 2);
        VectorXd line_lengths = line_vectors.rowwise().norm();
        int long_line_number = 0;
        MatrixXd long_merge_lines(total_line_number, 4);
        for (int i = 0; i < total_line_number; i++)
        {
            if (line_lengths(i) > edge_length_threshold)
            {
                long_merge_lines.row(long_line_number) = merge_lines_out.row(i);
                long_line_number++;
            }
        }
        merge_lines_out = long_merge_lines.topRows(long_line_number);
    }
    else
        merge_lines_out.conservativeResize(total_line_number, NoChange);
}

void getVanishingPoints(const Matrix3d &KinvR, double yaw_esti, Vector2d &vp_1, Vector2d &vp_2, Vector2d &vp_3)
{
    vp_1 = homo_to_real_coord_vec<double>(KinvR * Vector3d(cos(yaw_esti), sin(yaw_esti), 0));  // for object x axis
    vp_2 = homo_to_real_coord_vec<double>(KinvR * Vector3d(-sin(yaw_esti), cos(yaw_esti), 0)); // for object y axis
    vp_3 = homo_to_real_coord_vec<double>(KinvR * Vector3d(0, 0, 1));                          // for object z axis
}

double box_edge_sum_dists(const cv::Mat &dist_map, const MatrixXi &box_corners_2d, const MatrixXi &edge_pt_ids, bool reweight_edge_distance)
{
    // give some edges, sample some points on line then sum up distance from dist_map
    // input: visible_edge_pt_ids is n*2  each row stores an edge's two end point's index from box_corners_2d
    // if weight_configs: for configuration 1, there are more visible edges compared to configuration2, so we need to re-weight
    // [1 2;2 3;3 4;4 1;2 6;3 5;4 8;5 8;5 6];  reweight vertical edge id 5-7 by 2/3, horizontal edge id 8-9 by 1/2
    float sum_dist = 0;   //bottom 5678,then top 1234
    for (int edge_id = 0; edge_id < edge_pt_ids.rows(); edge_id++)
    {
        Vector2i corner_tmp1 = box_corners_2d.col(edge_pt_ids(edge_id, 0));
        Vector2i corner_tmp2 = box_corners_2d.col(edge_pt_ids(edge_id, 1));
        for (double sample_ind = 0; sample_ind < 11; sample_ind++)
        {
            Vector2i sample_pt =  sample_ind / 10.0 * corner_tmp1 + (1 - sample_ind / 10.0) * corner_tmp2;
            float dist1 = dist_map.at<float>(int(sample_pt(1)), int(sample_pt(0))); //make sure dist_map is float type
            if (reweight_edge_distance)
            {
                if ((4 <= edge_id) && (edge_id <= 5))
                    dist1 = dist1 * 3.0 / 2.0;
                if (6 == edge_id)
                    dist1 = dist1 * 2.0;
            }
            sum_dist = sum_dist + dist1;
        }
    }
    return double(sum_dist);
}

// remove the jumping angles from -pi to pi.   to make the raw angles smoothly change.
void smooth_jump_angles(const VectorXd &raw_angles, VectorXd &new_angles)
{
    new_angles = raw_angles;
    if (raw_angles.rows() == 0)
        return;

    double angle_base = raw_angles(0); // choose a new base angle.   (assume that the all the angles lie in [-pi pi] around the base)
    for (int i = 0; i < raw_angles.rows(); i++)
    {
        if ((raw_angles(i) - angle_base) < -M_PI)
            new_angles(i) = raw_angles(i) + 2 * M_PI;
        else if ((raw_angles(i) - angle_base) > M_PI)
            new_angles(i) = raw_angles(i) - 2 * M_PI;
    }
}

template <class T>
T normalize_to_pi(T angle)
{
    if (angle > M_PI / 2)
        return angle - M_PI; // # change to -90 ~90
    else if (angle < -M_PI / 2)
        return angle + M_PI;
    else
        return angle;
}
template double normalize_to_pi(double);

// VPs 3*2   edge_mid_pts: n*2   vp_support_angle_thres 1*2
// output: 3*2  each row is a VP's two boundary supported edges' angle.  if not found, nan for that entry
Eigen::MatrixXd VP_support_edge_infos(Eigen::MatrixXd &VPs, Eigen::MatrixXd &edge_mid_pts, Eigen::VectorXd &edge_angles,
                                      Eigen::Vector2d vp_support_angle_thres)
{
    MatrixXd all_vp_bound_edge_angles = MatrixXd::Ones(3, 2) * nan(""); // initialize as nan  use isnan to check
    if (edge_mid_pts.rows() > 0)
    {
        for (int vp_id = 0; vp_id < VPs.rows(); vp_id++)
        {
            double vp_angle_thre;
            if (vp_id != 2)
                vp_angle_thre = vp_support_angle_thres(0) / 180.0 * M_PI;
            else
                vp_angle_thre = vp_support_angle_thres(1) / 180.0 * M_PI;

            std::vector<int> vp_inlier_edge_id;
            VectorXd vp_edge_midpt_angle_raw_inlier(edge_angles.rows());
            for (int edge_id = 0; edge_id < edge_angles.rows(); edge_id++)
            {
                double vp1_edge_midpt_angle_raw_i = atan2(edge_mid_pts(edge_id, 1) - VPs(vp_id, 1), edge_mid_pts(edge_id, 0) - VPs(vp_id, 0));
                double vp1_edge_midpt_angle_norm_i = normalize_to_pi<double>(vp1_edge_midpt_angle_raw_i);
                double angle_diff_i = std::abs(edge_angles(edge_id) - vp1_edge_midpt_angle_norm_i);
                angle_diff_i = std::min(angle_diff_i, M_PI - angle_diff_i);
                if (angle_diff_i < vp_angle_thre)
                {
                    vp_edge_midpt_angle_raw_inlier(vp_inlier_edge_id.size()) = vp1_edge_midpt_angle_raw_i;
                    vp_inlier_edge_id.push_back(edge_id);
                }
            }
            if (vp_inlier_edge_id.size() > 0) // if found inlier edges
            {
                VectorXd vp1_edge_midpt_angle_raw_inlier_shift;
                smooth_jump_angles(vp_edge_midpt_angle_raw_inlier.head(vp_inlier_edge_id.size()),
                                   vp1_edge_midpt_angle_raw_inlier_shift);
                int vp1_low_edge_id;
                vp1_edge_midpt_angle_raw_inlier_shift.maxCoeff(&vp1_low_edge_id);
                int vp1_top_edge_id;
                vp1_edge_midpt_angle_raw_inlier_shift.minCoeff(&vp1_top_edge_id);
                if (vp_id > 0)
                    std::swap(vp1_low_edge_id, vp1_top_edge_id);                                      // match matlab code
                all_vp_bound_edge_angles(vp_id, 0) = edge_angles(vp_inlier_edge_id[vp1_low_edge_id]); // it will be 0*1 matrix if not found inlier edges.
                all_vp_bound_edge_angles(vp_id, 1) = edge_angles(vp_inlier_edge_id[vp1_top_edge_id]);
            }
        }
    }
    return all_vp_bound_edge_angles;
}

void filter_cuboid(cv::Mat& rgb_img, Eigen::Matrix<double,4,4> transToWolrd, std::vector<cuboid *>  &vcuboid, Eigen::MatrixXd all_lines_raw,
                   const Eigen::Vector4d &obj_bbox_coors)
{

    //----------------------------------------TODO
    Matrix3d Kalib;
    Kalib << 535.4,  0,  320.1,   // for TUM cabinet data.
            0,  539.2, 247.6,
            0,      0,     1;

    Eigen::Matrix3d KinvR;
    KinvR = Kalib * transToWolrd.topLeftCorner<3, 3>().inverse();
    //----------------------------------------
    cv::Mat rgb_img_raw;
    rgb_img_raw = rgb_img.clone();
    cv::Mat gray_img;
    if (rgb_img.channels() == 3)
        cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);
    else
        gray_img = rgb_img;

    int img_width = rgb_img.cols;
    int img_height = rgb_img.rows;

    align_left_right_edges(all_lines_raw); // this should be guaranteed when detecting edges

    bool whether_plot_detail_images = false;
    bool reweight_edge_distance = false;
    //bool
    //bool
    //bool
    // parameters for cuboid generation
    double vp12_edge_angle_thre = 10;
    double vp3_edge_angle_thre = 10;	// 10  10  parameters


    if (whether_plot_detail_images)
    {
        cv::Mat output_img;
        plot_image_with_edges(rgb_img, output_img, all_lines_raw, cv::Scalar(255, 0, 0));
        cv::imshow("Raw detected Edges", output_img);
        cv::waitKey(0);
    }

    int object_id = 0;    //in tum dataset only one object each image  TODO for other datasets
    //2d bbox of object
    int left_x_raw = obj_bbox_coors(0);   //[x1y1,x2y2]
    int top_y_raw = obj_bbox_coors(1);
    int right_x_raw = obj_bbox_coors(2);
    int down_y_raw = obj_bbox_coors(3);
    int obj_width_raw = right_x_raw - left_x_raw;
    int obj_height_raw = down_y_raw - top_y_raw;
    int distmap_expand_wid = 10;     //  expand 10 piexls

    int left_x_expan_distmap = max(0, left_x_raw - distmap_expand_wid);
    int right_x_expan_distmap = min(img_width - 1, right_x_raw + distmap_expand_wid);
    int top_y_expan_distmap = max(0, top_y_raw - distmap_expand_wid);
    int down_y_expan_distmap = min(img_height - 1, down_y_raw + distmap_expand_wid);
    int height_expan_distmap = down_y_expan_distmap - top_y_expan_distmap;
    int width_expan_distmap = right_x_expan_distmap - left_x_expan_distmap;

    Vector2d expan_distmap_lefttop = Vector2d(left_x_expan_distmap, top_y_expan_distmap);
    Vector2d expan_distmap_rightbottom = Vector2d(right_x_expan_distmap, down_y_expan_distmap);

    // find edges inside the object bounding box
    MatrixXd all_lines_inside_object(all_lines_raw.rows(), all_lines_raw.cols()); // first allocate a large matrix, then only use the toprows to avoid copy, alloc
    int inside_obj_edge_num = 0;  //两端点都在扩大后的框内则表示此线段在物体框内
    for (int edge_id = 0; edge_id < all_lines_raw.rows(); edge_id++)
        if (check_inside_box(all_lines_raw.row(edge_id).head<2>(), expan_distmap_lefttop, expan_distmap_rightbottom))
            if (check_inside_box(all_lines_raw.row(edge_id).tail<2>(), expan_distmap_lefttop, expan_distmap_rightbottom))
            {
                all_lines_inside_object.row(inside_obj_edge_num) = all_lines_raw.row(edge_id);
                inside_obj_edge_num++;
            }

    // merge edges and remove short lines, after finding object edges.  edge merge in small regions should be faster than all.
    double pre_merge_dist_thre = 20;
    double pre_merge_angle_thre = 5;
    double edge_length_threshold = 30;    //livingroom数据集远处物体是否适应这个阈值?
    MatrixXd all_lines_merge_inobj;
    merge_break_lines(all_lines_inside_object.topRows(inside_obj_edge_num), all_lines_merge_inobj, pre_merge_dist_thre,
                      pre_merge_angle_thre, edge_length_threshold);

    // compute edge angels and middle points
    VectorXd lines_inobj_angles(all_lines_merge_inobj.rows());//LSD检测的线段的角度
    MatrixXd edge_mid_pts(all_lines_merge_inobj.rows(), 2); //LSD检测的线段的中点, each row an angle
    for (int i = 0; i < all_lines_merge_inobj.rows(); i++)
    {
        lines_inobj_angles(i) = std::atan2(all_lines_merge_inobj(i, 3) - all_lines_merge_inobj(i, 1), all_lines_merge_inobj(i, 2) - all_lines_merge_inobj(i, 0)); //“高”与底边的正切 [-pi/2 -pi/2]
        edge_mid_pts.row(i).head<2>() = (all_lines_merge_inobj.row(i).head<2>() + all_lines_merge_inobj.row(i).tail<2>()) / 2; //前点+后点除以2
    }

    // detect canny edges and compute distance transform  NOTE opencv canny maybe different from matlab. but roughly same
    cv::Rect object_bbox = cv::Rect(left_x_expan_distmap, top_y_expan_distmap, width_expan_distmap, height_expan_distmap); //expanded 2d-bbox
    cv::Mat im_canny;
    cv::Canny(gray_img(object_bbox), im_canny, 80, 200); // low thre, high thre    im_canny 0 or 255   [80 200  40 100]
    cv::Mat dist_map;
    cv::distanceTransform(255 - im_canny, dist_map, CV_DIST_L2, 3); // dist_map is float datatype

    if (whether_plot_detail_images)
    {
        cv::imshow("im_canny", im_canny);
        cv::Mat dist_map_img;
        cv::normalize(dist_map, dist_map_img, 0.0, 1.0, cv::NORM_MINMAX);
        cv::imshow("normalized distance map", dist_map_img);
        cv::waitKey(0);
    }
    std::cout<<"size of vcuboid: "<<vcuboid.size()<<std::endl;
//    cout<<"111111"<<vcuboid[0]->rotY<<endl;
//    cout<<"222222"<<vcuboid[1]->rotY<<endl;
    std::vector<double> dist_error;
    for(size_t i=0; i < vcuboid.size(); i++)
    {
        //std::cout<<"yaw of object: "<<vcuboid[i]->rotY<<std::endl;
        Eigen::MatrixXi box_corners_2d_float(2,8); //2*8
        box_corners_2d_float = vcuboid[i]->box_corners_2d;   //2d point in image
        double sum_dist;
        //std::cout<<"coordinate in 2dbbox: \n"<<box_corners_2d_float<<std::endl;
        //transform the coordinate base on 2dbbox
        Eigen::MatrixXi box_corners_2d_float_shift(2,8);
        //std::cout<<"left_x_expan_distmap: "<<left_x_expan_distmap<<",top_y_expan_distmap:  "<<top_y_expan_distmap<<std::endl;
        box_corners_2d_float_shift.row(0) = box_corners_2d_float.row(0).array() - left_x_expan_distmap;
        box_corners_2d_float_shift.row(1) = box_corners_2d_float.row(1).array() - top_y_expan_distmap;
        //std::cout<<"coordinate in 2dbbox: \n"<<box_corners_2d_float_shift<<std::endl;
        MatrixXi visible_edge_pt_ids, vps_box_edge_pt_ids;
        visible_edge_pt_ids.resize(5,2);                 //(7, 2);
        //two range
        if(vcuboid[i]->yaw_in_cam > -1.25 && vcuboid[i]->yaw_in_cam < 2.8)
        visible_edge_pt_ids << 5, 6, 6, 7, 7, 8, 8, 5, 7, 3;   //5, 6, 6, 7, 7, 8, 8, 5, 8, 4, 5, 1, 1, 4;  //1, 2, 2, 3, 3, 4, 4, 1, 2, 6, 3, 5, 5, 6;
        else
        visible_edge_pt_ids << 5, 6, 6, 7, 7, 8, 8, 5, 5, 1; 
        vps_box_edge_pt_ids.resize(3, 4);
        vps_box_edge_pt_ids << 5, 6, 7, 8, 5, 8, 4, 1, 5, 1, 8, 4; // 1, 2, 3, 4, 4, 1, 5, 6, 3, 5, 2, 6  six edges. each row represents two edges [e1_1 e1_2   e2_1 e2_2;...] of one VP
        visible_edge_pt_ids.array() -= 1;
        vps_box_edge_pt_ids.array() -= 1;

        //for debug: drawing the chosen points
        // rgb_img = rgb_img_raw.clone();
        // cv::Point2i point1, point2, point3, point4, point_up, point_down;
        // point1 = cv::Point(box_corners_2d_float(0,4),box_corners_2d_float(1,4));
        // point2 = cv::Point(box_corners_2d_float(0,5),box_corners_2d_float(1,5));
        // point3 = cv::Point(box_corners_2d_float(0,6),box_corners_2d_float(1,6));
        // point4 = cv::Point(box_corners_2d_float(0,7),box_corners_2d_float(1,7));
        // point_up = cv::Point(box_corners_2d_float(0,visible_edge_pt_ids(4,0)), box_corners_2d_float(1,visible_edge_pt_ids(4,0)));
        // point_down = cv::Point(box_corners_2d_float(0,visible_edge_pt_ids(4,1)), box_corners_2d_float(1,visible_edge_pt_ids(4,1)));
        // cv::line(rgb_img, point_up, point_down, cv::Scalar(255, 0, 0), 2, CV_AA, 0); 
        // cv::line(rgb_img, point1, point2, cv::Scalar(0, 0, 255), 2, CV_AA, 0); //line 12
        // cv::line(rgb_img, point2, point3, cv::Scalar(0, 0, 255), 2, CV_AA, 0); //line 23
        // cv::line(rgb_img, point3, point4, cv::Scalar(0, 0, 255), 2, CV_AA, 0); //line 34
        // cv::line(rgb_img, point4, point1, cv::Scalar(0, 0, 255), 2, CV_AA, 0); //line 41
        // cv::imshow("cube_image_for_dist_diff",rgb_img);
        // cv::waitKey(0);
        //std::cout<<"alles ok bis hier"<<std::endl;
        //distance difference
        sum_dist = box_edge_sum_dists(dist_map, box_corners_2d_float_shift, visible_edge_pt_ids, reweight_edge_distance);
        sum_dist = abs(sum_dist);
        dist_error.push_back(sum_dist);
        std::cout<<"yaw of object: "<<vcuboid[i]->rotY<<std::endl;
        std::cout<<"sum_dist: "<<sum_dist<<std::endl;
        //cv::waitKey(0);


        //angle difference------------------------------------------------------------
        Eigen::Matrix<int, 2, 8> corners_2d;
        corners_2d = vcuboid[i]->box_corners_2d;
        cv::Point pt5 = cv::Point(corners_2d(0,4),corners_2d(1,4));  //点5
        cv::Point pt6 = cv::Point(corners_2d(0,5),corners_2d(1,5));  //6
        cv::Point pt7 = cv::Point(corners_2d(0,6),corners_2d(1,6));  //7
        cv::Point pt8 = cv::Point(corners_2d(0,7),corners_2d(1,7));
        rgb_img = rgb_img_raw.clone();
        cv::line(rgb_img, pt5, pt6, cv::Scalar(0, 255, 0), 2, CV_AA, 0); //line 1
        cv::line(rgb_img, pt6, pt7, cv::Scalar(255, 0, 0), 2, CV_AA, 0);
        //cv::line(rgb_img, pt2, pt3, cv::Scalar(0, 0, 255), 2, CV_AA, 0); //line 2
        //cv::line(rgb_img, pt3, pt0, cv::Scalar(255, 0, 0), 2, CV_AA, 0);

        double total_angle_diff;
        Vector2d vp_1, vp_2, vp_3;
        getVanishingPoints(KinvR, vcuboid[i]->rotY, vp_1, vp_2, vp_3);
        MatrixXd all_vps(3, 2);
        all_vps.row(0) = vp_1;
        all_vps.row(1) = vp_2;
        all_vps.row(2) = vp_3;
        std::cout<<"VP: \n"<<all_vps<<std::endl;
        //3*2 matrix 每行代表vp点对应的边的角度范围，例如第一行是vp1点对应的边的最小角度和最大角度（顺序可能是乱的，不是左小右大），在这个范围的边就都是vp1点的对应边
        MatrixXd all_vp_bound_edge_angles = VP_support_edge_infos(all_vps, edge_mid_pts, lines_inobj_angles,
                                                                Vector2d(vp12_edge_angle_thre, vp3_edge_angle_thre));
        //std::cout<<"all_vp_bound_edge_angles: \n"<<all_vp_bound_edge_angles<<std::endl;
        std::cout<<"vp1_2edges_angle_range: "<<all_vp_bound_edge_angles.row(0)<<std::endl;
        std::cout<<"vp2_2edges_angle_range: "<<all_vp_bound_edge_angles.row(1)<<std::endl;
        std::cout<<"point5: "<<pt5.x<<","<<pt5.y<<endl;
        std::cout<<"point6: "<<pt6.x<<","<<pt6.y<<endl;
        std::cout<<"point7: "<<pt7.x<<","<<pt7.y<<endl;
        double vp1_mean_angle = (all_vp_bound_edge_angles(0,0) + all_vp_bound_edge_angles(0,1))/2;
        double vp2_mean_angle = (all_vp_bound_edge_angles(1,0) + all_vp_bound_edge_angles(1,1))/2;

        Eigen::Vector2d line56_mid_pts, line67_mid_pts;
        line56_mid_pts<< (pt5.x+pt6.x)/2 , (pt5.y+pt6.y)/2;
        line67_mid_pts<< (pt6.x+pt7.x)/2 , (pt6.y+pt7.y)/2;

        double line56_angl,line67_angl;
        if(pt5.x > pt6.x)
            line56_angl = std::atan2(pt5.y - pt6.y, pt5.x - pt6.x); //“高”与底边的正切
        else
            line56_angl = std::atan2(pt6.y - pt5.y, pt6.x - pt5.x);
        if(pt6.x > pt7.x)
            line67_angl = std::atan2(pt6.y - pt7.y, pt6.x - pt7.x);
        else
            line67_angl = std::atan2(pt7.y - pt6.y, pt7.x - pt6.x);

        std::cout<<"1_line_angle: "<<line56_angl<<"; 2_line_angle: "<<line67_angl<<std::endl;
        double vp1_cube_edge_midpt_angle, vp2_cube_edge_midpt_angle;
        if(vp_1[0] < line56_mid_pts[0])
            vp1_cube_edge_midpt_angle = atan2(line56_mid_pts[1] - vp_1[1], line56_mid_pts[0] - vp_1[0]);
        else
            vp1_cube_edge_midpt_angle = atan2(vp_1[1] - line56_mid_pts[1], vp_1[0] - line56_mid_pts[0]);
        
        if(vp_2[0] < line56_mid_pts[0])
            vp2_cube_edge_midpt_angle = atan2(line56_mid_pts[1] - vp_2[1], line56_mid_pts[0] - vp_2[0]);
        else
            vp2_cube_edge_midpt_angle = atan2(vp_2[1] - line56_mid_pts[1], vp_2[0] - line56_mid_pts[0]);

        std::cout<<"vp1_cube_edge_midpt_angle: "<<vp1_cube_edge_midpt_angle<<std::endl;
        std::cout<<"vp2_cube_edge_midpt_angle: "<<vp2_cube_edge_midpt_angle<<std::endl;
        std::cout<<"line56_angl: "<<line56_angl<<std::endl;
        std::cout<<"line67_angl: "<<line67_angl<<std::endl;

        double vp1_corr_cube_edge_angle, vp2_corr_cube_edge_angle;
        if((vp1_cube_edge_midpt_angle - line56_angl) < (vp2_cube_edge_midpt_angle - line56_angl))
        {
            vp1_corr_cube_edge_angle = line56_angl;
            vp2_corr_cube_edge_angle = line67_angl;
        }
        else
        {
            vp2_corr_cube_edge_angle = line56_angl;
            vp1_corr_cube_edge_angle = line67_angl;
        }

        //if(vcuboid[i]->rotY < 1.57)
        //total_angle_diff = abs(line1_angl - vp1_mean_angle) + abs(line2_angl - vp2_mean_angle);
        // else
        // double line2_diff = std::min(abs(line2_angl - vp1_mean_angle), M_PI - abs(line2_angl - vp1_mean_angle));
        // double line1_diff = std::min(abs(line1_angl - vp2_mean_angle), M_PI - abs(line1_angl - vp2_mean_angle));
        // double line2_diff2 = std::min(abs(line2_angl - vp2_mean_angle), M_PI - abs(line2_angl - vp2_mean_angle));
        // double line1_diff2 = std::min(abs(line1_angl - vp1_mean_angle), M_PI - abs(line1_angl - vp1_mean_angle));
        total_angle_diff = std::abs(vp1_corr_cube_edge_angle - vp1_mean_angle) + std::abs(vp2_corr_cube_edge_angle - vp2_mean_angle);
        // //total_angle_diff = abs(line2_angl - vp1_mean_angle) + abs(line1_angl - vp2_mean_angle);
        // double total_angle_diff22;
        // total_angle_diff22 = line1_diff2 + line2_diff2;
        std::cout<<"yaw : "<<vcuboid[i]->rotY <<std::endl;
        std::cout<<"!!!!!!angle_diff: "<<total_angle_diff<<std::endl;
        // std::cout<<"!!!!!!angle_diff22: "<<total_angle_diff22<<std::endl;
        //    double final_diff = total_angle_diff * sum_dist;
        //    std::cout<<"-------------final_diff: "<<final_diff<<std::endl;
        cv::imshow("cube_image_for_angle_diff",rgb_img);
        cv::waitKey(0);

        //cv::waitKey(0);

    } //end of different sample cube

    auto smallest = std::min_element(std::begin(dist_error), std::end(dist_error));
    std::cout << "min element is " << *smallest<< " at position " <<std::distance(std::begin(dist_error), smallest) << std::endl;
    int position = std::distance(std::begin(dist_error), smallest);
    swap(vcuboid[0], vcuboid[position]);     //change best cube to first place 
    std::cout<<"best_sample_yaw: "<<vcuboid[0]->rotY<<std::endl;
    //cv::waitKey(0);








}//end of function
