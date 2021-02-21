#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <cmath>

//ours
#include <line_lbd/line_lbd_allclass.h>
#include <line_lbd/line_descriptor.hpp>


#include "main.h"
#include "filter_cube.h"

using namespace std;
using namespace Eigen;

#define PI 3.1415926
#define deg2rad(x) ((x)*PI/180)

template <class T>
bool read_all_number_txt(const std::string txt_file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &read_number_mat)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    std::ifstream filetxt(txt_file_name.c_str());
    int row_counter = 0;
    std::string line;
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);

    while (getline(filetxt, line))
    {
        T t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();

    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows

    return true;
}

void read_yaml(const string &path_to_yaml, Eigen::Matrix3d & Kalib, float& depth_scale)
{
    // string strSettingPath = path_to_dataset + "/ICL.yaml";
    cv::FileStorage fSettings(path_to_yaml, cv::FileStorage::READ);

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Load camera parameters from settings file

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    Kalib<< fx,  0,  cx,
            0,  fy,  cy,
            0,  0,   1;
        depth_scale = fSettings["DepthMapFactor"];

}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

template <class T>
void linespace(T starting, T ending, T step, std::vector<T> &res)
{
    res.reserve((ending - starting) / step + 2);
    while (starting <= ending)
    {
        res.push_back(starting);
        starting += step; // TODO could recode to better handle rounding errors
        if (res.size() > 1000)
        {
            std::cout << "Linespace too large size!!!!" << std::endl;
            break;
        }
    }
}

void cuboid::print_cuboid()
{
    std::cout << "printing cuboids info...." << std::endl;
    std::cout << "pos   " << pos.transpose() << std::endl;
    std::cout << "scale   " << scale.transpose() << std::endl;
    std::cout << "rotY   " << rotY << std::endl;
    std::cout << "box_config_type   " << box_config_type.transpose() << std::endl;
    std::cout << "box_corners_2d \n"
              << box_corners_2d << std::endl;
    std::cout << "box_corners_3d_world \n"
              << box_corners_3d_world << std::endl;
}

bool read_obj_detection_txt(const std::string txt_file_name, Eigen::MatrixXd &read_number_mat, std::vector<std::string> &all_strings)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    all_strings.clear(); //清除上一步可能残留的类别字符
    std::ifstream filetxt(txt_file_name.c_str());
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);
    int row_counter = 0;
    std::string line;
    while (getline(filetxt, line))
    {
        double t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            std::string classname;
            ss >> classname;
            all_strings.push_back(classname);

            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();
    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows
    return true;
}

bool wether_same(constraint &left, constraint &top)
{
    if(left.x == top.x && left.y == top.y && left.z == top.z )
        return true;
    else
        return false;
}

float calc_theta_ray(cv::Mat rgb_img, Eigen::Vector4d box_2d, Eigen::Matrix<double, 3, 4>proj_matrix )
{
    float width = rgb_img.cols;
    float x = (box_2d[0] + box_2d[2]) / 2;
    float u_distance = x - width/2;
    //std::cout<<"u_distance : "<< u_distance <<std::endl;
    int mult = 1;
    if(u_distance < 0)
        mult = -1;
    u_distance = abs(u_distance);
    //std::cout<<"u_distance : "<< u_distance <<std::endl;
    float focal_length = proj_matrix(0,0);
    //std::cout<<"focal length : "<< focal_length <<std::endl;
    float rot_ray = atan(u_distance / focal_length);
    //std::cout<<"rot_ray "<<rot_ray<<std::endl;
    rot_ray= rot_ray * mult;
    float angle = rot_ray;

    return angle;

}

void plot_3d_box_with_label(cv::Mat &img, Eigen::Matrix<double, 3, 4> cam_to_img, double rot_y,
                            Eigen::Vector3d dims, Eigen::Vector3d location)
{
    std::vector<Eigen::Vector2i> box_2d; //cube corner in image coordinate [u,v]
    std::vector<int> wid_leng {1, -1};
    std::vector<int> height {0,1};
    for(auto i : wid_leng)
        for(auto j : wid_leng)
            for(auto k : height)
            {
                Eigen::Vector4d point(4);
                point[0] = location[0] + i * dims[1]/2 * cos(-rot_y + PI/2) + (j*i) * dims[2]/2 * cos(-rot_y);
                point[2] = location[2] + i * dims[1]/2 * sin(-rot_y + PI/2) + (j*i) * dims[2]/2 * sin(-rot_y);
                point[1] = location[1] - k * dims[0];
                point[3] = 1;
                Eigen::Vector3d u_v_s = cam_to_img * point;
                u_v_s = u_v_s / u_v_s[2];
                Eigen::Vector2i u_v;
                u_v[0]= (int) u_v_s[0];
                u_v[1]= (int) u_v_s[1];
                box_2d.push_back(u_v);
                //std::cout<<"pixel coordinate:"<< u_v.transpose()<<std::endl;
            }

    std::vector<cv::Point> front_mark;
    for(int j=0; j < 4; j++)
    {
        cv::line(img, cv::Point(box_2d[2*j][0], box_2d[2*j][1]),
                cv::Point(box_2d[2*j+1][0], box_2d[2*j+1][1]),
                cv::Scalar(0,0,255), 2, CV_AA,0);
        if( j == 0 || j == 3)
        {
            front_mark.push_back(cv::Point(box_2d[2*j][0], box_2d[2*j][1]));
            front_mark.push_back(cv::Point(box_2d[2*j+1][0], box_2d[2*j+1][1]));
        }
    }
    cv::line(img, front_mark[0], front_mark.back(), cv::Scalar(255,0,0), 1, CV_AA,0);
    cv::line(img, front_mark[1], front_mark[2], cv::Scalar(255,0,0), 1, CV_AA,0);
    for(int i=0; i < 8; i++)
    {
        cv::line(img, cv::Point(box_2d[i][0], box_2d[i][1]),
                 cv::Point(box_2d[(i+2)%8][0], box_2d[(i+2)%8][1]),
                 cv::Scalar(0,0,255), 2, CV_AA,0);
    }
    cv::imshow("image_with_3dbbox", img);
    cv::waitKey(0);
}

void angle_to_matrix_Y(double angle, Eigen::Matrix3d &Rotmatrix)
{
    Rotmatrix<< cos(angle), 0, sin(angle),
            0 , 1, 0,
            -sin(angle),0,cos(angle);
}

template <class T>
void rot_to_euler_zyx(const Eigen::Matrix<T, 3, 3> &R, T &roll, T &pitch, T &yaw)
{
    pitch = asin(-R(2, 0));

    if (abs(pitch - M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) + roll;
    }
    else if (abs(pitch + M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) - roll;
    }
    else
    {
        roll = atan2(R(2, 1), R(2, 2));
        yaw = atan2(R(1, 0), R(0, 0));
    }
}
template void rot_to_euler_zyx<double>(const Matrix3d &, double &, double &, double &);
template void rot_to_euler_zyx<float>(const Matrix3f &, float &, float &, float &);


template <class T>
void quat_to_euler_zyx(const Eigen::Quaternion<T> &q, T &roll, T &pitch, T &yaw)
{
    T qw = q.w();
    T qx = q.x();
    T qy = q.y();
    T qz = q.z();

    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    pitch = asin(2 * (qw * qy - qz * qx));
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}
template void quat_to_euler_zyx<double>(const Eigen::Quaterniond &, double &, double &, double &);
template void quat_to_euler_zyx<float>(const Eigen::Quaternionf &, float &, float &, float &);

template <class T>
Eigen::Matrix<T, 3, 3> euler_zyx_to_rot(const T &roll, const T &pitch, const T &yaw)
{
    T cp = cos(pitch);
    T sp = sin(pitch);
    T sr = sin(roll);
    T cr = cos(roll);
    T sy = sin(yaw);
    T cy = cos(yaw);

    Eigen::Matrix<T, 3, 3> R;  //顺序是：Rz*Ry*Rx(所以是先x后y最后z?)
    R << cp * cy, (sr * sp * cy) - (cr * sy), (cr * sp * cy) + (sr * sy),
        cp * sy, (sr * sp * sy) + (cr * cy), (cr * sp * sy) - (sr * cy),
        -sp, sr * cp, cr * cp;
    return R;
}
template Matrix3d euler_zyx_to_rot<double>(const double &, const double &, const double &);
template Matrix3f euler_zyx_to_rot<float>(const float &, const float &, const float &);

template <class T>
Eigen::Matrix<T, 3, 3> euler_xyz_to_rot(const T &roll, const T &pitch, const T &yaw)
{
    T cp = cos(pitch);
    T sp = sin(pitch);
    T sr = sin(roll);
    T cr = cos(roll);
    T sy = sin(yaw);
    T cy = cos(yaw);

    Eigen::Matrix<T, 3, 3> R;  //顺序是：Rz*Ry*Rx(所以是先x后y最后z?)
    R <<               cp * cy,                      -cp*sy,        sp,
        (sr * sp * cy)+(cr*sy),  (cr * cy) - (sr * sp * sy),    -sr*cp,
          (sr*sy) - (cr*sp*cy),  (sr * cy) + (cr * sp * sy),   cr * cp;
    return R;
}
template Matrix3d euler_xyz_to_rot<double>(const double &, const double &, const double &);
template Matrix3f euler_xyz_to_rot<float>(const float &, const float &, const float &);

void set_camera_param::set_cam_pose(const Matrix4d &transToWolrd)  //给cam_pose的各个值赋值
{
    //std::cout<<"transToWorld = \n"<<transToWolrd<<std::endl;
    cam_pose.transToWolrd = transToWolrd;
    cam_pose.rotationToWorld = transToWolrd.topLeftCorner<3, 3>();
    Vector3d euler_angles;
    quat_to_euler_zyx(Quaterniond(cam_pose.rotationToWorld), euler_angles(0), euler_angles(1), euler_angles(2));//roll pitch yaw
    cam_pose.euler_angle = euler_angles;
    cam_pose.euler_angle[0] = cam_pose.euler_angle[0] - 1.57078;  //rotate along x axis -90 degree
    cam_pose.invR = cam_pose.rotationToWorld.inverse();
    cam_pose.projectionMatrix = cam_pose.Kalib * transToWolrd.inverse().topRows<3>(); // project world coordinate to camera
    cam_pose.KinvR = cam_pose.Kalib * cam_pose.invR;
    cam_pose.camera_yaw = cam_pose.euler_angle(2);//yaw绕z轴，roll绕x轴，pitch绕y轴
    //TODO relative measure? not good... then need to change transToWolrd.
}

void set_camera_param::set_calibration(const Matrix3d &Kalib)
{
    cam_pose.Kalib = Kalib;
    cam_pose.invK = Kalib.inverse();
}

MatrixXd compute3D_BoxCorner_in_camera(Eigen::Vector3d& dimension, Eigen::Vector3d& location, Eigen::Matrix3d& local_rot_mat)
{
    MatrixXd corners_body(3, 8);
    corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    1, 1, 1, 1, -1, -1, -1, -1;
    Matrix3d scale_mat = dimension.asDiagonal();
    Matrix3d rot;
    rot = local_rot_mat;
    // rot << cos(ry), -sin(ry), 0,
    //     sin(ry), cos(ry), 0,
    //     0, 0, 1;                          // rotation around z (up), world coordinate
    // rot << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry); // rotation around y (up), camera coordinate

    MatrixXd corners_without_center = rot * scale_mat * corners_body;
    // std::cout << "dimension\n" << scale_mat * corners_body << std::endl;
    // std::cout << "rot\n" << rot << std::endl;
    // std::cout << "corners_without_center\n" << corners_without_center << std::endl;

    MatrixXd corners_3d(3, 8);
    for (size_t i = 0; i < 8; i++)
    {
      corners_3d(0,i) = corners_without_center(0,i) + location(0);
      corners_3d(1,i) = corners_without_center(1,i) + location(1);
      corners_3d(2,i) = corners_without_center(2,i) + location(2);
    }
    return corners_3d;
}

Eigen::MatrixXd project_camera_points_to_2d(Eigen::MatrixXd& points3d_3x8, Matrix3d& Kalib)
{
  Eigen::MatrixXd corners_2d(3,8);
  corners_2d = Kalib *  points3d_3x8;
  for (size_t i = 0; i < corners_2d.cols(); i++)
  {
      corners_2d(0,i) = corners_2d(0,i) /corners_2d(2,i);
      corners_2d(1,i) = corners_2d(1,i) /corners_2d(2,i);
      corners_2d(2,i) = corners_2d(2,i) /corners_2d(2,i);
  }
  Eigen::MatrixXd corners_2d_return(2,8);
  corners_2d_return = corners_2d.topRows(2);
  return corners_2d_return;
}

Eigen::Vector3d calc_location(Eigen::Vector3d dimension,Eigen::Matrix3d obj_rot_cam, Eigen::Matrix<double, 3, 4> proj_matrix,
                              Eigen::Vector4d box_2d, double alpha, double theta_ray)
{
    double orient = alpha + theta_ray;
    //std::cout<<"orient: "<<orient<<std::endl;
    Eigen::Matrix3d R;
    R = obj_rot_cam;    //------------new method----------------
//    R<< cos(orient), 0, sin(orient),
//        0 , 1, 0,
//        -sin(orient),0,cos(orient);
//    std::cout<<"R = "<< R << std::endl;
    Eigen::Vector3d euler_angl;
    euler_angl = R.eulerAngles(0,1,2);
    //std::cout<<"euler: \n"<<euler_angl.transpose()<<std::endl;
    double new_alpha = euler_angl[2];
    //std::cout<<"yaw: "<<new_alpha<<endl;
//    float Xmin = box_2d[0];           //in Cube SLAM the dimension is already half!!
//    float Ymax = box_2d[1];
//    float Xmax = box_2d[2];
//    float Ymin = box_2d[3];


    float dx = dimension(0) / 2.0;         //Attention: make sure the dimension(L,W,H) correnspend to whitch axis
    float dy = dimension(1) / 2.0;
    float dz = dimension(2) / 2.0;
    //std::cout<<"dx: "<<dx<<" dy: "<<dy<<" dz: "<<dz<<std::endl;
    int left_mult, right_mult, left_switch, right_switch;
    if( new_alpha > -1.36 && new_alpha < 1.428){
        left_mult = -1;
        right_mult = 1;
    }
    if(new_alpha > 1.428 && new_alpha < 1.68){
        left_mult = -1;
        right_mult = -1;
    }
    if(new_alpha > 1.68){
        left_mult = 1;
        right_mult = -1;
    }
    if(new_alpha < -1.6){
        left_mult = 1;
        right_mult = -1;
    }
    if(new_alpha > -1.6 && new_alpha < -1.36){
        left_mult = 1;
        right_mult = 1;
    }

    if(new_alpha > -2.86 && new_alpha < -0.11)
    {
       left_switch = -1;
       right_switch = 1;
    }
    if(new_alpha > -0.11 && new_alpha < 0.29)
    {
       left_switch = -1;
       right_switch = -1;
    }
    if(new_alpha > 0.29 && new_alpha < 2.99)
    {
       left_switch = 1;
       right_switch = -1;
    }
    if(new_alpha > 2.99)
    {
       left_switch = 1;
       right_switch = 1;
    }
    if(new_alpha < -2.86)
    {
       left_switch = 1;
       right_switch = 1;
    }

    //std::cout<<"left_mult: "<<left_mult<<"  right_mult: "<<right_mult<<"  left_switch: "<<left_switch<<"  right_switch: "<<right_switch<<std::endl;
    std::vector<int> vec {-1, 1};
    std::vector<constraint> left_constraints, right_constraints, top_constraints, bottom_constrains;
    for(auto i : vec)
    {
        constraint left_constra, right_constra;     //        constraint left_constra, right_constra;
        left_constra.x = left_mult * dx;            //        left_constra.x = left_mult * dx;
        left_constra.y = left_switch * dy;          //        left_constra.y = i * dy;
        left_constra.z = i * dz;                    //        left_constra.z = -switch_mult * dz;
        left_constraints.push_back(left_constra);   //        left_constraints.push_back(left_constra);

        right_constra.x = right_mult * dx;           //        right_constra.x = right_mult * dx;
        right_constra.y = right_switch * dy;         //        right_constra.y = i * dy;
        right_constra.z = i * dz;                    //        right_constra.z = switch_mult * dz;
        right_constraints.push_back(right_constra);  //        right_constraints.push_back(right_constra);

    }
    //std::cout<< "alles ok bis hier 1"<<std::endl;
    for(auto i : vec)
        for(auto j : vec)
        {
            constraint top_constra, bottom_constra;
            top_constra.x = i * dx;
            top_constra.y = j * dy;
            top_constra.z = dz;
            top_constraints.push_back(top_constra);

            bottom_constra.x = i * dx;
            bottom_constra.y = j * dy;
            bottom_constra.z = -dz;
            bottom_constrains.push_back(bottom_constra);
        }
    //left, top, right, bottom
    std::vector<constraint> four_constraint(4);
    std::vector<std::vector<constraint>> final_constraint;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 4; j++)
            for(int k = 0; k < 2; k++)
                for(int l = 0; l < 4; l++)
                {
                    four_constraint[0] = left_constraints[i];
                    four_constraint[1] = top_constraints[j];
                    four_constraint[2] = right_constraints[k];
                    four_constraint[3] = bottom_constrains[l];

                    final_constraint.push_back(four_constraint);
                }
    //--------------new method-----------------
    float best_error = 1e09;
    int num = 0;
    Eigen::Vector3d best_loca;
    //std::cout<<"size of final_constraints: "<<final_constraint.size()<<std::endl;
    //std::cout<< "alles ok bis hier 2"<<std::endl;  //for(int i = 0; i < 1; i++)
    for(int i = 0; i < final_constraint.size(); i++)
    {
        constraint Xa, Xb, Xc, Xd;
        Xa = final_constraint[i][0];    //left
        Xb = final_constraint[i][1];    //top
        Xc = final_constraint[i][2];    //right
        Xd = final_constraint[i][3];    //bottom

        if( wether_same(Xa,Xc) || wether_same(Xb,Xd))
            continue;
        //std::cout<< "alles ok bis hier 3"<<std::endl;
        Eigen::Vector3d point_left, point_top, point_right, point_bottom;
        point_left<< Xa.x, Xa.y, Xa.z;
        point_top<<  Xb.x, Xb.y, Xb.z;
        point_right<< Xc.x, Xc.y, Xc.z;
        point_bottom<< Xd.x, Xd.y, Xd.z;
        //std::cout<< "point left: "<< point_left<<std::endl;
        std::vector<Eigen::Vector3d> point(4);
        point[0] = point_left;
        point[1] = point_top;
        point[2] = point_right;
        point[3] = point_bottom;
        //std::cout<<"size of point: "<<point.size()<<std::endl;
        //std::cout<< "alles ok bis hier 5"<<std::endl;
        //std::cout<<"left: "<<point_left.transpose()<<" top:"<<point_top.transpose()<<" right:"<<point_right.transpose()<<" bottom:"<<point_bottom.transpose()<<std::endl;
        Eigen::Matrix<double, 4, 3> A;
        Eigen::Vector4d b;
        std::vector<int> index{0,1,0,1};
        for(int j = 0; j < index.size(); j++)
        {
            Eigen::Matrix4d Pre_M;
            Pre_M.setIdentity();
            Pre_M.col(3).head(3) = R * point[j];
            Eigen::Matrix<double, 3, 4> M = proj_matrix * Pre_M;    //Ma, Mb, Mc, Md (3x4)

            A.row(j) = M.block(index[j],0,1,3) - box_2d[j] * M.block(2,0,1,3);
            b[j] = box_2d[j] * M(2,3) - M(index[j],3);


        }
//        cout<<"num: "<<num<<endl;
//        std::cout<<"A: "<<endl<<A<<endl;
//        std::cout<<"b: "<<endl<<b<<endl;

        Eigen::Vector3d x_jacobiSvd;
        x_jacobiSvd = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b); //SVD method
        //x_jacobiSvd = (A.transpose()*A).inverse()*A.transpose()*b;
        //std::cout<<"SVD proposals "<< x_jacobiSvd.transpose()<<std::endl;
        //error
        Eigen::VectorXd error_vec =b - A * x_jacobiSvd;
        double error = error_vec.norm();

        bool included = false;
        Eigen::Matrix3d Kalib = proj_matrix.block(0,0,3,3);
        //std::cout<<"Kalib: \n"<<Kalib<<std::endl;
        Eigen::Vector3d dim;
        dim<<dx,dy,dz;
        //std::cout<<"dimension: \n"<<dim.transpose()<<std::endl;
        Eigen::MatrixXd points3d_camera_3x8 = compute3D_BoxCorner_in_camera(dim, x_jacobiSvd, obj_rot_cam); // same result.
        Eigen::MatrixXd points2d_camera_2x8 = project_camera_points_to_2d(points3d_camera_3x8, Kalib);
        Eigen::Vector4d bbox_new;
        bbox_new << points2d_camera_2x8.row(0).minCoeff(), points2d_camera_2x8.row(1).minCoeff(),
                    points2d_camera_2x8.row(0).maxCoeff(), points2d_camera_2x8.row(1).maxCoeff();
        Eigen::Vector4d bbox_delta = bbox_new - box_2d;
        double delta = 20;
        if( bbox_delta(0) > -delta && bbox_delta(1) > -delta && // xmin, ymin
            bbox_delta(2) < delta && bbox_delta(3) <   delta ) // xmax, ymax
            included = true;
        //included = true;
        if( error < best_error && included)
        {
            best_error = error;
            best_loca = x_jacobiSvd;
            //std::cout<<"new_better_constraint:"<<"left: "<<point_left.transpose()<<" top:"<<point_top.transpose()<<" right:"<<point_right.transpose()<<" bottom:"<<point_bottom.transpose()<<std::endl;
            //std::cout<<"bbox_delta: "<<bbox_delta.transpose()<<std::endl;
        }
        num++;
    }
        //best_loca[1] += dimension[0] / 2;      //for TUM in my opinion no need this
        return best_loca;
}

Matrix4d similarityTransformation(const cuboid &cube_obj)
{
    Matrix3d rot;
    rot << cos(cube_obj.rotY), -sin(cube_obj.rotY), 0,
        sin(cube_obj.rotY), cos(cube_obj.rotY), 0,
        0, 0, 1;
    Matrix3d scale_mat = cube_obj.scale.asDiagonal();

    Matrix4d res = Matrix4d::Identity();
    res.topLeftCorner<3, 3>() = rot * scale_mat;
    res.col(3).head(3) = cube_obj.pos;
    return res;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_homo_out;
    int raw_rows = pts_in.rows();
    int raw_cols = pts_in.cols();

    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
        Matrix<T, 1, Dynamic>::Ones(raw_cols);
    return pts_homo_out;
}
template MatrixXd real_to_homo_coord<double>(const MatrixXd &);
template MatrixXf real_to_homo_coord<float>(const MatrixXf &);

template <class T>
void real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_out)
{
    int raw_rows = pts_in.rows();
    int raw_cols = pts_in.cols();

    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
        Matrix<T, 1, Dynamic>::Ones(raw_cols);
}
template void real_to_homo_coord<double>(const MatrixXd &, MatrixXd &);
template void real_to_homo_coord<float>(const MatrixXf &, MatrixXf &);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_out(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N

    return pts_out;
}
template MatrixXd homo_to_real_coord<double>(const MatrixXd &);
template MatrixXf homo_to_real_coord<float>(const MatrixXf &);

template <class T>
void homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_out)
{
    pts_out.resize(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N
}
template void homo_to_real_coord<double>(const MatrixXd &, MatrixXd &);
template void homo_to_real_coord<float>(const MatrixXf &, MatrixXf &);

template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> homo_to_real_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> pt_out;
    if (pts_homo_in.rows() == 4)
        pt_out = pts_homo_in.head(3) / pts_homo_in(3);
    else if (pts_homo_in.rows() == 3)
        pt_out = pts_homo_in.head(2) / pts_homo_in(2);

    return pt_out;
}
template VectorXd homo_to_real_coord_vec<double>(const VectorXd &);
template VectorXf homo_to_real_coord_vec<float>(const VectorXf &);

Matrix3Xd compute3D_BoxCorner(const cuboid &cube_obj)
{
    MatrixXd corners_body;
    corners_body.resize(3, 8);     //at first are 4 points at bottom(5,6,7,8), then top points(1,2,3,4)
    corners_body << 1,  1, -1, -1, 1,  1, -1, -1,
                    1, -1, -1,  1, 1, -1, -1,  1,
                   -1, -1, -1, -1, 1,  1,  1,  1;
    MatrixXd corners_world = homo_to_real_coord<double>(similarityTransformation(cube_obj) * real_to_homo_coord<double>(corners_body));
    return corners_world;
}

//function about edge
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat &rgb_img, cv::Mat &output_img, MatrixXd &all_lines, const cv::Scalar &color)
{
    output_img = rgb_img.clone();
    for (int i = 0; i < all_lines.rows(); i++)
        cv::line(output_img, cv::Point(all_lines(i, 0), all_lines(i, 1)), cv::Point(all_lines(i, 2), all_lines(i, 3)), cv::Scalar(255, 0, 0), 2, 8, 0);
}

// make sure edges start from left to right
void align_left_right_edges(MatrixXd &all_lines)
{
    for (int line_id = 0; line_id < all_lines.rows(); line_id++)
    {
        if (all_lines(line_id, 2) < all_lines(line_id, 0))
        {
            Vector2d temp = all_lines.row(line_id).tail<2>();
            all_lines.row(line_id).tail<2>() = all_lines.row(line_id).head<2>();
            all_lines.row(line_id).head<2>() = temp;
        }
    }
}

Eigen::MatrixXi compute2D_BoxCorner(const cuboid &cube_obj, const Eigen::Matrix<double, 3, 4>& projectMatrix)
{
    Eigen::MatrixXi corners_2d_return(2, 8);    // same type with cube.box_corners_2d
    Eigen::Matrix<double, 3, 8> corners_2d;
    Eigen::Matrix<double, 4, 8> corners_3d;
    corners_3d.block(0,0,3,8) = cube_obj.box_corners_3d_world;
    for (size_t i = 0; i < corners_3d.cols(); i++)
        corners_3d(3,i) = 1.0;

    corners_2d = projectMatrix  * corners_3d ;
    for (size_t i = 0; i < corners_2d.cols(); i++)
    {
        corners_2d(0,i) = corners_2d(0,i) /corners_2d(2,i);
        corners_2d(1,i) = corners_2d(1,i) /corners_2d(2,i);
        corners_2d(2,i) = corners_2d(2,i) /corners_2d(2,i);
    }
    corners_2d_return = corners_2d.topRows(2).cast <int> ();
    return corners_2d_return;
}

void plot_image_with_cuboid_new(cv::Mat &plot_img, const cuboid *cube_obj)
 {
    Eigen::MatrixXi edge_order(2, 12); // normally, the corners are in order and edges are in order
    edge_order << 1, 2, 3, 4, 1, 3, 5, 7, 1, 2, 5, 6,  // z axis: 15, 26, ... x axis: 12, 34, ... y axis: 14, 23, ...
                  5, 6, 7, 8, 2, 4, 6, 8, 4, 3, 8, 7;
    Eigen::Matrix2Xi box_corners_2d = cube_obj->box_corners_2d; //2*8
    for (int edge_id = 0; edge_id < edge_order.cols(); edge_id++)
    {
        cv::Point pt0 = cv::Point (box_corners_2d(0, edge_order(0, edge_id)-1), box_corners_2d(1, edge_order(0, edge_id)-1));
        cv::Point pt1 = cv::Point (box_corners_2d(0, edge_order(1, edge_id)-1), box_corners_2d(1, edge_order(1, edge_id)-1));
        if(edge_id < 4)
            cv::line(plot_img, pt0, pt1, cv::Scalar(0, 255, 0), 2, CV_AA, 0);
        if(edge_id >3 && edge_id < 8)
            cv::line(plot_img, pt0, pt1, cv::Scalar(255, 0, 0), 2, CV_AA, 0);
        if(edge_id > 7)
            cv::line(plot_img, pt0, pt1, cv::Scalar(0, 0, 255), 2, CV_AA, 0);
    }
    // for (size_t i = 0; i < cube_obj->box_corners_2d.cols(); i++)
    // {
    //     cv::circle(plot_img, cv::Point(cube_obj->box_corners_2d(0,i),cube_obj->box_corners_2d(1,i)),
    //                 i,cv::Scalar(0, 255, 0),1,8, 0);
    // }
 }

//Benchun's location_function
void calculate_location(Eigen::Vector3d& dimension, Eigen::Matrix<double, 3, 4>& proj_matrix,
                        Eigen::Vector4d& bbox, double& alpha, double& theta_ray, Eigen::Vector3d& location)
{
    double orient = alpha + theta_ray; // global_angle = local + theta_ray
    Eigen::Matrix3d R_Matrix;
    R_Matrix << cos(orient), 0, sin(orient), 0, 1, 0, -sin(orient), 0, cos(orient);
    // std::cout << "alpha in degree: " << alpha/M_PI*180 << std::endl;
    // std::cout << "orient in degree: " << orient/M_PI*180 << std::endl;
    // R_Matrix << cos(orient), -sin(orient), 0,
    //     sin(orient), cos(orient), 0,
    //     0, 0, 1;
    // kitti dataset is different, the order is height, width, length
    // double dx = dimension(2) / 2.0;
    // double dy = dimension(0) / 2.0;
    // double dz = dimension(1) / 2.0;
    double dx = dimension(0) ;
    double dy = dimension(2) ;
    double dz = dimension(1) ;

    double left_mult = -1;
    double right_mult = 1;
    double switch_mult = -1;

    // # below is very much based on trial and error
    // # based on the relative angle, a different configuration occurs
    // # negative is back of car, positive is front
    left_mult = 1;
    right_mult = -1;

    // # about straight on but opposite way
    if (alpha < 92.0/180.0*M_PI && alpha > 88.0/180.0*M_PI)
    {   left_mult = 1;
        right_mult = 1;
    }
    // # about straight on and same way
    else if (alpha < -88.0/180.0*M_PI && alpha > -92.0/180.0*M_PI)
    {    left_mult = -1;
        right_mult = -1;
    }
    // # this works but doesnt make much sense
    else if (alpha < 90/180.0*M_PI && alpha > -90.0/180.0*M_PI)
    {    left_mult = -1;
        right_mult = 1;
    }
    // # if the car is facing the oppositeway, switch left and right
    switch_mult = -1;
    if (alpha > 0)
        switch_mult = 1;

    Eigen::MatrixXd left_constraints(2,3);
    Eigen::MatrixXd right_constraints(2,3);
    Eigen::MatrixXd top_constraints(4,3);
    Eigen::MatrixXd bottom_constraints(4,3);
    Eigen::Vector2i negetive_positive = Eigen::Vector2i(-1,1);
    for (size_t i = 0; i < 2; i++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      left_constraints.row(i) = Eigen::Vector3d(left_mult*dx, k*dy, -switch_mult*dz).transpose();
      right_constraints.row(i) = Eigen::Vector3d(right_mult*dx, k*dy, switch_mult*dz).transpose();
    }

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
    {
      int k = negetive_positive(i); // transfrom from (0,1) to (-1, 1)
      int h = negetive_positive(j); // transfrom from (0,1) to (-1, 1)
      top_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, -dy, h*dz).transpose();
      bottom_constraints.row(i*2+j) = Eigen::Vector3d(k*dx, dy, h*dz).transpose();
    }

    // std::cout << "left_constraints: \n" << left_constraints<< std::endl;
    // std::cout << "right_constraints: \n" << right_constraints<< std::endl;
    // std::cout << "top_constraints: \n" << top_constraints<< std::endl;
    // std::cout << "bottom_constraints: \n" << bottom_constraints<< std::endl;

    Eigen::MatrixXd all_constraints(64,12);
    size_t cnt_rows=0;
    for (size_t l_id = 0; l_id < 2; l_id++)
      for (size_t r_id = 0; r_id < 2; r_id++)
        for (size_t t_id = 0; t_id < 4; t_id++)
          for (size_t b_id = 0; b_id < 4; b_id++)
          {
            all_constraints.block(cnt_rows,0,1,3) = left_constraints.row(l_id);
            all_constraints.block(cnt_rows,3,1,3) = top_constraints.row(t_id);
            all_constraints.block(cnt_rows,6,1,3) = right_constraints.row(r_id);
            all_constraints.block(cnt_rows,9,1,3) = bottom_constraints.row(b_id);
            cnt_rows++;
          }
    // how to filter same rows
    // // True if equal
    // bool r = a.isApprox(b, 1e-5);
    Eigen::Vector3d best_x;
    double best_error=1e9;
    for (size_t constraints_id = 0; constraints_id < 64; constraints_id++)
    {
      Eigen::MatrixXd A(4,3);
      Eigen::VectorXd b(4);
      for (size_t i = 0; i < 4; i++)
      {
        Eigen::Vector3d constrait = all_constraints.block(constraints_id,i*3,1,3).transpose(); // left,right,top,or bottom
        Eigen::Vector3d RX_vec = R_Matrix * constrait;
        Eigen::Matrix4d RX_Matrix;
        RX_Matrix.setIdentity();
        RX_Matrix.col(3).head(3) = RX_vec;

        Eigen::MatrixXd M(3,4);
        M = proj_matrix * RX_Matrix;

        if(i%2==0) // i=0,1 left and right, corresponding to xmin,xmax
        {
          A.row(i) = M.block(0,0,1,3) - bbox(i) * M.block(2,0,1,3); // xmin, xmax
          b(i) = bbox(i) * M(2,3) - M(0,3);
        }
        else
        {
          A.row(i) = M.block(1,0,1,3) - bbox(i) * M.block(2,0,1,3); // ymin, ymax
          b(i) = bbox(i) * M(2,3) - M(1,3);
        }
        // std::cout << "The M is:\n" << M << std::endl;

      }
      // test different solution for Ax=b
      // std::cout << "The A is:\n" << A << std::endl;
      // std::cout << "The b is:\n" << b.transpose() << std::endl;
      // Vector3d x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
      // double error = (b-A*x).norm();
      // Vector3d x2 = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
      // Vector3d x1 = A.colPivHouseholderQr().solve(b);
      // // Vector3f x2 = A_matrix.llt().solve(b_matrix);
      // // Vector3f x3 = A_matrix.ldlt().solve(b_matrix);
      // std::cout << "The solution is:" << x.transpose() << " error:" << (b-A*x).norm() << std::endl;
      // std::cout << "The solution is:" << x1.transpose() << " error:" << (b-A*x1).norm() << std::endl;
      // std::cout << "The solution is:" << x2.transpose() << " error:" << (b-A*x2).norm() << std::endl;

      Vector3d x = (A.transpose()*A).inverse()*A.transpose()*b;
      double error = (b-A*x).norm();
      // std::cout << constraints_id << " solution is:" << x.transpose() << " error:" << (b-A*x).norm() << std::endl;

      if(error < best_error)
      {
        best_x = x;
        best_error = error;
      }
    }

    location = best_x;
    //location(1) += dimension(1) ; // bring the KITTI center up to the middle of the object
    // std::cout << "best solution is:" << location.transpose() << " error:" << best_error << std::endl;
}

int main(int argc, char **argv)
{

    string dataset_path = argv[1];

    // Load camera parameters from settings file
    string strSettingPath = dataset_path + "TUM3.yaml";
    Eigen::Matrix3d kalib;
    float depth_map_scaling;
    read_yaml( strSettingPath, kalib , depth_map_scaling);

    string truth_cam_pose_path = dataset_path + "truth_cam_poses.txt"; //time, x, y, z, qx, qy, qz, qw
    Eigen::MatrixXd truth_frame_poses(60,8);
        if (!read_all_number_txt(truth_cam_pose_path,truth_frame_poses))
            return -1;

    string truth_cuboid_path = dataset_path + "cabinet_truth.txt";// x, y, z, yaw, length, width, height
    Eigen::MatrixXd truth_cuboid_list(1,7);
        if (!read_all_number_txt(truth_cuboid_path, truth_cuboid_list))
            return -1;

    bool whether_plot_detail_images = false;
    bool whether_sample_cam_roll_pitch = false;
    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = dataset_path + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    vector<string> vstrBboxFilenames;
    string strFile_yolo = dataset_path + "/yolov3_bbox.txt";
    LoadImages(strFile_yolo, vstrBboxFilenames, vTimestamps);

   //-------for multiple objects in one image
//    std::vector<string> all_class;
//    Eigen::MatrixXd line(1,14);
//    if(!read_obj_detection_txt(label_path, line, all_class))
//        return -1;
    std::vector<cuboid *> Veccuboid;        //for all final tum esti_location
    int total_frame_number = truth_frame_poses.rows();
    //std::cout<<"totally has: "<<total_frame_number<<" frames"<<std::endl;
    //total_frame_number = 1;
    for(int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
        char frame_index_c[256];
        sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
        std::cout << "frame_index: " << frame_index << std::endl;

        //load image
        cv::Mat rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);
        cv::Mat rgb_img_raw = rgb_img;
        //read cleaned yolo 2d object detection
        Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
        raw_2d_objs.setZero();
        if (!read_all_number_txt(dataset_path + "/" + vstrBboxFilenames[frame_index], raw_2d_objs))
        return -1;

        //camera truth pose
        Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>();
        //std::cout<<"cam_pose_Twc: "<< cam_pose_Twc << std::endl;
        Matrix<double,4,4> transToWolrd;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        std::cout << "transToWolrd_orignal: \n" << transToWolrd << std::endl;

        //set camera param
        set_camera_param current_camera_param;
        current_camera_param.set_calibration(kalib);
        current_camera_param.set_cam_pose(transToWolrd);
        //std::cout<<"transToworld: \n"<<transToWolrd<<std::endl;
        std::cout<<"transtoworld_in class: \n"<<current_camera_param.cam_pose.transToWolrd<<std::endl;
        //edge detection
        line_lbd_detect line_lbd_obj;
        line_lbd_obj.use_LSD = true;
        line_lbd_obj.line_length_thres = 15;  // to remove short edges
        cv::Mat all_lines_mat;
        line_lbd_obj.detect_filter_lines(rgb_img, all_lines_mat);
        Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
        for (int rr=0;rr<all_lines_mat.rows;rr++)
          for (int cc=0;cc<4;cc++)
              all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);

        if (whether_plot_detail_images)
        {
            cv::Mat output_img;
            plot_image_with_edges(rgb_img, output_img, all_lines_raw, cv::Scalar(255, 0, 0));
            cv::imshow("Raw detected Edges", output_img);
            cv::waitKey(0);
        }

        //std::cout<<"col size of line: "<<raw_2d_objs.rows()<<std::endl;
        for(int i = 0; i < raw_2d_objs.rows(); i++)
        {
//            string current_class = all_class[i];      //for multiple objects
//            std::cout<<"current class is "<< current_class<<std::endl;

            //2d-bbox
            Eigen::Vector4d toplef_rigbot;    // top_left and right_bottom point [x1y1,x2y2]
            toplef_rigbot<< raw_2d_objs(i,0), raw_2d_objs(i,1), raw_2d_objs(i,0) + raw_2d_objs(i,2), raw_2d_objs(i,1) + raw_2d_objs(i,3);
            //std::cout<<"2d_bbox is: "<<toplef_rigbot.transpose()<<std::endl;

            Eigen::Vector3d dimension, location;          //groundtruth location and dimension
            dimension = 2 * truth_cuboid_list.block(0,4,1,3).transpose();  //length, width, height  note:in tum cabinet the dimension is half value
            location = truth_cuboid_list.block(0,0,1,3).transpose();    //x,y,z
            //std::cout<<"dimension is: "<<dimension.transpose() <<std::endl <<"groundtruth_location is: "<<location.transpose() <<std::endl;
            //proj_matrix baesd on camera as orignal coodinate
            Eigen::Matrix<double, 3, 4> proj_matrix;
            Eigen::Matrix<double, 3, 4> R_t;
            R_t<<1,0,0,0,
                  0,1,0,0,
                    0,0,1,0;
            proj_matrix = kalib * R_t;
//            proj_matrix = Kalib * transToWolrd.inverse().topRows<3>();         //our thought at first, should be wrong

            double theta_ray = calc_theta_ray(rgb_img, toplef_rigbot, proj_matrix);
            std::cout<<"theta_ray: "<< theta_ray << std::endl;
            //std::cout<<"proj_matrix: "<< proj_matrix << std::endl;
            std::cout<<"euler_from_set_function: "<<current_camera_param.cam_pose.euler_angle.transpose()<<std::endl;
            
            Eigen::Vector3d best_location;
            double final_rotY;
            double best_error = 1e09;
            //sample camera roll and pitch
            Eigen::Vector3d cam_euler;
            Eigen::Matrix3d trans;
            trans = transToWolrd.block(0,0,3,3); //roll pitch yaw in world
            cam_euler = trans.eulerAngles(0,1,2);
            std::cout<<"euler by myself: "<<cam_euler.transpose()<<std::endl;
            Eigen::Matrix3d transToWolrd_test = euler_xyz_to_rot<double>(cam_euler[0],cam_euler[1],cam_euler[2]);
            std::cout<<"test_transtoword: \n"<<transToWolrd_test<<std::endl;
            std::vector<double> cam_roll_samples;
			std::vector<double> cam_pitch_samples;
			if (whether_sample_cam_roll_pitch)
			{
				linespace<double>(cam_euler(0) - 6.0 / 180.0 * M_PI, cam_euler(0) + 6.0 / 180.0 * M_PI, 3.0 / 180.0 * M_PI, cam_roll_samples);
				linespace<double>(cam_euler(1) - 6.0 / 180.0 * M_PI, cam_euler(1) + 6.0 / 180.0 * M_PI, 3.0 / 180.0 * M_PI, cam_pitch_samples);
			}
			else
			{
                cam_roll_samples.push_back(cam_euler(0));         //不sample roll和pitch的话就直接取相机当前的roll和pitch,size为1
				cam_pitch_samples.push_back(cam_euler(1));
			}
            //sample object yaw
            //std::cout<<"Camera yaw: "<<current_camera_param.cam_pose.camera_yaw<<std::endl;
            std::vector<cuboid *> vcuboid_sample;
            std::vector<double> rotation_y_samples;
            linespace<double>( 0 / 180.0 * M_PI,  180.0 / 180.0 * M_PI, 5.0 / 180.0 * M_PI, rotation_y_samples);
            std::cout<<"size of sampled roll: "<<cam_roll_samples.size()<<std::endl;
            std::cout<<"size of sampled pitch: "<<cam_pitch_samples.size()<<std::endl;

            //for(int k = 0; k < 1; k++)
            for (int cam_roll_id = 0; cam_roll_id < cam_roll_samples.size(); cam_roll_id++)
				for (int cam_pitch_id = 0; cam_pitch_id < cam_pitch_samples.size(); cam_pitch_id++)
                    for(int obj_yaw_id = 0; obj_yaw_id < rotation_y_samples.size(); obj_yaw_id++)
                    {
                        //-------------new method-----------------
                        
                        double yaw = rotation_y_samples[obj_yaw_id];
                        //yaw = 3.08234;
                        if (whether_sample_cam_roll_pitch)
						{
							Matrix4d transToWolrd_new = transToWolrd;
                            //不应该是原有位置变换，而不是直接替代原有旋转矩阵？  ->因为就是在原位置基础上采样的
                            //此处要非常注意欧拉角向旋转矩阵的变换，顺序是xyz！！！而不是zyx
                            transToWolrd_new.topLeftCorner<3, 3>() = euler_xyz_to_rot<double>(cam_roll_samples[cam_roll_id], cam_pitch_samples[cam_pitch_id], cam_euler[2]);
                            transToWolrd = transToWolrd_new;

						}
                        // std::cout<<"new_transToworld: \n"<<transToWolrd<<std::endl;
                        // std::cout<<"current yaw: "<<yaw<<std::endl;
                        // std::cout<<"current roll: "<<cam_roll_samples[cam_roll_id]<<std::endl;
                        // std::cout<<"current pitch: "<<cam_pitch_samples[cam_pitch_id]<<std::endl;

                        Eigen::Matrix4d obj_rot_world;
                        obj_rot_world<<  cos(yaw), -sin(yaw), 0, 0,
                                        sin(yaw),  cos(yaw), 0, 0,
                                                0,        0,  1, 0,
                                                0,        0,  0, 1;
                        Eigen::Matrix4d obj_rot_cam;
                        obj_rot_cam = transToWolrd.inverse() * obj_rot_world;
                        double alpha = yaw - theta_ray;
        //                std::cout<<"alpha: "<<alpha<<std::endl;
        //                std::cout<<"alpha degree is: "<< alpha * 180/M_PI<<std::endl;
                        Eigen::Matrix3d obj_rot_in_cam;
                        obj_rot_in_cam = obj_rot_cam.block(0,0,3,3);
                        Eigen::Vector3d euler_in_cam;
                        euler_in_cam = obj_rot_in_cam.eulerAngles(0,1,2);
                        //double alpha = rotation_y_samples[k] - theta_ray;

                        Eigen::Vector3d location_esti;
                        location_esti = calc_location(dimension, obj_rot_in_cam , proj_matrix, toplef_rigbot, alpha, theta_ray);
                        //std::cout<< "location_estimate_in_cam: "<< location_esti.transpose()  << std::endl;
                        Eigen::Vector4d location_esti_final;
                        location_esti_final<< location_esti[0], location_esti[1], location_esti[2], 1;
                        location_esti_final = transToWolrd * location_esti_final;
                        location_esti = location_esti_final.head(3);
                        cuboid *mittel_obj = new cuboid;
                        mittel_obj->yaw_in_cam = euler_in_cam(2);
                        mittel_obj->rotY = yaw;
                        mittel_obj->pos = location_esti;
                        mittel_obj->scale = dimension / 2;
                        mittel_obj->box_corners_3d_world = compute3D_BoxCorner(*mittel_obj);
                        Eigen::Matrix<double, 3, 4> projectionMatrix = kalib * transToWolrd.inverse().topRows<3>(); // project world coordinate to camera
                        mittel_obj->box_corners_2d = compute2D_BoxCorner(*mittel_obj, projectionMatrix);
                        //wether outside the 2dbbox, powerful constraints!
                        Eigen::Vector4d bbox_new;
                        bbox_new << mittel_obj->box_corners_2d.row(0).minCoeff(), mittel_obj->box_corners_2d.row(1).minCoeff(),
                                    mittel_obj->box_corners_2d.row(0).maxCoeff(), mittel_obj->box_corners_2d.row(1).maxCoeff();
                        Eigen::Vector4d bbox_delta = bbox_new - toplef_rigbot;
                        double delta = 10;        //really depend on dataset 
                        if( bbox_delta(0) > -delta && bbox_delta(1) > -delta && // xmin, ymin
                            bbox_delta(2) < delta && bbox_delta(3) <   delta ) // xmax, ymax
                            vcuboid_sample.push_back(mittel_obj);

                    //????????????此处有疑问，为什么加上这里vcuboid_sample大小会有变换??????????????
                    //     rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);    //reload the orignal image
                    //     plot_image_with_cuboid_new(rgb_img, mittel_obj);
                    //     cv::rectangle(rgb_img, cv::Point(toplef_rigbot[0], toplef_rigbot[1]), cv::Point(toplef_rigbot[2],toplef_rigbot[3]), cv::Scalar(255,0,0), 2 , cv::LINE_8 ,0);
                    //     cv::imshow("cube_image",rgb_img);
                    // //std::cout<<"object_yaw: "<<mittel_obj->rotY<<"; roll: "<<roll<<"; pitch: "<<pitch<<std::endl;
                    //     std::cout<< "----location_estimate_in_world: "<< location_esti.transpose()  << std::endl;
                    //     std::cout << "----true location: " << location.transpose() << std::endl;
                    //     cv::waitKey(0);
                        //for compare to groundtruth
        //                double error = (location_esti - location).norm();
        //                if(error < best_error)
        //                {
        //                    best_error = error;
        //                    best_location = location_esti;
        //                    final_rotY = rotation_y_samples[k];
        //                }

                    }   //end of sample loop of yaw(world)
               
            std::cout<<"size of vcuboid_sample: "<<vcuboid_sample.size()<<std::endl;
            //cv::waitKey(0);
                //--------------for testing sample result---------------
            // rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);    //reload the orignal image
            // for(size_t i=0; i<vcuboid_sample.size(); i++){
            //     rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);
            //     plot_image_with_cuboid_new(rgb_img,vcuboid_sample[i]);
            //     cv::rectangle(rgb_img, cv::Point(toplef_rigbot[0], toplef_rigbot[1]), cv::Point(toplef_rigbot[2],toplef_rigbot[3]), cv::Scalar(255,0,0), 2 , cv::LINE_8 ,0);
            //     cv::imshow("cube_image",rgb_img);
            //     cv::waitKey(0);
            //}
                //---------------testing end-----------------------------
                //----------------next for filter 3d_cuboid-----------scoring-----------------
            rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);    //reload the orignal image
            filter_cuboid(rgb_img, transToWolrd, vcuboid_sample, all_lines_raw, toplef_rigbot);
            // //-----------------end of filter--------------------------------------------------------
            plot_image_with_cuboid_new(rgb_img, vcuboid_sample[0]);
            cv::rectangle(rgb_img, cv::Point(toplef_rigbot[0], toplef_rigbot[1]), cv::Point(toplef_rigbot[2],toplef_rigbot[3]), cv::Scalar(255,0,0), 2 , cv::LINE_8 ,0);
            cv::imshow("cube_image",rgb_img);
            cv::waitKey(0);
            Veccuboid.push_back(vcuboid_sample[0]);
            //now the final_rotY is on the camera coordinate
//                Eigen::Matrix3d rotmatrix;
//                angle_to_matrix_Y(final_rotY,rotmatrix);
//                cuboid * final_object = new cuboid;
//                final_object->pos = best_location;
//                //final_object->rotY = 3.08234;
//                final_object->scale = dimension / 2;
//                Veccuboid.push_back(final_object);

                //-------------------sample of object yaw from----------old---------

//                double yaw_init = current_camera_param.cam_pose.camera_yaw - 90.0 / 180.0 * M_PI; //初始object yaw
//                std::vector<double> obj_yaw_samples; //采样的15个角度
//                // BRIEF linespace()函数从 a 到 b 以步长 c 产生采样的 d.
//                linespace<double>(yaw_init - 90.0/180.0*M_PI, yaw_init + 90.0/180.0*M_PI, 6.0/180.0*M_PI, obj_yaw_samples);
//                std::cout<<"sample size: "<<obj_yaw_samples.size()<<std::endl;
//                for(int i=0;i<obj_yaw_samples.size();i++)
//                std::cout<<"obj_yaw_samples: "<<obj_yaw_samples[i]<<std::endl;
//                ObjectSet all_object_cuboid;
//                //std::cout<<"size of all_object_cuboid: "<<all_object_cuboid.size()<<std::endl;

//                for(size_t i=0; i<obj_yaw_samples.size(); i++)
//                {
//                    cuboid * final_object = new cuboid;
//                    final_object->pos = best_location;
//                    //final_object->rotY = 3.08234;
//                    final_object->scale = dimension / 2;
//                    //Veccuboid.push_back(final_object); //for output txt
//                    final_object->rotY = obj_yaw_samples[i];
//                    //std::cout<<"current object yaw: "<<final_object->rotY<<std::endl;
//                    //print to 2d cuboid
//                    final_object->box_corners_3d_world = compute3D_BoxCorner(*final_object);
//                    Eigen::Matrix<double, 3, 4> projectionMatrix = kalib * transToWolrd.inverse().topRows<3>(); // project world coordinate to camera
//                    final_object->box_corners_2d = compute2D_BoxCorner(*final_object, projectionMatrix);

//                    rgb_img = cv::imread(dataset_path + "/" + vstrImageFilenames[frame_index], 1);
//                    cv::rectangle(rgb_img, cv::Point(toplef_rigbot[0]-10, toplef_rigbot[1]-10), cv::Point(toplef_rigbot[2]+20,toplef_rigbot[3]+20), cv::Scalar(255,0,0), 2 , cv::LINE_8 ,0);
//                    //plot cuboid
//                    plot_image_with_cuboid_new(rgb_img, final_object);
//                    cv::imshow("cube_image",rgb_img);
//                    std::cout<<"current object yaw: "<<final_object->rotY<<std::endl;
//                    all_object_cuboid.push_back(final_object);
//                    //std::cout<<"all_objet_cuboid: "<<all_object_cuboid.size()<<std::endl;
//                    cv::waitKey(0);

//                }
                //---------------------end of scoring---------------------------------------------------
                 //std::cout<<"size of all_object_cuboid: "<<all_object_cuboid.size()<<std::endl;
//                for(int k=0;k<all_object_cuboid.size();k++){
//                    cout<<"66666: "<<all_object_cuboid[k]->rotY<<endl;
//                }

        } //end of different object
    } //end of different frame

    std::cout<<"total number cuboid: "<<Veccuboid.size()<<std::endl;
    //print the result
    std::string saved_cuboid_folder = dataset_path + "detect_cuboids_saved.txt";
    std::ofstream print_loca;
    print_loca.open(saved_cuboid_folder.c_str());
    for(int i =0; i<Veccuboid.size(); i++)
    {
        print_loca<<i<<"  "<<Veccuboid[i]->pos[0]<<"  "<<Veccuboid[i]->pos[1]<<"  "<<Veccuboid[i]->pos[2]<<"  "<<Veccuboid[i]->rotY<<"  "
                 <<Veccuboid[i]->scale[0]<<"  "<<Veccuboid[i]->scale[1]<<"  "<<Veccuboid[i]->scale[2]<<"  "<<0<<"\n";

    }
    print_loca.close();

    return 0;
}


