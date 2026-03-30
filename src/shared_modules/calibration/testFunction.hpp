#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core.hpp>



using namespace std;


//�ĵ����ƽ��
void FP_analytic(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, double &A, double &B, double &C, double &D);

void FP_analytic2(vector<cv::Point3f> pts, cv::Mat &plane);

//����ֱ��ͬƽ�潻��
int PLP_analytic(cv::Mat plane, cv::Mat pp1, cv::Mat pp2, cv::Point3f &plp);

//RT->ƽ��
void FP_analytic3(cv::Mat R, cv::Mat T, cv::Mat &plane);