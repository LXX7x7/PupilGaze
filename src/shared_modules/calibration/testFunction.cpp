#include "testFunction.hpp"


//����ֱ��ƽ�潻��
// plane ƽ�淽��A,B,C,Dʽ��
int PLP_analytic(cv::Mat plane, cv::Mat pp1, cv::Mat pp2, cv::Point3f &plp)
{

	double a = plane.at<float>(0, 0);
	double b = plane.at<float>(1, 0);
	double c = plane.at<float>(2, 0);
	double d = plane.at<float>(3, 0);

	cv::Point3f p1 = cv::Point3f(pp1.at<float>(0, 0), pp1.at<float>(1, 0), pp1.at<float>(2, 0));
	cv::Point3f p2 = cv::Point3f(pp2.at<float>(0, 0), pp2.at<float>(1, 0), pp2.at<float>(2, 0));

	double l = p2.x - p1.x;//ֱ�ߵķ�������(m,n,l)
	double m = p2.y - p1.y;
	double n = p2.z - p1.z;

	double eps = 0;			//������ֵ�ļ��㾫��
	if (fabs(l*a + m*b + n*c) <= eps)//ֱ������ƽ��
	{
		plp = cv::Point3f(0, 0, 0);
		return -1;
	}

	double dis = sqrt(l*l + m*m + n*n);
	double cos_alpha = l / dis;
	double cos_beta = m / dis;
	double cos_gamma = n / dis;

	double x1 = p1.x; double y1 = p1.y; double z1 = p1.z;

	double t = (a*x1 + b*y1 + c*z1 + d) / (a*cos_alpha + b*cos_beta + c*cos_gamma);

	plp.x = x1 - t * cos_alpha;
	plp.y = y1 - t * cos_beta;
	plp.z = z1 - t * cos_gamma;
	return 1;
}

//�����������ƽ�����
// OK 
void FP_analytic(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, double &A, double &B, double &C, double &D)
{
	double x1 = p1.x; double y1 = p1.y; double z1 = p1.z;
	double x21 = p2.x - p1.x;	double x31 = p3.x - p1.x;
	double y21 = p2.y - p1.y;	double y31 = p3.y - p1.y;
	double z21 = p2.z - p1.z;	double z31 = p3.z - p1.z;
	A = (y21 * z31 - z21 * y31);
	B = (z21 * x31 - x21 * z31);
	C = (x21 * y31 - y21 * x31);
	D = -x1 * A - y1 * B - z1 * C;

	double L = sqrt(A*A + B*B + C*C);
	A = A / L;
	B = B / L;
	C = C / L;
	D = D / L;
	if (A < 0)
	{
		A = -A;
		B = -B;
		C = -C;
		D = -D;
	}
}
void FP_analytic2(vector<cv::Point3f> pts, cv::Mat &plane)
{
	plane = cv::Mat::zeros(4, 1, CV_32F);
	if (pts.size() < 3){
		cout << "Not enough points to calculate plane para" << endl;
	}
	cv::Mat  M = cv::Mat::zeros(pts.size(), 4, CV_32F);
	for (unsigned int n = 0; n < pts.size(); n++)
	{
		int ind = n;
		M.at<float>(ind, 0) = pts[n].x;
		M.at<float>(ind, 1) = pts[n].y;
		M.at<float>(ind, 2) = pts[n].z;
		M.at<float>(ind, 3) = 1;
	}
	// Do SVD
	cv::Mat U, D, W, VT;
	cv::SVDecomp(M, W, U, VT);
	cv::Mat vec = VT.t().col(3);
	double L = cv::norm(vec.rowRange(0, 3));
	vec = vec / (L + 1e-5);
	if (vec.at<float>(0) < 0)
	{
		vec = -vec;
	}
	plane = vec.clone();
}

//RT->ƽ��xxxxx
void FP_analytic3(cv::Mat R, cv::Mat T, cv::Mat &plane)
{
	cv::Point3f p1, p2, p3;
	double A, B, C, D;
	cv::Mat P0 = (cv::Mat_<float>(3, 1) << 0, 0, 0);
	cv::Mat P1 = (cv::Mat_<float>(3, 1) << 1, 0, 0);
	cv::Mat P2 = (cv::Mat_<float>(3, 1) << 0, 1, 0);

	P0 = R*P0 + T;
	P1 = R*P1 + T; 
	P2 = R*P2 + T;

	p1 = cv::Point3f(P0.at<float>(0, 0), P0.at<float>(1, 0), P0.at<float>(2, 0));
	p2 = cv::Point3f(P1.at<float>(0, 0), P1.at<float>(1, 0), P1.at<float>(2, 0));
	p3 = cv::Point3f(P2.at<float>(0, 0), P2.at<float>(1, 0), P2.at<float>(2, 0));

	FP_analytic(p1, p2, p3, A, B, C, D);
	plane = (cv::Mat_<float>(4, 1) << A, B, C, D);
}
