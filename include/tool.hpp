#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
/*得到一个三维点集pts_3d_set，尺寸是4*3*4，分别记录了四个矩形的四个角点的三维坐标*/
std::vector<std::vector<cv::Point3d>> pts_3d_set;
std::vector<int> pts_ids;//四个id的标号分别是0,10,20,30

/*像素坐标与物理坐标之间的转换*/


/* 获取Z=0平面上的矩形区域四个顶点坐标 */
//屏幕坐标系在空间坐标系下的表示，x轴水平，y轴垂直，z轴朝向屏幕
std::vector<cv::Point3d> getRectPts3d(cv::Rect roi)
{
	std::vector<cv::Point3d> pts_3d;
	int x = roi.x;
	int y = roi.y;
	int w = roi.width;
	int h = roi.height;
	pts_3d.emplace_back(cv::Point3d(x, y, 0));
	pts_3d.emplace_back(cv::Point3d(x + w, y, 0));
	pts_3d.emplace_back(cv::Point3d(x + w, y + h, 0));
	pts_3d.emplace_back(cv::Point3d(x, y + h, 0));
	return pts_3d;
}

bool getRT(std::vector<std::vector<cv::Point3d>> pts3d_set, std::vector<std::vector<cv::Point2f>> pts2d_set, std::vector<int> pts_2d_ids, cv::Mat K, cv::Mat d, cv::Mat& r, cv::Mat& T)
{
	/**
	 * @brief: get RT
	 * @param: pts3d_set: 四个aruco标记的四个角点三维像素坐标，z=0
	 * @param: pts2d_set: 四个aruco标记的四个角点二维像素坐标
	 * @param: pts_2d_ids: 四个标记的id
	 * @param: K: 相机内参矩阵
	 * @param: d: 畸变系数
	 * @param: r: 旋转矩阵
	 * @param: t: 平移向量
	 */
	
	std::vector<cv::Point3d> pts_3d;
	std::vector<cv::Point2d> pts_2d;
	/*寻找与检测到的标记序号对应的标记*/
	for (int i = 0; i < pts_2d_ids.size(); i++)
	{
		for (int j = 0; j < pts_ids.size(); j++)
		{
			if (pts_2d_ids[i] == pts_ids[j])
			{
				for (int p = 0; p < pts2d_set[i].size(); p++)
				{
					pts_3d.push_back(pts3d_set[j][p]);
					pts_2d.push_back(pts2d_set[i][p]);
				}
				break;
			}
		}
	}
	if (pts_3d.size() >= 12)
	{
		cv::solvePnP(pts_3d, pts_2d, K, d, r, T);
		r.convertTo(r, CV_64F);
		T.convertTo(T, CV_64F);
		return true;
	}
	return false;
}

/* 生成Aruco标志图 */
//四个aruco标记左上角像素坐标分别是（450,250）,(1270,250),(1270,830),(450,830)
cv::Mat generateArucoImg()
{
	cv::Mat aruco_img = cv::Mat(cv::Size(1920, 1080), CV_8U);
	aruco_img.setTo(255);
	std::vector<cv::Mat> marker_imgs(4);
	int size = 200;   //标志长宽
	int border_x = 450;    //标志到图像x方向边界距离
	int border_y = 250;    //标志到图像y方向边界距离
	/*像素坐标变换到物理坐标*/
	/*屏幕像素尺寸为1920*1080，物理尺寸为537*296 mm */
	double w_physical = 537/1920;//单位像素宽度的物理尺寸
	double h_physical = 296/1080;//单位像素高度的物理尺寸
	double width = size*w_physical;
	double height = size*h_physical;
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
	for (int i = 0; i < 4; ++i)
	{
		cv::aruco::drawMarker(dictionary, i * 10, size, marker_imgs[i], 1);
		pts_ids.emplace_back(i * 10);//四个id的标号分别是0,10,20,30
	}
	cv::Rect roi;
	roi = cv::Rect(border_x*w_physical, border_y*h_physical, width, height);//左上角矩形的左上角像素坐标是(450,250)，再转化为物理坐标
	pts_3d_set.emplace_back(getRectPts3d(roi));//得到左上角矩形四个角点的3d坐标，z=0
	//marker_imgs[0].copyTo(aruco_img(roi));//将左上角标记画到aruco_img的左上角矩形区域中
	roi = cv::Rect((aruco_img.cols - size - border_x)*w_physical, border_y*h_physical, width, height);
	pts_3d_set.emplace_back(getRectPts3d(roi));
	//marker_imgs[1].copyTo(aruco_img(roi));
	roi = cv::Rect((aruco_img.cols - size - border_x)*w_physical, (aruco_img.rows - size - border_y)*h_physical, width, height);
	pts_3d_set.emplace_back(getRectPts3d(roi));
	//marker_imgs[2].copyTo(aruco_img(roi));
	roi = cv::Rect(border_x*w_physical, (aruco_img.rows - size - border_y)*h_physical, width, height);
	pts_3d_set.emplace_back(getRectPts3d(roi));
	//marker_imgs[3].copyTo(aruco_img(roi));
	/*得到一个三维点集pts_3d_set，尺寸是4*3*4，分别记录了四个矩形的四个角点的三维坐标*/
	//cv::imwrite("D:/disImg.bmp", aruco_img);
	//cv::imshow("aruco_img", aruco_img);
	//cv::waitKey();
	return aruco_img;
}

/* 检测Aruco图并估计位姿 */
void detectAruco(cv::Mat img, cv::Mat K, cv::Mat d, cv::Mat& R, cv::Mat& T)
{
	std::vector<int> ids;//每个aruco标记对应的标识符
	std::vector<std::vector<cv::Point2f>> corners;//每个aruo标记的四个角点坐标，从左上角开始按顺时针顺序排列
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
	cv::aruco::detectMarkers(img, dictionary, corners, ids);
	if (ids.size() > 0)
	{
		cv::aruco::drawDetectedMarkers(img, corners, ids);
		cv::Mat r;
		if (getRT(pts_3d_set, corners, ids, K, d, r, T))
		{
			cv:drawFrameAxes(img, K, d, r, T, 300, 3);
			cv::Rodrigues(r, R);
			R.convertTo(R, CV_64F);
		}
	}
}


void arDemo()
{
	cv::Mat aruco_img = generateArucoImg();
	cv::VideoCapture cap(1);
	//需传入标定参数
	cv::Mat K = (cv::Mat_<double>(3, 3) << 777.598586727608, 0, 290.274367732162,
		0, 770.675455625097, 236.607744724502,
		0, 0, 1);
	cv::Mat d = (cv::Mat_<double>(1, 5) << -0.0695693343254078, 0.531304460065513, 0, 0, 0);
	while (true)
	{
		cv::Mat img;
		cap >> img;
		if (img.empty())
		{
			break;
		}
		cv::Mat R, T;
		detectAruco(img, K, d, R, T);
		cv::imshow("img", img);
		cv::imshow("aruco_img", aruco_img);
		cv::waitKey(1);
	}
	cap.release();
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string getVideoDevice(const std::string& deviceName) {
    std::string cmd = "v4l2-ctl --list-devices";
    std::string output = exec(cmd.c_str());
   
    std::istringstream stream(output);
    std::string line;
    bool foundDevice = false;
    while (std::getline(stream, line)) {
		cout << line << endl;
        if (line.find(deviceName) != std::string::npos) {
            foundDevice = true;
        } 
		else if (foundDevice && line.find("/dev/video") != std::string::npos) 
		{
			cout << __LINE__ << line << endl;
			// 去掉前后的空格
    		line.erase(0, line.find_first_not_of(" \t\n\r"));
    		line.erase(line.find_last_not_of(" \t\n\r") + 1);
			return line;
        }
    }
}
