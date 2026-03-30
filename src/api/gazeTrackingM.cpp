#include "gazeTracingM.h"
#include "opencv2/opencv.hpp"
#include "../shared_cpp/include/common/types.h"
#include "../shared_modules/calibration/testFunction.hpp"


CGazeTrakingM::CGazeTrakingM():mEyeModel(527.61)  //�����ڴ˴���ֵ
{
	m_filename = "";
	m_isOnLineCalibration = false;
	cv::Mat angle(1,2,CV_64FC1);
	angle.at<double>(0,0) = 0;
	angle.at<double>(0,1) = 0;
	m_Angle = angle;

	m_imageL = cv::Mat::zeros(480,640,CV_8UC1);
	m_imageR = cv::Mat::zeros(480,640,CV_8UC1);


	m_pupil.center.x = -1;
	m_pupil.center.y = -1;

	m_PupilPoision.resize(16);
	m_screenPoision.resize(16);
	

	m_BlinkDection.SetParams(6, 0.5, 0.5);


	m_eye0ToWorld = cv::Mat::eye(cv::Size(4, 4), CV_32FC1);
	m_planeInCamera = cv::Mat::ones(4, 1, CV_32FC1);

	maxSample = 40;
	maxCalibrationPoint = 15;

	m_discardNum = 0;
}

CGazeTrakingM::~CGazeTrakingM()
{

}

bool CGazeTrakingM::ReadParams(std::string filename)
{
	if (0 == filename.length())
	{
		return false;
	}
	m_filename = filename;
	cv::FileStorage storage1up(filename, cv::FileStorage::READ);
	storage1up["R"] >> m_R;   
	storage1up["T"] >> m_T;   
	storage1up["K1"] >> m_K1;   
	storage1up["K2"] >> m_K2; 
	storage1up["D1"] >> m_D1;
	storage1up["D2"] >> m_D2;


	mEyeModel.m_cx = 320;
	mEyeModel.m_cy = 240;

	storage1up["Rpd"] >> m_Rpd;   
	storage1up["Tpd"] >> m_Tpd; 

	cv::Mat normalVec(3,1,CV_32F),normalVec1;
	m_plane = cv::Mat::zeros(4,1,CV_32F);
	normalVec.at<float>(0,0) = 0;
	normalVec.at<float>(1,0) = 0;
	normalVec.at<float>(2,0) = 1;
	normalVec1 = m_Rpd * normalVec;
	
	normalVec1.copyTo(m_plane.rowRange(0,3).colRange(0,1));
	//cout<<norm(m_Tpd)<<endl;
	m_plane.at<float>(3,0) = norm(m_Tpd);

	m_Plane_scale =  0.0473;
	storage1up["LightPosition"] >> m_LightPosition;  
	if (true == m_isOnLineCalibration)
	{
		storage1up["angle"] >> m_Angle; 
	}	
	storage1up.release();
}

bool CGazeTrakingM::ReadImage(cv::Mat imageL, cv::Mat imageR)
{
	if (true == imageL.empty() || true == imageR.empty())
	{
		return false;
	}
	m_imageL = imageL;
	m_imageR = imageR;
	return true;
}

bool CGazeTrakingM::ReadImage(uchar* image)
{
	if (NULL == image)
	{
		return false;
	}
	int width  = 640;
	int height = 480;
	//for(int i = 0 ; i < height; i++)
	//{
	//	memcpy(m_imageL.data+width*i,	image+(2*i)*width,	width);
	//	memcpy(m_imageR.data+width*i,	image+(2*i+1)*width,	width);
	//}
	memcpy(m_imageL.data, image, 640 * 480);
	return true;
}

bool IntterSortLightPosition(std::vector<cv::Point2f> &irptsL)
{
	if (irptsL.size() > 1)
	{
	}
	else
	{
		return false;
	}
	// L �����
    std::vector<cv::Point2f> tempirptsLL;
	std::vector<cv::Point2f> tempirptsLR;
	double meanxL = 320;
	double meanyL = 240;
	double sumxL = 0;
	double sumyL = 0;
	for (int i = 0; i < irptsL.size(); i++) // �������ĵ㣻
	{
		sumxL += irptsL[i].x;
		sumyL += irptsL[i].y;
	}
	meanxL = sumxL / irptsL.size();
	meanyL = sumxL / irptsL.size();


	// �������߷ֿ�
	for (int i = 0; i < irptsL.size(); i++)
	{
		if (irptsL[i].x < meanxL)  // ���
		{
			tempirptsLL.push_back(irptsL[i]);
		}
		else //�ұ�
		{
			tempirptsLR.push_back(irptsL[i]);
		}
	}
	//  ����� ��� ���·ֿ�;
	if (2 == tempirptsLL.size())
	{
		if (tempirptsLL[0].y > (tempirptsLL[0].y + tempirptsLL[1].y) / 2)
		{

		}
		else
		{
			cv::Point2f temppoint;
			temppoint = tempirptsLL[0];
			tempirptsLL[0] = tempirptsLL[1];
			tempirptsLL[1] = temppoint;
		}
	}
	else
	{
		if (tempirptsLL[0].y > meanyL)
		{
			cv::Point2f temppoint;
			temppoint.x = -1;
			temppoint.y = -1;
			tempirptsLL.push_back(temppoint);
		}
		else
		{
			cv::Point2f temppoint2;
			temppoint2 = tempirptsLL[0];
			cv::Point2f temppoint1;
			temppoint1.x = -1;
			temppoint1.y = -1;
			tempirptsLL[0] = temppoint1;
			tempirptsLL.push_back(temppoint2);
		}


	}
	//  ����� �ұ� ���·ֿ�;

	if (2 == tempirptsLR.size())
	{
		if (tempirptsLR[0].y < (tempirptsLR[0].y + tempirptsLR[1].y) / 2)
		{

		}
		else
		{
			cv::Point2f temppoint;
			temppoint = tempirptsLR[0];
			tempirptsLR[0] = tempirptsLR[1];
			tempirptsLR[1] = temppoint;
		}
	}
	else
	{
		if (tempirptsLR[0].y < meanyL)
		{
			cv::Point2f temppoint;
			temppoint.x = -1;
			temppoint.y = -1;
			tempirptsLR.push_back(temppoint);
		}
		else
		{
			cv::Point2f temppoint2;
			temppoint2 = tempirptsLR[0];
			cv::Point2f temppoint1;
			temppoint1.x = -1;
			temppoint1.y = -1;
			tempirptsLR[0] = temppoint1;
			tempirptsLR.push_back(temppoint2);
		}
	}
	irptsL.resize(4);
	irptsL[0] = tempirptsLL[0];
	irptsL[1] = tempirptsLL[1];
	irptsL[2] = tempirptsLR[0];
	irptsL[3] = tempirptsLR[1];
	return true;
}

bool CGazeTrakingM::SortLightPosition(cv::Point2f pupilL, cv::Point2f pupilR, std::vector<cv::Point2f> &irptsL, std::vector<cv::Point2f> &irptsR, int model)
{
	if (pupilL.x > 0 && pupilL.y > 0 && pupilR.x > 0 && pupilR.y > 0 && pupilL.x <640 && pupilL.y < 480 && pupilR.x <640 && pupilR.y < 480)
	{
		if (0 == model) // ���߱궨
		{
			if (irptsL.size() > 2 && irptsR.size() > 2)  //����������
			{
				// L �����
				IntterSortLightPosition(irptsL);
				IntterSortLightPosition(irptsR);
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			if (irptsL.size() > 1 && irptsR.size() > 1)  //����2���� ����취
			{
				// L �����

				return true;
			}
			else
			{

				return false;
			}

		}
		
		
	}
	else
	{
		return false;
	}
	

}

bool CGazeTrakingM::onlineCalibration(cv::Point2f gazePoint)
{
	
	return true;
}

bool CGazeTrakingM::onlineCalibration(uchar* image, double disTheshold, cv::Point2f gazePoint) 
{
	return true;
}

// �ɼ��궨����
bool CGazeTrakingM::onlineCalibration(uchar* image, double disTheshold, cv::Point2f gazePoint, int index)
{
	singleeyefitter::Detector2DProperties  props;
	props.intensity_range = 23;                                // ��ֵ������ֵ������С��һ��ʹ��
	props.blur_size = 5;                                       // ��ֵ�˲�����Ĵ�С
	props.canny_treshold = 160;                                // canny ��ֵ1
	props.canny_ration = 2;                                    // ��canny_treshold һ�����canny��ֵ2
	props.canny_aperture = 5;                                  // canny�˲�����Ĵ�С
	props.pupil_size_max = 200;                                // ͫ�׳������ֵ��ֱ����
	props.pupil_size_min = 30;                                 // ͫ�׳�����Сֵ��ֱ����
	props.strong_perimeter_ratio_range_min = 0.6;              // 
	props.strong_perimeter_ratio_range_max = 1.1;
	props.strong_area_ratio_range_min = 0.8;                   // ������͹�����/������ϵ���Բ�����Сֵ
	props.strong_area_ratio_range_max = 1.1;
	props.contour_size_min = 5;                                // ��������С����
	props.ellipse_roundness_ratio = 0.09;                      // ����ȳ������Сֵ
	props.initial_ellipse_fit_treshhold = 4.3;                 // �����Բ�㵽��Բ�ı߽����ķ�������ֵ
	props.final_perimeter_ratio_range_min = 0.5;               // �����жϣ����ԭʼ�߽��/�����Բ���ܳ�
	props.final_perimeter_ratio_range_max = 1.0;
	props.ellipse_true_support_min_dist = 3.0;                 // ԭʼ�����㵽�����Բ�߽�ľ�����ֵ��С�ڴ�ֵ����Ϊ��OK��

	
	
	try
	{
		if (index > 15 || index < 0)
		{
			return false;
		}
		if (NULL == image)
		{
			return false;
		}
		int width = 640;
		int height = 480;

		cv::Mat image_temp = cv::Mat::zeros(480, 640, CV_8U);
		for (int i = 0; i < height; i++)
		{
			memcpy(m_imageL.data + width*i, image + (2 * i)*width, width);
			//memcpy(m_imageR.data + width*i, image + (2 * i + 1)*width, width);
		}

		box2d pupil;
		pupil.center.x = -1;
		pupil.center.y = -1;

		
		cv::resize(m_imageL, image_temp, cv::Size(320, 240));
		
		cv::Mat color_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::cvtColor(image_temp, color_image, cv::COLOR_GRAY2RGB);
		cv::Mat debug_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Rect roi;
		detect2d.coarse_center1(image_temp, 20, 30, roi);
		//roi.x = 10;
		//roi.y = 10;
		//roi.width = 600;
		//roi.height = 440;
		
		std::shared_ptr<Detector2DResult> presultPupil = detect2d.detect(props, image_temp, color_image, debug_image, roi, true, false);
		m_PupilConfidence = presultPupil->confidence;
		m_PupilRadius = presultPupil->ellipse.major_radius;
		m_BlinkDection.PushPupilDetectConfidence(m_PupilConfidence);

		cv::imshow("color_image", color_image);
		
		if (fabs(presultPupil->confidence) > 0.2)
		{
			pupil.center.x = presultPupil->ellipse.center[0];
			pupil.center.y = presultPupil->ellipse.center[1];
		}
		else
		{

			return false;
		}


		if (pupil.center.x < 0 || pupil.center.y < 0 || pupil.center.x > 640 || pupil.center.y > 480)
		{
			return false;
		}
		else 
		{
			if (m_calibratePupil.size() < 15)
			{
				m_calibratePupil.push_back(pupil.center);
			}			
		}


		if (15 == m_calibratePupil.size())
		{
			double sumx = 0;
			double sumy = 0;
			for (int i = 0; i < m_calibratePupil.size(); i++)
			{
				sumx += m_calibratePupil[i].x;
				sumy += m_calibratePupil[i].y;
			}
			double avex = 0;
			double avey = 0; 

			avex = sumx / 15;
			avey = sumy / 15;
			double varx = 0;
			double vary = 0;

			for (int i = 0; i < m_calibratePupil.size(); i++)
			{
				varx += (m_calibratePupil[i].x - avex) * (m_calibratePupil[i].x - avex);
				vary += (m_calibratePupil[i].y - avey) * (m_calibratePupil[i].y - avey);
			}

			if (sqrt(varx + vary) / 15 < disTheshold)
			{
				//m_PupilPoision[index].x = pupil.center.x;
				//m_PupilPoision[index].y = pupil.center.y;
				m_PupilPoision[index].x = avex;
				m_PupilPoision[index].y = avey;
				m_screenPoision[index] = gazePoint;
				m_calibratePupil.clear();
				return true;
			}
			else
			{
				std::vector<pointf>::iterator it = m_calibratePupil.begin();
				m_calibratePupil.erase(it);
			}
		}

		if (m_calibratePupil.size() > 15)
		 {
			 m_calibratePupil.clear();
			 //---------------------------------------------------����---------------------------------------------+
		 }


		return false;
	}
	catch (...)
	{
		return false;
	}
}

// �ɼ��궨���� 3D-pc
bool CGazeTrakingM::onlineCalibration3D(uchar* image, double disTheshold, cv::Point3f gazePoint, int index)
{


	singleeyefitter::Detector2DProperties  props;
	props.intensity_range = 23;                                // ��ֵ������ֵ������С��һ��ʹ��
	props.blur_size = 5;                                       // ��ֵ�˲�����Ĵ�С
	props.canny_treshold = 160;                                // canny ��ֵ1
	props.canny_ration = 2;                                    // ��canny_treshold һ�����canny��ֵ2
	props.canny_aperture = 5;                                  // canny�˲�����Ĵ�С
	props.pupil_size_max = 200;                                // ͫ�׳������ֵ��ֱ����
	props.pupil_size_min = 30;                                 // ͫ�׳�����Сֵ��ֱ����
	props.strong_perimeter_ratio_range_min = 0.6;              // 
	props.strong_perimeter_ratio_range_max = 1.1;
	props.strong_area_ratio_range_min = 0.8;                   // ������͹�����/������ϵ���Բ�����Сֵ
	props.strong_area_ratio_range_max = 1.1;
	props.contour_size_min = 5;                                // ��������С����
	props.ellipse_roundness_ratio = 0.09;                      // ����ȳ������Сֵ
	props.initial_ellipse_fit_treshhold = 4.3;                 // �����Բ�㵽��Բ�ı߽����ķ�������ֵ
	props.final_perimeter_ratio_range_min = 0.5;               // �����жϣ����ԭʼ�߽��/�����Բ���ܳ�
	props.final_perimeter_ratio_range_max = 1.0;
	props.ellipse_true_support_min_dist = 3.0;                 // ԭʼ�����㵽�����Բ�߽�ľ�����ֵ��С�ڴ�ֵ����Ϊ��OK��



	try
	{
		if (index > maxCalibrationPoint || index < 0)
		{
			return false;
		}
		if (NULL == image)
		{
			return false;
		}
		int width = 640;
		int height = 480;

		cv::Mat image_temp = cv::Mat::zeros(480, 640, CV_8U);
		//for (int i = 0; i < height; i++)
		//{
		//	memcpy(m_imageL.data + width*i, image + (2 * i)*width, width);
		//	//memcpy(m_imageR.data + width*i, image + (2 * i + 1)*width, width);
		//}
		memcpy(image_temp.data, image, 640 * 480);

		
		box2d pupil;
		pupil.center.x = -1;
		pupil.center.y = -1;


		cv::Mat color_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::cvtColor(image_temp, color_image, cv::COLOR_GRAY2RGB);
		cv::Mat debug_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Rect roi;
		detect2d.coarse_center1(image_temp, 40, 60, roi);
		//roi.x = 10;
		//roi.y = 10;
		//roi.width = 600;
		//roi.height = 440;

		std::shared_ptr<Detector2DResult> presultPupil = detect2d.detect(props, image_temp, color_image, debug_image, roi, true, false);
		m_PupilConfidence = presultPupil->confidence;
		m_PupilRadius = presultPupil->ellipse.major_radius;
		m_BlinkDection.PushPupilDetectConfidence(m_PupilConfidence);
		if (presultPupil->confidence < 0.1)
		{
			return false;
		}

		singleeyefitter::Detector3DProperties detectProperties3D;
		detectProperties3D.model_sensitivity = 0.997;
		Detector3DResult result3D = mEyeModel.updateAndDetect(presultPupil, detectProperties3D, false);



#if 0    //show debug image  y�ᷴ
		cv::Mat showImage;
		cv::cvtColor(image_temp, showImage, cv::COLOR_GRAY2RGB);

		// 3DԲ����ͼ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.ellipse.center[0] + 320), (int)(240 - result3D.ellipse.center[1])), cv::Size((int)result3D.ellipse.major_radius, (int)result3D.ellipse.minor_radius), -result3D.ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(255, 255, 255));
		// 2D��ƽ���⵽����Բͫ��
		cv::ellipse(showImage, cv::Point((int)(presultPupil->ellipse.center[0] + 320), (int)(240 - presultPupil->ellipse.center[1])), cv::Size((int)presultPupil->ellipse.major_radius, (int)presultPupil->ellipse.minor_radius), -presultPupil->ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(0, 0, 255));
		// 3D ������2D��ƽ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 320), (int)(240 - result3D.projectedSphere.center[1])), cv::Size(2, 2), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
		// ����
		cv::line(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 320), (int)(240 - result3D.projectedSphere.center[1])), cv::Point((int)(result3D.ellipse.center[0] + 320), (int)(240 - result3D.ellipse.center[1])), cv::Scalar(0, 0, 255), 2);

		cv::imshow("live-debug", showImage);



#endif

#if 1    //show debug image y��δ��
		cv::Mat showImage;
		cv::cvtColor(image_temp, showImage, cv::COLOR_GRAY2RGB);

		// 3DԲ����ͼ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.ellipse.center[0] + 320), (int)( result3D.ellipse.center[1] + 240)), cv::Size((int)result3D.ellipse.major_radius, (int)result3D.ellipse.minor_radius), result3D.ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(255, 255, 255));
		// 2D��ƽ���⵽����Բͫ��
		cv::ellipse(showImage, cv::Point((int)(presultPupil->ellipse.center[0] + 320), (int)(presultPupil->ellipse.center[1] + 240)), cv::Size((int)presultPupil->ellipse.major_radius, (int)presultPupil->ellipse.minor_radius), presultPupil->ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(0, 0, 255));
		// 3D ������2D��ƽ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 320), (int)(result3D.projectedSphere.center[1] + 240)), cv::Size(2, 2), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
		// ����
		cv::line(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 320), (int)(result3D.projectedSphere.center[1] + 240)), cv::Point((int)(result3D.ellipse.center[0] + 320), (int)(result3D.ellipse.center[1] + 240)), cv::Scalar(0, 0, 255), 2);

		cv::imshow("live-debug", showImage);



#endif

		if (fabs(result3D.confidence) > 0.75)
		{
			pupil.center.x = presultPupil->ellipse.center[0];
			pupil.center.y = presultPupil->ellipse.center[1];
		}
		else
		{

			return false;
		}
		
		if (m_calibratePupil.size() < maxSample)                                                                                       // ��ʱ�ɼ����������㣻
		{
			m_discardNum++;
			if (m_discardNum > 30)
			{
				if (m_3DPupilPoisionALL.size() == 0)                                                                                  // ��һ���궨��
				{
					m_calibratePupil.push_back(pupil.center);
					m_3DPupilPoisionTemp.push_back(result3D);
					m_3DScreenPosionTemp.push_back(gazePoint);
				}
				else                                                                                                                   // �����궨�㣬��ǰһ���궨������߽�ҪС��0.99                                                                                             
				{
					double cos_thita = m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].circle.normal[0] * result3D.circle.normal[0] +
						m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].circle.normal[1] * result3D.circle.normal[1] +
						m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].circle.normal[2] * result3D.circle.normal[2];

					std::cout << "�ɼ�����ʱ������ɵĽǶȣ�" << cos_thita << std::endl;
					if (cos_thita < 0.99)
					{
						m_calibratePupil.push_back(pupil.center);
						m_3DPupilPoisionTemp.push_back(result3D);
						m_3DScreenPosionTemp.push_back(gazePoint);
					}
				}
			}
		}

		if (maxSample == m_calibratePupil.size())
		{
			double sumx = 0;
			double sumy = 0;
			for (int i = 0; i < m_calibratePupil.size(); i++)
			{
				sumx += m_calibratePupil[i].x;
				sumy += m_calibratePupil[i].y;
			}
			double avex = 0;
			double avey = 0;

			avex = sumx / maxSample;
			avey = sumy / maxSample;
			double varx = 0;
			double vary = 0;

			for (int i = 0; i < m_calibratePupil.size(); i++)
			{
				varx += (m_calibratePupil[i].x - avex) * (m_calibratePupil[i].x - avex);
				vary += (m_calibratePupil[i].y - avey) * (m_calibratePupil[i].y - avey);
			}

			if (sqrt(varx + vary) / maxSample < disTheshold)
			{
				//m_PupilPoision[index].x = pupil.center.x;
				//m_PupilPoision[index].y = pupil.center.y;
				m_PupilPoision[index].x = avex;
				m_PupilPoision[index].y = avey;
				//m_3DScreenPosion[index] = gazePoint;

				m_3DPupilPoisionALL.push_back(m_3DPupilPoisionTemp);
				m_3DScreenPosionALL.push_back(m_3DScreenPosionTemp);

				m_3DPupilPoisionTemp.clear();
				m_3DScreenPosionTemp.clear();
				m_calibratePupil.clear();

				m_discardNum = 0;

				return true;
			}
			else
			{
				std::vector<pointf>::iterator it = m_calibratePupil.begin();
				std::vector<singleeyefitter::Detector3DResult>::iterator it1 = m_3DPupilPoisionTemp.begin();
				std::vector<cv::Point3f>::iterator it2 = m_3DScreenPosionTemp.begin();
				  
				m_calibratePupil.erase(it);
				m_3DPupilPoisionTemp.erase(it1);
				m_3DScreenPosionTemp.erase(it2);
			}
		}

		if (m_calibratePupil.size() > maxSample)
		{
			m_calibratePupil.clear();
			m_3DPupilPoisionTemp.clear();
			m_3DScreenPosionTemp.clear();
			assert(0);
			//---------------------------------------------------����---------------------------------------------+
		}


		return false;
	}
	catch (...)
	{
		return false;
	}
}



// mapping
int CGazeTrakingM::GetMappingResult(double threshold)
{
	//   Xs = a0 + a1xe + a2ye + a3xeye + a4xe^2 + a5ye^2;
	//   Ys = a6 + a7xe + a8ye + a9xeye + a10xe^2 + a11ye^2;

	int ret = 0; 
	int nums = 0;
	cv::Mat Xe(12, 1, CV_32F);
	cv::Mat A = cv::Mat ::zeros(30,12,CV_32F);
	cv::Mat b(30, 1, CV_32F);



	for (int i = 0; i < 15; i++)
	{
		A.at<float>(i, 0) = 1;
		A.at<float>(i, 1) = m_PupilPoision[i].x;
		A.at<float>(i, 2) = m_PupilPoision[i].y;
		A.at<float>(i, 3) = m_PupilPoision[i].x * m_PupilPoision[i].y;
		A.at<float>(i, 4) = m_PupilPoision[i].x * m_PupilPoision[i].x;
		A.at<float>(i, 5) = m_PupilPoision[i].y * m_PupilPoision[i].y;

		b.at<float>(i, 0) = m_screenPoision[i].x;
				
	}
	for (int i = 15; i < 30; i++)
	{
		A.at<float>(i, 6) = 1;
		A.at<float>(i, 7) = m_PupilPoision[i - 15].x;
		A.at<float>(i, 8) = m_PupilPoision[i - 15].y;
		A.at<float>(i, 9) = m_PupilPoision[i - 15].x * m_PupilPoision[i - 15].y;
		A.at<float>(i, 10) = m_PupilPoision[i - 15].x * m_PupilPoision[i - 15].x;
		A.at<float>(i, 11) = m_PupilPoision[i - 15].y * m_PupilPoision[i - 15].y;

		b.at<float>(i, 0) = m_screenPoision[i-15].y;
	}
	cv::solve(A, b, Xe, cv::DECOMP_SVD);
	// 
	std::cout << norm(A * Xe - b)/15 << std::endl;

	if (norm(A *Xe - b) > threshold * 15)
	{
		//return ret;
	}
	else
	{
		//ret += 1 << 15;
	}
	
	m_mapping[0] = Xe.at<float>(0, 0);
	m_mapping[1] = Xe.at<float>(1, 0);
	m_mapping[2] = Xe.at<float>(2, 0);
	m_mapping[3] = Xe.at<float>(3, 0);
	m_mapping[4] = Xe.at<float>(4, 0);
	m_mapping[5] = Xe.at<float>(5, 0);
	m_mapping[6] = Xe.at<float>(6, 0);
	m_mapping[7] = Xe.at<float>(7, 0);
	m_mapping[8] = Xe.at<float>(8, 0);
	m_mapping[9] = Xe.at<float>(9, 0);
	m_mapping[10] = Xe.at<float>(10, 0);
	m_mapping[11] = Xe.at<float>(11, 0);
	
	//  ���ÿ����
	double errorx, errory;
	for (int i = 0; i < 15; i ++)
	{
		errorx = m_screenPoision[i].x - (m_mapping[0] + m_mapping[1] * m_PupilPoision[i].x + m_mapping[2] * m_PupilPoision[i].y + m_mapping[3] * m_PupilPoision[i].x * m_PupilPoision[i].y + m_mapping[4] * m_PupilPoision[i].x * m_PupilPoision[i].x + m_mapping[5] * m_PupilPoision[i].y * m_PupilPoision[i].y);

		errory = m_screenPoision[i].y - (m_mapping[6] + m_mapping[7] * m_PupilPoision[i].x + m_mapping[8] * m_PupilPoision[i].y + m_mapping[9] * m_PupilPoision[i].x * m_PupilPoision[i].y + m_mapping[10] * m_PupilPoision[i].x * m_PupilPoision[i].x + m_mapping[11] * m_PupilPoision[i].y  *m_PupilPoision[i].y);

		std::cout << i << ": " << sqrt(errorx * errorx + errory * errory) << std::endl;
		if (sqrt(errorx * errorx + errory * errory) > threshold)
		{

		}
		else
		{
			ret += 1 << i;
			nums ++ ;
		}
	}

	if (nums > 12)
	{
		ret += 1 << 15;
	}

	return ret;
}

int CGazeTrakingM::GetMappingResult9(double threshold)
{
	//   Xs = a0 + a1xe + a2ye + a3xeye + a4xe^2 + a5ye^2;
	//   Ys = a6 + a7xe + a8ye + a9xeye + a10xe^2 + a11ye^2;

	int ret = 0;
	int nums = 0;
	cv::Mat Xe(12, 1, CV_32F);
	cv::Mat A = cv::Mat::zeros(18, 12, CV_32F);
	cv::Mat b(18, 1, CV_32F);



	for (int i = 0; i < 9; i++)
	{
		A.at<float>(i, 0) = 1;
		A.at<float>(i, 1) = m_PupilPoision[i].x;
		A.at<float>(i, 2) = m_PupilPoision[i].y;
		A.at<float>(i, 3) = m_PupilPoision[i].x * m_PupilPoision[i].y;
		A.at<float>(i, 4) = m_PupilPoision[i].x * m_PupilPoision[i].x;
		A.at<float>(i, 5) = m_PupilPoision[i].y * m_PupilPoision[i].y;

		b.at<float>(i, 0) = m_screenPoision[i].x;

	}
	for (int i = 15; i < 24; i++)
	{
		A.at<float>(i - 6, 6) = 1;
		A.at<float>(i - 6, 7) = m_PupilPoision[i - 15].x;
		A.at<float>(i - 6, 8) = m_PupilPoision[i - 15].y;
		A.at<float>(i - 6, 9) = m_PupilPoision[i - 15].x * m_PupilPoision[i - 15].y;
		A.at<float>(i - 6, 10) = m_PupilPoision[i - 15].x * m_PupilPoision[i - 15].x;
		A.at<float>(i - 6, 11) = m_PupilPoision[i - 15].y * m_PupilPoision[i - 15].y;

		b.at<float>(i - 6, 0) = m_screenPoision[i - 15].y;
	}
	cv::solve(A, b, Xe, cv::DECOMP_SVD);
	// 
	std::cout << norm(A * Xe - b) / 9 << std::endl;

	if (norm(A *Xe - b) > threshold * 9)
	{
		//return ret;
	}
	else
	{
		//ret += 1 << 15;
	}

	m_mapping[0] = Xe.at<float>(0, 0);
	m_mapping[1] = Xe.at<float>(1, 0);
	m_mapping[2] = Xe.at<float>(2, 0);
	m_mapping[3] = Xe.at<float>(3, 0);
	m_mapping[4] = Xe.at<float>(4, 0);
	m_mapping[5] = Xe.at<float>(5, 0);
	m_mapping[6] = Xe.at<float>(6, 0);
	m_mapping[7] = Xe.at<float>(7, 0);
	m_mapping[8] = Xe.at<float>(8, 0);
	m_mapping[9] = Xe.at<float>(9, 0);
	m_mapping[10] = Xe.at<float>(10, 0);
	m_mapping[11] = Xe.at<float>(11, 0);

	//  ���ÿ����
	double errorx, errory;
	for (int i = 0; i < 9; i++)
	{
		errorx = m_screenPoision[i].x - (m_mapping[0] + m_mapping[1] * m_PupilPoision[i].x + m_mapping[2] * m_PupilPoision[i].y + m_mapping[3] * m_PupilPoision[i].x * m_PupilPoision[i].y + m_mapping[4] * m_PupilPoision[i].x * m_PupilPoision[i].x + m_mapping[5] * m_PupilPoision[i].y * m_PupilPoision[i].y);

		errory = m_screenPoision[i].y - (m_mapping[6] + m_mapping[7] * m_PupilPoision[i].x + m_mapping[8] * m_PupilPoision[i].y + m_mapping[9] * m_PupilPoision[i].x * m_PupilPoision[i].y + m_mapping[10] * m_PupilPoision[i].x * m_PupilPoision[i].x + m_mapping[11] * m_PupilPoision[i].y  *m_PupilPoision[i].y);

		std::cout << i << ": " << sqrt(errorx * errorx + errory * errory) << std::endl;
		if (sqrt(errorx * errorx + errory * errory) > threshold)
		{

		}
		else
		{
			ret += 1 << i;
			nums++;
		}
	}

	if (nums > 7)
	{
		ret += 1 << 15;
	}

	return ret;
}

// ����任����--3D
#if 0 //BA
int CGazeTrakingM::GetMappingResult9_3D(double threshold)
{
	double scalarZ;
	if (!(m_3DScreenPosionALL.size() == m_3DPupilPoisionALL.size() && m_3DScreenPosionALL.size() != 0))
	{
		return 0;  // ���ݲɼ�������Ҫ��
	}
	cv::Mat ref_points_3d_unscaled(3, m_3DScreenPosionALL.size() * m_3DScreenPosionALL[0].size(), CV_32FC1);
	cv::Mat gaze0_dir(3, m_3DPupilPoisionALL.size()*m_3DPupilPoisionALL[0].size(), CV_32FC1);

	// ������Ļ3�ĵ��ͫ�����������ľ�������
	float* pdata = ref_points_3d_unscaled.ptr<float>(0);
	float* pgaze0 = gaze0_dir.ptr<float>(0);


	for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
	{
		int sampleNumTemp = m_3DPupilPoisionALL[i].size();
		for (int j = 0; j < sampleNumTemp; j++)
		{
			pgaze0[i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[0];
			pgaze0[m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[1];
			pgaze0[2 * m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[2];
		}
	}

	for (int i = 0; i < m_3DScreenPosionALL.size(); i++)
	{
		int sampleNumTemp = m_3DScreenPosionALL[i].size();
		for (int j = 0; j < sampleNumTemp; j++)
		{
			pdata[i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].x;
			pdata[m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].y;
			pdata[2 * m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].z;
		}
	}
	//test--��ȡ����
	//std::cout << ref_points_3d_unscaled.t() << std::endl;
	//std::cout << gaze0_dir.t() << std::endl;
	//std::cout << gaze0_dir.col(0).t() * gaze0_dir.col(79) << std::endl;
	//std::cout << gaze0_dir.col(50).t() * gaze0_dir.col(400) << std::endl;
	for (int i = 1; i < m_3DPupilPoisionALL.size(); i++)
	{
		std::cout << "cos thita: " << m_3DPupilPoisionALL[0][0].circle.normal[0] * m_3DPupilPoisionALL[i][0].circle.normal[0] + m_3DPupilPoisionALL[0][0].circle.normal[1] * m_3DPupilPoisionALL[i][0].circle.normal[1] + m_3DPupilPoisionALL[0][0].circle.normal[2] * m_3DPupilPoisionALL[i][0].circle.normal[2] << std::endl;

	}





	double smallest_residual = 1000.0;                 //  �궨����޶�
	std::vector<double> optimalPose;
	// ��ʼ�궨��s�ɱ䣬�൱�ڵ�Ч��ƽ��ǰ���ƶ����������ԭ���Ч�������м�
	for (double s = 0.7; s < 50; s += 0.186)                   // s ʹ�ý����󣬾Ϳ��Բ����ٹ����ˣ�--��Ϊ������Ľ������ֱ�Ӹɵ�z��Ϳ�����
	{
		cv::Mat scalarMatrix = cv::Mat::eye(3, 3, CV_32FC1);
		scalarMatrix.at<float>(1, 1) = -1;
		scalarMatrix.at<float>(2, 2) = s;

		cv::Mat ref_points_3d = scalarMatrix * ref_points_3d_unscaled;

		// 1����ʼ�������ƣ�ƽ�ƣ�����ͫ��ȷ������ת���ݸ���任����ȷ����
		cv::Mat initial_translation0 = (cv::Mat_<float>(3, 1) << -30, 0, 0);      // ����x��Ϊ��

		cv::Mat H = Find_rigid_transform(gaze0_dir * 500, ref_points_3d);
		std::cout << "H ��ʼ���ƣ� " << H << std::endl;
		cv::Mat initial_R0 = H.rowRange(0, 3).colRange(0, 3);
		cv::Mat rotation_angle_axis;
		Rodrigues(initial_R0, rotation_angle_axis);

		// ǰ������R,T ��������ϵ����������ϵ�ı任 ---- Ϊ��BA�㷨�ķ��㣬����任Ϊ��������ϵ����������ϵ
		rotation_angle_axis *= -1;
		initial_translation0 *= -1;

		std::vector<Observer> observers;
		Observer observer;
		observer.fix_rotation = false;
		//observer.fix_translation = true;
		observer.fix_translation = false;

		observer.pose.push_back((double)rotation_angle_axis.at<float>(0, 0));
		observer.pose.push_back((double)rotation_angle_axis.at<float>(1, 0));
		observer.pose.push_back((double)rotation_angle_axis.at<float>(2, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(0, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(1, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(2, 0));

		for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
		{
			for (int j = 0; j < m_3DPupilPoisionALL[i].size(); j++)
			{
				Eigen::Matrix<double, 3, 1> temp;
				temp[0] = (double)m_3DPupilPoisionALL[i][j].circle.normal[0];
				temp[1] = (double)m_3DPupilPoisionALL[i][j].circle.normal[1];
				temp[2] = (double)m_3DPupilPoisionALL[i][j].circle.normal[2];

				observer.observations.push_back(temp);
			}

		}
		observers.push_back(observer);

		std::vector<Eigen::Matrix<double, 3, 1>> points;
		for (int i = 0; i < ref_points_3d.cols; i++)
		{
			Eigen::Matrix<double, 3, 1> temp;
			temp[0] = (double)ref_points_3d.at<float>(0, i);
			temp[1] = (double)ref_points_3d.at<float>(1, i);
			temp[2] = (double)ref_points_3d.at<float>(2, i);

			points.push_back(temp);
		}

		double final_cost = bundleAdjustCalibration(observers, points, true);


		if (final_cost == -1)
		{
			continue;
		}
		if (final_cost <= smallest_residual)
		{
			smallest_residual = final_cost;
			std::cout << "bundleAdjustCalibration--final_cost: " << final_cost << std::endl;
			std::cout << "bundleAdjustCalibration--s: " << s << std::endl;
			// ������תƽ��
			optimalPose = observers[0].pose;
			scalarZ = s;
		}

	}
	std::cout << "bundleAdjustCalibration--smallest_residual: " << smallest_residual << std::endl;
	if (optimalPose.size() != 6)
	{
		return 0;   //�궨���ɹ�
	}

	cv::Mat rotation_angle_axis_eye0_to_world(3, 1, CV_32FC1);

	rotation_angle_axis_eye0_to_world.at<float>(0, 0) = -1 * optimalPose[0];
	rotation_angle_axis_eye0_to_world.at<float>(1, 0) = -1 * optimalPose[1];
	rotation_angle_axis_eye0_to_world.at<float>(2, 0) = -1 * optimalPose[2];

	cv::Mat translation_eye0_to_world(3, 1, CV_32FC1);
	translation_eye0_to_world.at<float>(0, 0) = -1 * optimalPose[3];
	translation_eye0_to_world.at<float>(1, 0) = -1 * optimalPose[4];
	translation_eye0_to_world.at<float>(2, 0) = -1 * optimalPose[5];

	cv::Mat R_eye0_to_world(3, 3, CV_32FC1);
	cv::Rodrigues(rotation_angle_axis_eye0_to_world, R_eye0_to_world);

	// ������Ļ�㵽���ߵľ��룬���Ա궨�Ƿ�׼ȷ��

	// �����������ϵ����������ϵ��R,T

	cv::Mat sphere_pos0(3, 1, CV_32FC1);
	sphere_pos0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[0];
	sphere_pos0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[1];
	sphere_pos0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[2];

	cv::Mat translation_camera_to_world0 = translation_eye0_to_world - R_eye0_to_world * sphere_pos0;


	R_eye0_to_world.copyTo(m_eye0ToWorld.rowRange(0, 3).colRange(0, 3));
	translation_camera_to_world0.copyTo(m_eye0ToWorld.rowRange(0, 3).colRange(3, 4));


	std::cout << "m_eye0ToWorld" << m_eye0ToWorld << std::endl;

	// test	
	{
		std::cout << "scalarZ: " << scalarZ << std::endl;
		for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
		{
			for (int j = 0; j < m_3DPupilPoisionALL[i].size(); j++)
			{
				// �������
				cv::Mat sphere_pos0(3, 1, CV_32FC1);
				sphere_pos0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[0];
				sphere_pos0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[1];
				sphere_pos0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[2];

				// �������
				cv::Mat gaze_dir0(3, 1, CV_32FC1);
				gaze_dir0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[0];
				gaze_dir0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[1];
				gaze_dir0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[2];

				// gaze_point_in_camera
				cv::Mat gaze_point_in_camera = 400 * gaze_dir0 + sphere_pos0;
				cv::Mat gaze_point_in_world = m_eye0ToWorld.rowRange(0, 3).colRange(0, 3) * gaze_point_in_camera + m_eye0ToWorld.rowRange(0, 3).colRange(3, 4);

				//std::cout << "X: " << m_3DScreenPosionALL[i][j].x << " Y: " << m_3DScreenPosionALL[i][j].y << std::endl;
				//std::cout << "X: " << (gaze_point_in_world.at<float>(0, 0) / gaze_point_in_world.at<float>(2, 0)) * 100 * scalarZ << " Y: " << -(gaze_point_in_world.at<float>(1, 0) / gaze_point_in_world.at<float>(2, 0)) * 100 * scalarZ << std::endl;
				//std::cout << "X: " << gaze_point_in_world.at<float>(0, 0) << " Y: " << -gaze_point_in_world.at<float>(1, 0) << " Z:" << gaze_point_in_world.at<float>(2, 0) << std::endl;



				//���߷������

				cv::Mat world3D = (cv::Mat_<float>(3, 1) << m_3DScreenPosionALL[i][j].x, -m_3DScreenPosionALL[i][j].y, m_3DScreenPosionALL[i][j].z);
				std::cout << "no scalar" << norm(world3D.cross(m_eye0ToWorld.rowRange(0, 3).colRange(0, 3)*gaze_dir0)) << std::endl;


				cv::Mat world3D1 = (cv::Mat_<float>(3, 1) << m_3DScreenPosionALL[i][j].x, -m_3DScreenPosionALL[i][j].y, m_3DScreenPosionALL[i][j].z*scalarZ);
				std::cout << "scalar" << norm(world3D1.cross(m_eye0ToWorld.rowRange(0, 3).colRange(0, 3)*gaze_dir0)) << std::endl;

			}
		}

	}




	return 1 << 15;
}//BA
#endif

double CGazeTrakingM::GetMappingResult9_3D(double threshold)
{
	double scalarZ;
	if (!(m_3DScreenPosionALL.size() == m_3DPupilPoisionALL.size() && m_3DScreenPosionALL.size() != 0))
	{
		return 0;  // ���ݲɼ�������Ҫ��
	}
	cv::Mat ref_points_3d_unscaled(3, m_3DScreenPosionALL.size() * m_3DScreenPosionALL[0].size(), CV_32FC1);
	cv::Mat gaze0_dir(3, m_3DPupilPoisionALL.size()*m_3DPupilPoisionALL[0].size(), CV_32FC1);

	// ������Ļ3�ĵ��ͫ�����������ľ�������
	float* pdata = ref_points_3d_unscaled.ptr<float>(0);
	float* pgaze0 = gaze0_dir.ptr<float>(0);


	for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
	{
		int sampleNumTemp = m_3DPupilPoisionALL[i].size();
		for (int j = 0; j < sampleNumTemp; j++)
		{
			pgaze0[i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[0];
			pgaze0[m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[1];
			pgaze0[2 * m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DPupilPoisionALL[i][j].circle.normal[2];
		}
	}

	for (int i = 0; i < m_3DScreenPosionALL.size(); i++)
	{
		int sampleNumTemp = m_3DScreenPosionALL[i].size();
		for (int j = 0; j < sampleNumTemp; j++)
		{
			pdata[i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].x;
			pdata[m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].y;
			pdata[2 * m_3DPupilPoisionALL.size() * sampleNumTemp + i * sampleNumTemp + j] = m_3DScreenPosionALL[i][j].z;
		}
	}
	//test--��ȡ����
	//std::cout << ref_points_3d_unscaled.t() << std::endl;
	//std::cout << gaze0_dir.t() << std::endl;
	//std::cout << gaze0_dir.col(0).t() * gaze0_dir.col(79) << std::endl;
	//std::cout << gaze0_dir.col(50).t() * gaze0_dir.col(400) << std::endl;
	for (int i = 1; i < m_3DPupilPoisionALL.size(); i++)
	{
		std::cout << "cos thita: " << m_3DPupilPoisionALL[0][0].circle.normal[0] * m_3DPupilPoisionALL[i][0].circle.normal[0] + m_3DPupilPoisionALL[0][0].circle.normal[1] * m_3DPupilPoisionALL[i][0].circle.normal[1] + m_3DPupilPoisionALL[0][0].circle.normal[2] * m_3DPupilPoisionALL[i][0].circle.normal[2] << std::endl;

	}





	double smallest_residual = 1000.0;                 //  �궨����޶�
	std::vector<double> optimalPose;
	// ��ʼ�궨��s�ɱ䣬�൱�ڵ�Ч��ƽ��ǰ���ƶ����������ԭ���Ч�������м�
	for (double s = 0.7; s < 50; s += 0.1)                   // s ʹ�ý����󣬾Ϳ��Բ����ٹ����ˣ�--��Ϊ������Ľ������ֱ�Ӹɵ�z��Ϳ�����
	{
		cv::Mat scalarMatrix = cv::Mat::eye(3, 3, CV_32FC1);
		scalarMatrix.at<float>(1, 1) = -1;
		scalarMatrix.at<float>(2, 2) = s;

		cv::Mat ref_points_3d = scalarMatrix * ref_points_3d_unscaled;
		//����Ļ�ο����100 
		//for (int i = 0; i < ref_points_3d.cols; i++)
		//{
		//	float tempnorm = norm(ref_points_3d.rowRange(0, 3).colRange(i, i + 1));
		//	ref_points_3d.rowRange(0, 3).colRange(i, i + 1) = 100.0 * ref_points_3d.rowRange(0, 3).colRange(i, i + 1) / tempnorm;
		//}


		// 1����ʼ�������ƣ�ƽ�ƣ�����ͫ��ȷ������ת���ݸ���任����ȷ����
		cv::Mat initial_translation0 = (cv::Mat_<float>(3, 1) << -30, 0, 0);      // ����x��Ϊ��
		

		cv::Mat H = Find_rigid_transform(gaze0_dir * 100, ref_points_3d);
		//std::cout << "H ��ʼ���ƣ� " << H << std::endl;
		cv::Mat initial_R0 = H.rowRange(0, 3).colRange(0, 3);
		cv::Mat rotation_angle_axis;
		Rodrigues(initial_R0, rotation_angle_axis);

		// ǰ������R,T ��������ϵ����������ϵ�ı任 ---- Ϊ��BA�㷨�ķ��㣬����任Ϊ��������ϵ����������ϵ
		rotation_angle_axis *= -1;
		initial_translation0 *= -1;

		std::vector<Observer> observers;
		Observer observer;
		observer.fix_rotation = false;
		//observer.fix_translation = true;
		observer.fix_translation = false;

		observer.pose.push_back((double)rotation_angle_axis.at<float>(0, 0));
		observer.pose.push_back((double)rotation_angle_axis.at<float>(1, 0));
		observer.pose.push_back((double)rotation_angle_axis.at<float>(2, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(0, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(1, 0));
		observer.pose.push_back((double)initial_translation0.at<float>(2, 0));

		for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
		{
			for (int j = 0; j < m_3DPupilPoisionALL[i].size(); j++)
			{
				Eigen::Matrix<double, 3, 1> temp;
				temp[0] = (double)m_3DPupilPoisionALL[i][j].circle.normal[0];
				temp[1] = (double)m_3DPupilPoisionALL[i][j].circle.normal[1];
				temp[2] = (double)m_3DPupilPoisionALL[i][j].circle.normal[2];

				observer.observations.push_back(temp);
			}

		}
		observers.push_back(observer);

		std::vector<Eigen::Matrix<double, 3, 1>> points;
		for (int i = 0; i < ref_points_3d.cols; i++)
		{
			Eigen::Matrix<double, 3, 1> temp;
			temp[0] = (double)ref_points_3d.at<float>(0, i);
			temp[1] = (double)ref_points_3d.at<float>(1, i);
			temp[2] = (double)ref_points_3d.at<float>(2, i);

			points.push_back(temp);
		}

		double final_cost = bundleAdjustCalibration(observers, points, true);


		if (final_cost == -1)
		{
			continue;
		}
		if (final_cost <= smallest_residual)
		{
			smallest_residual = final_cost;
			//std::cout << "bundleAdjustCalibration--final_cost: " << final_cost << std::endl;
			//std::cout << "bundleAdjustCalibration--s: " << s << std::endl;
			//std::cout << "bundleAdjustCalibration--oritation: " << observers[0].pose[0] << " " << observers[0].pose[1] << " " << observers[0].pose[2] << std::endl;
			//std::cout << "bundleAdjustCalibration--tanslation: " << observers[0].pose[3] <<" "<<observers[0].pose[4] <<" "<<observers[0].pose[5] << std::endl;
			// ������תƽ��
			optimalPose = observers[0].pose;
			scalarZ = s;
		}

	}
	
	if (optimalPose.size() != 6)
	{
		return 0;   //�궨���ɹ�
	}

	cv::Mat rotation_angle_axis_eye0_to_world(3, 1, CV_32FC1);

	rotation_angle_axis_eye0_to_world.at<float>(0, 0) = -1 * optimalPose[0];
	rotation_angle_axis_eye0_to_world.at<float>(1, 0) = -1 * optimalPose[1];
	rotation_angle_axis_eye0_to_world.at<float>(2, 0) = -1 * optimalPose[2];

	cv::Mat translation_eye0_to_world(3, 1, CV_32FC1);
	translation_eye0_to_world.at<float>(0, 0) = -1 * optimalPose[3];
	translation_eye0_to_world.at<float>(1, 0) = -1 * optimalPose[4];
	translation_eye0_to_world.at<float>(2, 0) = -1 * optimalPose[5];

	cv::Mat R_eye0_to_world(3, 3, CV_32FC1);
	cv::Rodrigues(rotation_angle_axis_eye0_to_world, R_eye0_to_world);

	// ������Ļ�㵽���ߵľ��룬���Ա궨�Ƿ�׼ȷ��

	// �����������ϵ����������ϵ��R,T

	cv::Mat sphere_pos0(3, 1, CV_32FC1);
	sphere_pos0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[0];
	sphere_pos0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[1];
	sphere_pos0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[m_3DPupilPoisionALL.size() - 1][m_3DPupilPoisionALL[0].size() - 1].sphere.center[2];

	cv::Mat translation_camera_to_world0 = translation_eye0_to_world - R_eye0_to_world * sphere_pos0;


	R_eye0_to_world.copyTo(m_eye0ToWorld.rowRange(0, 3).colRange(0, 3));
	translation_camera_to_world0.copyTo(m_eye0ToWorld.rowRange(0, 3).colRange(3, 4));

	std::cout << "bundleAdjustCalibration--smallest_residual: " << smallest_residual << std::endl;
	std::cout << "Tew: " << translation_eye0_to_world << std::endl;
	std::cout << "z axis scalar�� " << scalarZ << std::endl;
	std::cout << "m_eye0ToWorld" << m_eye0ToWorld << std::endl;
	

	// test	
	{
		std::cout << "scalarZ: " << scalarZ << std::endl;

		// ��������ϵ�е���ƽ���ʾ
		std::vector<cv::Point3f> screenPosition;
		for (int i = 0; i < m_3DScreenPosionALL.size(); i++)
		{
			cv::Point3f tempPoint;
			cv::Mat tempPointm(4, 1, CV_32FC1);
			tempPointm.at<float>(0, 0) = m_3DScreenPosionALL[i][0].x;
			tempPointm.at<float>(1, 0) = -m_3DScreenPosionALL[i][0].y;
			tempPointm.at<float>(2, 0) = m_3DScreenPosionALL[i][0].z * scalarZ;
			tempPointm.at<float>(3, 0) = 1.0;

			std::cout << "depart calc: " << R_eye0_to_world.inv()*(tempPointm.rowRange(0, 3).colRange(0, 1) - translation_camera_to_world0)<<std::endl;
			tempPointm = m_eye0ToWorld.inv()*tempPointm;
			std::cout << "all calc: " << tempPointm << std::endl;




			tempPoint.x = tempPointm.at<float>(0, 0) / tempPointm.at<float>(3, 0);
			tempPoint.y = tempPointm.at<float>(1, 0) / tempPointm.at<float>(3, 0);
			tempPoint.z = tempPointm.at<float>(2, 0) / tempPointm.at<float>(3, 0);
			std::cout << "all calc /de: " << tempPoint.x << " " << tempPoint.y << " " << tempPoint.z<< std::endl;






			screenPosition.push_back(tempPoint);
		}

		cv::Mat plane;
		FP_analytic2(screenPosition, plane);
		std::cout << "plane equation: " << plane << std::endl;
		m_planeInCamera = plane;



		for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
		{
			for (int j = 0; j < 1/*m_3DPupilPoisionALL[i].size()*/; j++)
			{
				// �������
				cv::Mat sphere_pos0(3, 1, CV_32FC1);
				sphere_pos0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[0];
				sphere_pos0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[1];
				sphere_pos0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[2];

				// �������
				cv::Mat gaze_dir0(3, 1, CV_32FC1);
				gaze_dir0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[0];
				gaze_dir0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[1];
				gaze_dir0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[2];

				cv::Point3f plp;
				PLP_analytic(plane, sphere_pos0, sphere_pos0 + gaze_dir0, plp);
				
				cv::Mat pointInCamera(4, 1, CV_32FC1);
				pointInCamera.at<float>(0, 0) = plp.x;
				pointInCamera.at<float>(1, 0) = plp.y;
				pointInCamera.at<float>(2, 0) = plp.z;
				pointInCamera.at<float>(3, 0) = 1;

				std::cout << "stdard: " << m_3DScreenPosionALL[i][j].x << " Y: " << -m_3DScreenPosionALL[i][j].y << std::endl;
				std::cout << "screen Point: " << (m_eye0ToWorld * pointInCamera).t() << std::endl;

			}
		}

	}
#if 0


	// test	
	{
		std::cout << "scalarZ: " << scalarZ << std::endl;
		for (int i = 0; i < m_3DPupilPoisionALL.size(); i++)
		{
			for (int j = 0; j < m_3DPupilPoisionALL[i].size(); j++)
			{
				// �������
				cv::Mat sphere_pos0(3, 1, CV_32FC1);
				sphere_pos0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[0];
				sphere_pos0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[1];
				sphere_pos0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].sphere.center[2];

				// �������
				cv::Mat gaze_dir0(3, 1, CV_32FC1);
				gaze_dir0.at<float>(0, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[0];
				gaze_dir0.at<float>(1, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[1];
				gaze_dir0.at<float>(2, 0) = (float)m_3DPupilPoisionALL[i][j].circle.normal[2];

				// gaze_point_in_camera
				cv::Mat tempGaze = m_eye0ToWorld.rowRange(0, 3).colRange(0, 3) * gaze_dir0;
				cv::Mat tempTcw = m_eye0ToWorld.rowRange(0, 3).colRange(0, 3) * sphere_pos0 + m_eye0ToWorld.rowRange(0, 3).colRange(3, 4);

				float a = tempGaze.at<float>(0, 0) * tempGaze.at<float>(0, 0) + tempGaze.at<float>(1, 0)* tempGaze.at<float>(1, 0) + tempGaze.at<float>(2, 0)*tempGaze.at<float>(2, 0);
				float b = 2 * (tempGaze.at<float>(0, 0) * tempTcw.at<float>(0, 0) + tempGaze.at<float>(1, 0) * tempTcw.at<float>(1, 0) + tempGaze.at<float>(2, 0)*tempTcw.at<float>(2, 0));
				float c = tempTcw.at<float>(0, 0) * tempTcw.at<float>(0, 0) + tempTcw.at<float>(1, 0)* tempTcw.at<float>(1, 0) + tempTcw.at<float>(2, 0)*tempTcw.at<float>(2, 0) - 10000;
				float delta = b * b - 4 * a * c;
				if (delta < 0)
				{
					system("pause");
					continue;
				}
				assert(-b - sqrt(delta) > 0);

				float x = (-b + sqrt(delta)) / (2 * a);
				std::cout << "gaze scalar sq: " << x << std::endl;

								
				



				//���߷������
				cv::Mat por = tempGaze*x + tempTcw;
				
				std::cout << "gaze_point_in_world_sphere��" << por.t() << std::endl;
				std::cout << "groudtruth: X: " << m_3DScreenPosionALL[i][j].x << " Y: " << m_3DScreenPosionALL[i][j].y << " Z: " << m_3DScreenPosionALL[i][j].z << std::endl;
				std::cout << "gaze_point_in_world_sphere��X:" << por.at<float>(0, 0) / por.at<float>(2, 0) * 100 << " Y: "<<-por.at<float>(1, 0) / por.at<float>(2, 0) * 100<<" Z:" <<100<< std::endl;
				//std::cout << "no scalar�� " << norm(world3D.cross(gaze_point_in_world)) << std::endl;




			}
		}

	}
	
#endif
	
	return scalarZ;
}



bool CGazeTrakingM::onlineCalibrationIsOK()
{
	m_isOnLineCalibration = true;
	return true;
}


// �����ӵ�
int CGazeTrakingM::CalcGazePointPosition()
{
	//   Xs = a0 + a1xe + a2ye + a3xeye + a4xe^2 + a5ye^2;
	//   Ys = a6 + a7xe + a8ye + a9xeye + a10xe^2 + a11ye^2;

	singleeyefitter::Detector2DProperties  props;
	props.intensity_range = 23;                                // ��ֵ������ֵ������С��һ��ʹ��
	props.blur_size = 5;                                       // ��ֵ�˲�����Ĵ�С
	props.canny_treshold = 160;                                // canny ��ֵ1
	props.canny_ration = 2;                                    // ��canny_treshold һ�����canny��ֵ2
	props.canny_aperture = 5;                                  // canny�˲�����Ĵ�С
	props.pupil_size_max = 200;                                // ͫ�׳������ֵ��ֱ����
	props.pupil_size_min = 30;                                 // ͫ�׳�����Сֵ��ֱ����
	props.strong_perimeter_ratio_range_min = 0.6;              // 
	props.strong_perimeter_ratio_range_max = 1.1;
	props.strong_area_ratio_range_min = 0.8;                   // ������͹�����/������ϵ���Բ�����Сֵ
	props.strong_area_ratio_range_max = 1.1;
	props.contour_size_min = 5;                                // ��������С����
	props.ellipse_roundness_ratio = 0.09;                      // ����ȳ������Сֵ
	props.initial_ellipse_fit_treshhold = 4.3;                 // �����Բ�㵽��Բ�ı߽����ķ�������ֵ
	props.final_perimeter_ratio_range_min = 0.5;               // �����жϣ����ԭʼ�߽��/�����Բ���ܳ�
	props.final_perimeter_ratio_range_max = 1.0;
	props.ellipse_true_support_min_dist = 3.0;                 // ԭʼ�����㵽�����Բ�߽�ľ�����ֵ��С�ڴ�ֵ����Ϊ��OK��
	try
	{

		box2d pupilL;
		pupilL.center.x = -1;
		pupilL.center.y = -1;
	    

		cv::Mat image_temp;
		cv::resize(m_imageL, image_temp, cv::Size(320, 240));
		
		cv::Mat color_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::cvtColor(image_temp, color_image, cv::COLOR_GRAY2RGB);
		cv::Mat debug_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Rect roi;
		detect2d.coarse_center1(image_temp, 20, 35, roi);
		std::shared_ptr<Detector2DResult> presultPupil = detect2d.detect(props, image_temp, color_image, debug_image, roi, true, false);

		m_PupilConfidence = presultPupil->confidence;
		m_PupilRadius = presultPupil->ellipse.major_radius;
		m_BlinkDection.PushPupilDetectConfidence(m_PupilConfidence);

		if (fabs(presultPupil->confidence) > 0.2)
		{
			pupilL.center.x = presultPupil->ellipse.center[0];
			pupilL.center.y = presultPupil->ellipse.center[1];
		}
		else
		{
			return 0;
		}




#if 1
		// ����ƫ��
		cv::Point2f gazePoint;

		if (pupilL.center.x < 0 || pupilL.center.y < 0)
		{

			return 0;
		}
		gazePoint.x = m_mapping[0] + m_mapping[1] * pupilL.center.x + m_mapping[2] * pupilL.center.y + m_mapping[3] * pupilL.center.x * pupilL.center.y + m_mapping[4] * pupilL.center.x * pupilL.center.x + m_mapping[5] * pupilL.center.y * pupilL.center.y;

		gazePoint.y = m_mapping[6] + m_mapping[7] * pupilL.center.x + m_mapping[8] * pupilL.center.y + m_mapping[9] * pupilL.center.x * pupilL.center.y + m_mapping[10] * pupilL.center.x * pupilL.center.x + m_mapping[11] * pupilL.center.y * pupilL.center.y;


		if (gazePoint.x < 0)
		{
			gazePoint.x = 0;
		}
		if (gazePoint.y < 0)
		{
			gazePoint.y = 0;
		}
		if (gazePoint.x > 1280)
		{
			gazePoint.x = 1280;
		}
		if (gazePoint.y > 1440)
		{
			gazePoint.y = 1440;
		}

		int a;
		unsigned int pointx = (unsigned int)gazePoint.x;
		unsigned int pointy = (unsigned int)gazePoint.y;

		a = pointx << 16;
		a = a + pointy;


		return a;
#endif
		//  ����
		//cvReleaseImage(&imageL);
		////cvReleaseImage(&imageR);
		//return 0;
		//
	}
	catch (...)
	{
		cv::imwrite("error ͼƬ.bmp", m_imageL);
        return 0;
	}
	return 0;
}

int CGazeTrakingM::CalcGazePointPosition3D(int *gazePoint)
{
	//   Xs = a0 + a1xe + a2ye + a3xeye + a4xe^2 + a5ye^2;
	//   Ys = a6 + a7xe + a8ye + a9xeye + a10xe^2 + a11ye^2;

	singleeyefitter::Detector2DProperties  props;
	props.intensity_range = 23;                                // ��ֵ������ֵ������С��һ��ʹ��
	props.blur_size = 5;                                       // ��ֵ�˲�����Ĵ�С
	props.canny_treshold = 160;                                // canny ��ֵ1
	props.canny_ration = 2;                                    // ��canny_treshold һ�����canny��ֵ2
	props.canny_aperture = 5;                                  // canny�˲�����Ĵ�С
	props.pupil_size_max = 200;                                // ͫ�׳������ֵ��ֱ����
	props.pupil_size_min = 30;                                 // ͫ�׳�����Сֵ��ֱ����
	props.strong_perimeter_ratio_range_min = 0.6;              // 
	props.strong_perimeter_ratio_range_max = 1.1;
	props.strong_area_ratio_range_min = 0.8;                   // ������͹�����/������ϵ���Բ�����Сֵ
	props.strong_area_ratio_range_max = 1.1;
	props.contour_size_min = 5;                                // ��������С����
	props.ellipse_roundness_ratio = 0.09;                      // ����ȳ������Сֵ
	props.initial_ellipse_fit_treshhold = 4.3;                 // �����Բ�㵽��Բ�ı߽����ķ�������ֵ
	props.final_perimeter_ratio_range_min = 0.5;               // �����жϣ����ԭʼ�߽��/�����Բ���ܳ�
	props.final_perimeter_ratio_range_max = 1.0;
	props.ellipse_true_support_min_dist = 3.0;                 // ԭʼ�����㵽�����Բ�߽�ľ�����ֵ��С�ڴ�ֵ����Ϊ��OK��
	try
	{

		box2d pupilL;
		pupilL.center.x = -1;
		pupilL.center.y = -1;


		cv::Mat image_temp;
		cv::resize(m_imageL, image_temp, cv::Size(640, 480));

		cv::Mat color_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::cvtColor(image_temp, color_image, cv::COLOR_GRAY2RGB);
		cv::Mat debug_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Rect roi;
		detect2d.coarse_center1(image_temp, 40, 60, roi);
		std::shared_ptr<Detector2DResult> presultPupil = detect2d.detect(props, image_temp, color_image, debug_image, roi, true, true); //false ��������ʱ��Ϊfalse

		m_PupilConfidence = presultPupil->confidence;
		m_PupilRadius = presultPupil->ellipse.major_radius;
		m_BlinkDection.PushPupilDetectConfidence(m_PupilConfidence);
		if (presultPupil->confidence < 0.1)
		{
			return 0;
		}

		singleeyefitter::Detector3DProperties detectProperties3D;
		detectProperties3D.model_sensitivity = 0.997;
		Detector3DResult result3D = mEyeModel.updateAndDetect(presultPupil, detectProperties3D, false);

		std::cout << "CalcGazePointPosition3D-*-result3D.confidence:" << result3D.confidence << std::endl;
		if (fabs(result3D.confidence) > 0.4)
		{
			pupilL.center.x = presultPupil->ellipse.center[0];
			pupilL.center.y = presultPupil->ellipse.center[1];
		}
		else
		{
			return 0;
		}


#if 1    //show debug image  y δ�ᷴ
		cv::Mat showImage;
		cv::cvtColor(image_temp, showImage, cv::COLOR_GRAY2RGB);

		// 3DԲ����ͼ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.ellipse.center[0] + 310), (int)(result3D.ellipse.center[1] + 260)), cv::Size((int)result3D.ellipse.major_radius, (int)result3D.ellipse.minor_radius), result3D.ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(255, 255, 255));
		// 2D��ƽ���⵽����Բͫ��
		cv::ellipse(showImage, cv::Point((int)(presultPupil->ellipse.center[0] + 310), (int)(presultPupil->ellipse.center[1] + 260)), cv::Size((int)presultPupil->ellipse.major_radius, (int)presultPupil->ellipse.minor_radius), presultPupil->ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(0, 0, 255));
		// 3D ������2D��ƽ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 310), (int)(result3D.projectedSphere.center[1] + 260)), cv::Size(2, 2), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
		// ����
		cv::line(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 310), (int)(result3D.projectedSphere.center[1] + 260)), cv::Point((int)(result3D.ellipse.center[0] + 320), (int)( result3D.ellipse.center[1] + 240)), cv::Scalar(0, 0, 255), 2);

		cv::imshow("live-debug", showImage);



#endif

#if 1
		// �������
		cv::Mat sphere_pos0(3, 1, CV_32FC1);
		sphere_pos0.at<float>(0, 0) = (float)result3D.sphere.center[0];
		sphere_pos0.at<float>(1, 0) = (float)result3D.sphere.center[1];
		sphere_pos0.at<float>(2, 0) = (float)result3D.sphere.center[2];

		// �������
		cv::Mat gaze_dir0(3, 1, CV_32FC1);
		gaze_dir0.at<float>(0, 0) = (float)result3D.circle.normal[0];
		gaze_dir0.at<float>(1, 0) = (float)result3D.circle.normal[1];
		gaze_dir0.at<float>(2, 0) = (float)result3D.circle.normal[2];


		
		return 1;
#endif

	}
	catch (...)
	{
		cv::imwrite("error ͼƬ.bmp", m_imageL);
		return 0;
	}
	return 0;
}

int CGazeTrakingM::CalcGazePointPosition3D(Detector3DResult &result3D, cv::Mat& gaze_img, cv::Mat& pupil_img)
{
	//   Xs = a0 + a1xe + a2ye + a3xeye + a4xe^2 + a5ye^2;
	//   Ys = a6 + a7xe + a8ye + a9xeye + a10xe^2 + a11ye^2;

	singleeyefitter::Detector2DProperties  props;
	props.intensity_range = 23;                                // ��ֵ������ֵ������С��һ��ʹ��
	props.blur_size = 15;                                       // ��ֵ�˲�����Ĵ�С
	props.canny_treshold = 160;                                // canny ��ֵ1
	props.canny_ration = 2;                                    // ��canny_treshold һ�����canny��ֵ2
	props.canny_aperture = 5;                                  // canny�˲�����Ĵ�С
	props.pupil_size_max = 250;                                // ͫ�׳������ֵ��ֱ����
	props.pupil_size_min = 50;                                 // ͫ�׳�����Сֵ��ֱ����
	props.strong_perimeter_ratio_range_min = 0.6;              // 
	props.strong_perimeter_ratio_range_max = 1.1;
	props.strong_area_ratio_range_min = 0.8;                   // ������͹�����/������ϵ���Բ�����Сֵ
	props.strong_area_ratio_range_max = 1.1;
	props.contour_size_min = 40;                                // ��������С����
	props.ellipse_roundness_ratio = 0.6;                      // ����ȳ������Сֵ
	props.initial_ellipse_fit_treshhold = 4.3;                 // �����Բ�㵽��Բ�ı߽����ķ�������ֵ
	props.final_perimeter_ratio_range_min = 0.25;               // �����жϣ����ԭʼ�߽��/�����Բ���ܳ�
	props.final_perimeter_ratio_range_max = 1.0;
	props.ellipse_true_support_min_dist = 3.0;                 // ԭʼ�����㵽�����Բ�߽�ľ�����ֵ��С�ڴ�ֵ����Ϊ��OK��
	try
	{

		box2d pupilL;
		pupilL.center.x = -1;
		pupilL.center.y = -1;


		cv::Mat image_temp;
		cv::resize(m_imageL, image_temp, cv::Size(640, 480));

		cv::Mat color_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::cvtColor(image_temp, color_image, cv::COLOR_GRAY2RGB);
		cv::Mat debug_image = cv::Mat(image_temp.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Rect roi;
		detect2d.coarse_center1(image_temp, 60, 160, roi);
		std::shared_ptr<Detector2DResult> presultPupil = detect2d.detect(props, image_temp, color_image, debug_image, roi, true, true); //false ��������ʱ��Ϊfalse
		pupil_img = color_image;
		cv::ellipse(pupil_img, cv::Point((int)(presultPupil->ellipse.center[0]), (int)(presultPupil->ellipse.center[1])), cv::Size((int)presultPupil->ellipse.major_radius, (int)presultPupil->ellipse.minor_radius), presultPupil->ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(0, 0, 255), 3);
		m_PupilConfidence = presultPupil->confidence;
		m_PupilRadius = presultPupil->ellipse.major_radius;
		m_BlinkDection.PushPupilDetectConfidence(m_PupilConfidence);
		if (presultPupil->confidence < 0.1)
		{
			return 0;
		}

		singleeyefitter::Detector3DProperties detectProperties3D;
		detectProperties3D.model_sensitivity = 0.997;
		result3D = mEyeModel.updateAndDetect(presultPupil, detectProperties3D, true);

		//std::cout << "CalcGazePointPosition3D-*-result3D.confidence:" << result3D.confidence << std::endl;
		if (fabs(result3D.confidence) > 0.2)
		{
			pupilL.center.x = presultPupil->ellipse.center[0];
			pupilL.center.y = presultPupil->ellipse.center[1];
		}
		else
		{
			return 0;
		}


#if 1    //show debug image  y δ�ᷴ
		cv::Mat showImage;
		cv::cvtColor(image_temp, showImage, cv::COLOR_GRAY2RGB);

		// 3DԲ����ͼ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.ellipse.center[0] + 310), (int)(result3D.ellipse.center[1] + 260)), cv::Size((int)result3D.ellipse.major_radius, (int)result3D.ellipse.minor_radius), result3D.ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(255, 255, 255));
		// 2D��ƽ���⵽����Բͫ��
		cv::ellipse(showImage, cv::Point((int)(presultPupil->ellipse.center[0] + 310), (int)(presultPupil->ellipse.center[1] + 260)), cv::Size((int)presultPupil->ellipse.major_radius, (int)presultPupil->ellipse.minor_radius), presultPupil->ellipse.angle / 3.14 * 180, 0, 360, cv::Scalar(0, 0, 255));
		// 3D ������2D��ƽ���ϵ�ͶӰ
		cv::ellipse(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 310), (int)(result3D.projectedSphere.center[1] + 260)), cv::Size(result3D.projectedSphere.major_radius, result3D.projectedSphere.minor_radius), result3D.projectedSphere.angle / 3.14 * 180, 0, 360, cv::Scalar(255, 0, 0), 2);
		// ����
		cv::line(showImage, cv::Point((int)(result3D.projectedSphere.center[0] + 310), (int)(result3D.projectedSphere.center[1] + 260)), cv::Point((int)(result3D.ellipse.center[0] + 310), (int)( result3D.ellipse.center[1] + 260)), cv::Scalar(0, 0, 255), 2);

		//cv::imshow("live-debug", showImage);
		//cv::imshow("color-debug", color_image);
		gaze_img = showImage;
		

#endif

#if 1
		// �������
		cv::Mat sphere_pos0(3, 1, CV_32FC1);
		sphere_pos0.at<float>(0, 0) = (float)result3D.sphere.center[0];
		sphere_pos0.at<float>(1, 0) = (float)result3D.sphere.center[1];
		sphere_pos0.at<float>(2, 0) = (float)result3D.sphere.center[2];

		// �������
		cv::Mat gaze_dir0(3, 1, CV_32FC1);
		gaze_dir0.at<float>(0, 0) = (float)result3D.circle.normal[0];
		gaze_dir0.at<float>(1, 0) = (float)result3D.circle.normal[1];
		gaze_dir0.at<float>(2, 0) = (float)result3D.circle.normal[2];


		
		return 1;
#endif

	}
	catch (...)
	{
		cv::imwrite("error ͼƬ.bmp", m_imageL);
		return 0;
	}
	return 0;
}

double CGazeTrakingM::GetPupilRadius()
{
	if (m_PupilConfidence < 0.1)
	{
		return 0;
	}
	else
	{
		return m_PupilRadius;
	}
}



double CGazeTrakingM::GetPupilConfidence()
{
	return m_PupilConfidence;
}

int CGazeTrakingM::GetBlinkDectect()
{
	return  m_BlinkDection.BlinkDetect();
}