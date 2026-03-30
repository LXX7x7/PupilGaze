#ifndef __H_BLINK_DETECTION_H__
#define __H_BLINK_DETECTION_H__
#include <vector>

class BlinkDetection
{
public:
	//*******************************************************
	//功    能： 【初始化构造函数】
	//输入参数： 
	//    [in] detectListLen                        连续检测瞳孔的图片数量,最好为偶数，如果奇数内部会-1；
	//    [in] onset_confidence_threshold           睁眼动作的阈值，此值越高，检测睁眼动作越严格；
	//    [in] offset_confidence_threshold          闭眼动作的阈值，此值越高，检测闭眼动作越严格；
	//返 回 值：
	//
	//修改日志：
	//*******************************************************
	BlinkDetection(int detectListLen = 6, double onset_confidence_threshold = 0.5, double offset_confidence_threshold = 0.5);

	~BlinkDetection();

	//*******************************************************
	//功    能： 【设置参数，方便当此对象作为其他类成员变量，初始化后参数还可以再进行设置】
	//输入参数： 
	//    [in] detectListLen                        连续检测瞳孔的图片数量,最好为偶数，如果奇数内部会-1；
	//    [in] onset_confidence_threshold           睁眼动作的阈值，此值越高，检测睁眼动作越严格；
	//    [in] offset_confidence_threshold          闭眼动作的阈值，此值越高，检测闭眼动作越严格；
	//返 回 值：
	//
	//修改日志：
	//*******************************************************
	void SetParams(int detectListLen = 6, double onset_confidence_threshold = 0.5, double offset_confidence_threshold = 0.5);

	//*******************************************************
	//功    能： 【输入检测瞳孔的置信度】
	//输入参数： 
    //
	//返 回 值：
	//
	//修改日志：
	//*******************************************************
	void PushPupilDetectConfidence(double confidence);

	//*******************************************************
	//功    能： 【眨眼检测】
	//输入参数： 
	//    
	//返 回 值：
	//     0：   代表维持原状（睁眼或闭眼）
	//     -1:   睁眼的动作
	//      1:   闭眼的动作
	//修改日志：
	//*******************************************************
	int BlinkDetect();



private:
	std::vector<double> m_vPupilConfidence;
	int                 m_detectListLen;

	double              m_onset_confidence_threshold;     // 睁眼
	double              m_offset_confidence_threshold;    // 闭眼
};
#endif