#include "Blink_Detection.hpp"


BlinkDetection::BlinkDetection(int detectListLen, double onset_confidence_threshold, double offset_confidence_threshold)
{
	m_detectListLen = detectListLen & -2;
	m_offset_confidence_threshold = offset_confidence_threshold;
	m_onset_confidence_threshold = onset_confidence_threshold;
}

BlinkDetection::~BlinkDetection()
{

}

void BlinkDetection::SetParams(int detectListLen /*= 7*/, double onset_confidence_threshold /*= 0.5*/, double offset_confidence_threshold /*= 0.5*/)
{
	m_detectListLen = detectListLen & -2;
	m_offset_confidence_threshold = offset_confidence_threshold;
	m_onset_confidence_threshold = onset_confidence_threshold;
}

void BlinkDetection::PushPupilDetectConfidence(double confidence)
{
	if (m_vPupilConfidence.size() < m_detectListLen)
	{

	}
	else
	{
		std::vector<double>::iterator it = m_vPupilConfidence.begin();
		m_vPupilConfidence.erase(it);
		
	}
	m_vPupilConfidence.push_back(confidence);
}

int BlinkDetection::BlinkDetect()
{
	if (m_vPupilConfidence.size() < m_detectListLen)
	{
		return 0;      //状态未知
	}	
	
	int beforeV = int(m_detectListLen / 2);
	double beforeRespose = 0;
	for (int i = 0; i < beforeV; i++)
	{
		beforeRespose += m_vPupilConfidence[i];
	}
	beforeRespose = beforeRespose / m_detectListLen;

	double afterRespose = 0;
	for (int i = beforeV; i < m_detectListLen; i++)
	{
		afterRespose += m_vPupilConfidence[i];
	}
	afterRespose = afterRespose / m_detectListLen;

	// 理想情况下filter_response在闭眼的情况下应该为1；
	double filter_response = (beforeRespose - afterRespose) / 0.45;

	if (filter_response < -m_onset_confidence_threshold)   // 睁眼
	{
		return -1;
	}
	else if (filter_response > m_offset_confidence_threshold)  // 闭眼
	{
		return 1;
	}
	else
	{
		return 0;
	}

}



//int main()
//{
//	BlinkDetection   BD;
//
//	BD.PushPupilDetectConfidence(0.0);
//	BD.PushPupilDetectConfidence(0.0);
//	BD.PushPupilDetectConfidence(0.);
//	BD.PushPupilDetectConfidence(0.9);
//	BD.PushPupilDetectConfidence(0.9);
//	int a = BD.BlinkDetect();
//	BD.PushPupilDetectConfidence(0.9);
//	a = BD.BlinkDetect();
//	BD.PushPupilDetectConfidence(0.0);
//	a = BD.BlinkDetect();
//	BD.PushPupilDetectConfidence(0.0);
//	a = BD.BlinkDetect();
//	BD.PushPupilDetectConfidence(0.);
//	BD.PushPupilDetectConfidence(0.9);
//	BD.PushPupilDetectConfidence(0.9);
//	BD.PushPupilDetectConfidence(0.9);
//	
//	a = BD.BlinkDetect();
//
//}


