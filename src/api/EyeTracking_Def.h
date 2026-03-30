#ifndef __HH_EYE_TRACKING_DEF_HH__
#define __HH_EYE_TRACKING_DEF_HH__

#define DIS_CV                   // 如果定义DIS_CV，则显示图像
#include <opencv2/opencv.hpp>


struct point {int x,y;};
struct pointf {double x,y;};

struct box2d
{
	pointf center;        /* Center of the box.                          */
	point  size;          /* Box width and length.                       */  //max 此处需要修改
	float  angle;         /* Angle between the horizontal axis           */
	                      /* and the first side (i.e. length) in degrees */
};

#endif