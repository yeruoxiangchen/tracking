#pragma once

#include"opencv2/highgui.hpp"
#include"opencv2/core.hpp"
#include"opencv2/imgproc.hpp"
#include"opencv2/xfeatures2d.hpp"
#include"opencv2/calib3d.hpp"
#include"opencv2/video.hpp"
#include"BFC/portable.h"
#include"BFC/bfstream.h"
#include"BFC/stdf.h"
#include"CVX/bfsio.h"
#include<iostream>
#include<time.h>
using namespace std;

#define _BFCS_API
#include"BFC/commands.h"

#include"CVX/gui.h"

#define CVRENDER_STATIC
#include"CVRender/cvrender.h"

//using namespace cv;

#ifndef _STATIC_BEG
#define _STATIC_BEG namespace{
#define _STATIC_END }
#endif _STATIC_BEG


#define _BFCS_API
#include"BFC/commands.h"

#include"CVX/gui.h"

//#define CVRENDER_STATIC
#include"CVRender/cvrender.h"

#include"re3d/base/base.h"
//using re3d::RigidPose;

//using namespace cv;

#ifndef _STATIC_BEG
#define _STATIC_BEG namespace{
#define _STATIC_END }
#endif _STATIC_BEG

#define _VX_BEG(vx) namespace vx{
#define _VX_END() }

#define D_YCBV std::string(R"(D:\YCB_Video_Dataset\)") 


