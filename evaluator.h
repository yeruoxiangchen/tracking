#pragma once

#include"tracker_base.h"
#include"utils.h"
#include"CVX/vis.h"
#include<fstream>

#undef string_t
#include"cvfx_new/json390/nlohmann/json.hpp"
using json = nlohmann::json;

inline std::vector<Pose> read_gt_poses(const std::string& file)
{
	std::ifstream is(file);
	if (!is)
		throw file;

	std::string tstr;
	std::getline(is, tstr); //ignore the first line

	std::vector<Pose> poses;
	Pose p;
	while (is)
	{
		for (int i = 0; i < 9; ++i)
			is >> p.R.val[i];
		for (int i = 0; i < 3; ++i)
			is >> p.t.val[i];
		if (is)
			poses.push_back(p);
	}
	return poses;
}

inline std::map<int, Object_P> read_gt_poses(const std::string& file, const std::string& vid)
{

	json jfile;
	std::ifstream is(file);
	if (!is)
		cout << 1 << endl;

	is >> jfile;
	json frames = jfile[vid];
	//从第一帧获取视频中跟踪物品的个数
	int CntObj = frames[0]["objs"].size();
	map<int, Object_P>objs;
	for (auto& frame : frames)
	{
		for (int i = 0; i < CntObj; i++)
		{
			int id = frame["objs"][i]["model_id"];
			objs[id].model_id = frame["objs"][i]["model_id"];
			Pose pp;
			for (int j = 0; j < 9; j++)
				pp.R.val[j] = frame["objs"][i]["R"][j];
			for (int j = 0; j < 3; j++)
				pp.t.val[j] = frame["objs"][i]["t"][j];
			objs[id].p.push_back(pp);
			objs[id].frame_cnt = frame["fid"];
		}
	}

	return objs;
}
inline int read_frame_num(std::map<int, Object_P>objs)
{
	return (*objs.begin()).second.frame_cnt;
}

inline std::vector<pair<int, string>> read_model(std::map<int, Object_P> Objs)
{
	std::string ycbvModelList[21] = {
	"002_master_chef_can",
	"003_cracker_box",
	"004_sugar_box",
	"005_tomato_soup_can",
	"006_mustard_bottle",
	"007_tuna_fish_can",
	"008_pudding_box",
	"009_gelatin_box",
	"010_potted_meat_can",
	"011_banana",
	"019_pitcher_base",
	"021_bleach_cleanser",
	"024_bowl",
	"025_mug",
	"035_power_drill",
	"036_wood_block",
	"037_scissors",
	"040_large_marker",
	"051_large_clamp",
	"052_extra_large_clamp",
	"061_foam_brick",
	};
	std::vector<pair<int, string>>modelList;
	for (auto obj : Objs)
	{
		modelList.push_back({ obj.first,ycbvModelList[obj.first - 1] });
	}
	return modelList;
}

inline vector<Pose>get_poses(int fid, std::map<int, Object_P>gt_Pose)
{
	vector<Pose>Poses;
	for (auto& obj : gt_Pose)
	{
		Poses.push_back(obj.second.p[fid - 1]);
	}
	return Poses;
}

inline float get_errorR(cv::Matx33f R_gt, cv::Matx33f R_pred) {
	cv::Matx33f tmp = R_pred.t() * R_gt;
	float trace = tmp(0, 0) + tmp(1, 1) + tmp(2, 2);
	return acos((trace - 1) / 2) * 180 / 3.14159265;
}

inline float get_errort(cv::Vec3f t_gt, cv::Vec3f t_pred) {
	float l22 = pow(t_gt[0] - t_pred[0], 2) + pow(t_gt[1] - t_pred[1], 2) + pow(t_gt[2] - t_pred[2], 2);
	return sqrt(l22);
}

inline void printLost(const Matx33f& R, const Vec3f& t, const Matx33f& R0, const Vec3f& t0)
{
	float errR = get_errorR(R, R0);
	float errt = get_errort(t, t0);

	printf("errR=%.2f, errt=%.2f   \n", errR, errt);
}

inline bool isLost(const Matx33f& R, const Vec3f& t, const Matx33f& R0, const Vec3f& t0)
{
	float errR = get_errorR(R, R0);
	float errt = get_errort(t, t0);

	if (errR > 5.f || errt > 50.f)
		//if (errR > 2.f || errt > 20.f)
	{
		return true;
	}

	return false;
}

inline bool isLost(const Pose& pose, const Pose& gt)
{
	return isLost(pose.R, pose.t, gt.R, gt.t);
}

inline Matx33f getRBOT_K()
{
	cv::Matx33f K = cv::Matx33f::zeros();
	K(0, 0) = 650.048;
	K(0, 2) = 324.328;
	K(1, 1) = 647.183;
	K(1, 2) = 257.323;
	K(2, 2) = 1.0f;
	return K;
}




class VisHandler
	:public UpdateHandler
{
public:
	CVRModel _model;
	CVRender _render;
	cv::Matx33f _K;
	Mat     _img;
	std::string _name;
	Mat     resultImg;
public:
	VisHandler(CVRModel& model, const cv::Matx33f& K)
		:_model(model), _render(_model), _K(K)
	{
	}
	void setBgImage(Mat img, std::string name = "vishdl")
	{
		_img = cv::convertBGRChannels(img, 3);
		_name = name;
	}

	void onUpdate(const cv::Matx33f& R, const cv::Vec3f& t, const UpdateHandler::Infos* infos = NULL)
	{
		Mat dimg = _img.clone();
		CVRMats mats;
		mats.mProjection = cvrm::fromK(_K, _img.size(), 1.f, 3000.f);
		mats.mModel = cvrm::fromR33T(R, t);

		auto rr = _render.exec(mats, dimg.size());
		Mat1b mask = getRenderMask(rr.depth);
		/*std::vector<std::vector<Point>> contours;
		cv::findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);*/
		dimg = cv::drawContours(dimg, mask);
		resultImg = dimg;
		imshow(_name, dimg);
	}

	void draw(Mat img, const cv::Matx33f& R, const cv::Vec3f& t)
	{
		setBgImage(img);
		onUpdate(R, t);
	}
};

inline Mat drawPose(std::string wndName, Mat img, CVRModel& model, Matx33f K, Matx33f R, Vec3f t)
{
	VisHandler hdl(model, K);
	hdl.setBgImage(img, wndName);
	hdl.onUpdate(R, t);
	return hdl.resultImg;
}
