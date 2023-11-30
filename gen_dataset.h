#pragma once

#include<fstream>
#include<iomanip>
#include"CVRender/cvrender.h"

#include"eval3dt.h"

typedef nlohmann::ordered_json json_t;

// 1 l 

inline void gen_model_points(const std::string& modelFile, const std::string& tarFile, int maxPoints = 2000)
{
	CVRModel model(modelFile, 0);
	auto& vertices = model.getVertices();
	double step = __max(1.0, double(vertices.size()) / maxPoints);
	FILE* fp = fopen(tarFile.c_str(), "w");
	if (!fp)
		throw "file open failed";
	for (double i = 0; i < (double)vertices.size(); i += step)
	{
		int j = int(i + 0.5);
		if (j < vertices.size())
			fprintf(fp, "%f\t%f\t%f\n", vertices[j][0], vertices[j][1], vertices[j][2]);
	}
	fclose(fp);
}

inline std::vector<float> formatPose(const eval3dt::ObjPose& pose)
{
	std::vector<float> v(12);
	memcpy(&v[0], pose.R.val, sizeof(float) * 9);
	memcpy(&v[9], pose.t.val, sizeof(float) * 3);
	return v;
}


inline std::vector<int> gen_seq_length(int totalFrames)
{
	cv::Range rngs[] = { {10,50}, {50,100}, {100,150}, {150,200} };
	int rngFrames = totalFrames / 4;

	auto addRange = [rngFrames](std::vector<int>& v, cv::Range r) mutable {
		int n = 0;
		rngFrames += (r.start + r.end) / 2;
		for (; n < rngFrames; )
		{
			int d = r.start + rand() % (r.end - r.start);
			v.push_back(d);
			n += d;
		}
	};

	std::vector<int> v;
	for (auto& r : rngs)
		addRange(v, r);
	return v;
}
inline std::vector<int>  gen_cyc_fid(int idMin, int idMax)
{
	std::vector<int> v;
	v.reserve((idMax - idMin + 1) * 2);
	for (int i = idMin; i <= idMax; ++i)
		v.push_back(i);
	for (int i = idMax - 1; i >= idMin + 1; --i)
		v.push_back(i);
	return v;
}

inline std::vector<int> get_frames(const std::vector<int>& cyc_fids, int& istart, int nFrames, int istep)
{
	std::vector<int> v;
	v.reserve(nFrames);

	int i = istart;
	while (v.size() < nFrames)
	{
		i = int((i + istep) % cyc_fids.size());
		v.push_back(cyc_fids[i]);
	}
	istart = i;

	return v;
}

#define D_RBOT R"(F:\store\datasets\RBOT_dataset\)"

