

#include"stdafx.h"
using ff::exec;

int main()
{
	exec("trackers.test_accuracy");  //test for a single sequence 
	
	//exec("trackers.test_accuracy_all"); //test for all sequences of RBOT dataset
	//exec("trackers.test_show"); //test for all sequences of RBOT dataset
	//exec("trackers.test_upd"); //test for all sequences of RBOT dataset
	//exec("3dmultiobj_show");
	//class BlockProbability
	//{
	//	Mat1f posProbility;
	//	int Block_Size;
	//public:
	//	float calculateForegroundProbability(const Mat& img, int row, int col)
	//	{
	//		int foreground_pixels = 0;
	//		int half_block_size = Block_Size / 2;

	//		for (int i = row - half_block_size; i <= row + half_block_size; i++)
	//		{
	//			for (int j = col - half_block_size; j <= col + half_block_size; j++)
	//			{
	//				if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
	//				{
	//					// 检查像素是否属于前景（假设前景像素值为255）
	//					if (img.at<uchar>(i, j) > 0)
	//					{
	//						foreground_pixels++;
	//					}
	//				}
	//			}
	//		}

	//		double probability = static_cast<double>(foreground_pixels) / (Block_Size * Block_Size);
	//		return probability;
	//	}

	//	void build(const Mat& img, int blocksize = 16)
	//	{
	//		posProbility = Mat::zeros(img.size(), CV_32F);
	//		Block_Size = blocksize;
	//	}

	//	void update(const Mat& img, float learningrate = 0.2f)
	//	{

	//		for (int row = 0; row < img.rows; row++)
	//		{
	//			for (int col = 0; col < img.cols; col++)
	//			{
	//				float probability = calculateForegroundProbability(img, row, col);
	//				posProbility.at<double>(row, col) = probability * learningrate + posProbility.at<double>(row, col) * (1 - learningrate);
	//			}
	//		}
	//	}

	//	Mat1f getProb()
	//	{
	//		return posProbility;
	//	}
	//};

	//class MaskPredict
	//{
	//	vector<ColorHistogram> _colorHistograms;
	//	BlockProbability _blockProbability;
	//	int modelcnt;
	//	Mat1f Prob;
	//	vector<Point2f>_Centers;
	//	Mat mask;
	//	vector<float>Wmax;
	//	vector<Rect>_rois;
	//	Mat3b curImg;
	//public:
	//	void update(vector<Object>& objs, const Mat3b& img, const vector<Pose>& poses, const Matx33f& K, float learningRate)
	//	{
	//		Prob = Mat::zeros(img.size(), CV_32F);
	//		modelcnt = objs.size();
	//		_colorHistograms.resize(modelcnt);
	//		Wmax.resize(modelcnt);
	//		vector<vector<Point2f>>cPts(modelcnt);
	//		vector<vector<Point>>cpts(modelcnt);
	//		curImg = img;
	//		for (int i = 0; i < modelcnt; i++)
	//		{
	//			auto& obj = objs[i];
	//			auto& pose = poses[i];
	//			auto& templ = obj.templ;
	//			auto modelCenter = obj.model.getCenter();

	//			Projector prj(K, pose.R, pose.t);
	//			int curView = templ._getNearestView(pose.R, pose.t);
	//			if (uint(curView) < templ.views.size())
	//			{
	//				Point2f objCenter = prj(modelCenter);
	//				Wmax[i] = std::max({ img.rows - objCenter.y,img.cols - objCenter.x,objCenter.y,objCenter.x });
	//				_Centers.push_back(objCenter);
	//				auto& view = templ.views[curView];

	//				for (auto& cp : view.contourPoints3d)
	//				{
	//					Point2f c = prj(cp.center);
	//					cPts[i].push_back(c);
	//					cpts[i].push_back(c);
	//				}
	//				Rect_<float> rectf = getBoundingBox2D(cPts[i]);
	//				Rect roi = Rect(rectf);
	//				_rois.push_back(roi);
	//			}
	//		}
	//		mask = Mat::zeros(img.size(), CV_32SC1); // 创建空白掩码图像
	//		for (int i = 0; i < cpts.size(); i++) {
	//			cv::fillPoly(mask, cpts[i], cv::Scalar(255)); // 填充多边形区域
	//		}
	//		_blockProbability.build(mask);//
	//		_blockProbability.update(mask, learningRate);//用前景更新块概率图
	//		for (int i = 0; i < modelcnt; i++)
	//		{
	//			//_colorHistograms[i].update(curImg, _Centers[i], cPts[i], learningRate);
	//			_colorHistograms[i].update(objs[i], img, poses[i], K, learningRate);
	//			std::time_t now = std::time(nullptr);
	//			std::tm* localTime = std::localtime(&now);
	//			char timeStr[100];
	//			std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d_%H-%M-%S", localTime);
	//			std::string outputPath = "D:/YCB_Video_Dataset/midproduct/foreground_prob_" + std::string(timeStr) + std::to_string(i) + ".png";
	//			cv::imwrite(outputPath, _colorHistograms[i].getProb(curImg));
	//		}
	//	}


	//	Mat1f getProb()
	//	{
	//		Prob = Mat::zeros(curImg.size(), CV_32F);
	//		vector<Mat1f>colorProbs;
	//		Mat1f posProb = _blockProbability.getProb();
	//		for (int i = 0; i < modelcnt; i++)
	//		{
	//			colorProbs.push_back({ _colorHistograms[i].getProb(curImg) });
	//		}
	//		std::time_t now = std::time(nullptr);
	//		std::tm* localTime = std::localtime(&now);
	//		char timeStr[100];
	//		std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d_%H-%M-%S", localTime);
	//		std::string outputPath = "D:/YCB_Video_Dataset/midproduct/foreground_prob_" + std::string(timeStr) + ".png";
	//		cv::imwrite(outputPath, posProb);
	//		std::string outputPath1 = "D:/YCB_Video_Dataset/midproduct/colorprob0_" + std::string(timeStr) + ".png";
	//		cv::imwrite(outputPath1, colorProbs[0]);
	//		std::string outputPath2 = "D:/YCB_Video_Dataset/midproduct/colorprob1_" + std::string(timeStr) + ".png";
	//		cv::imwrite(outputPath2, colorProbs[1]);
	//		std::string outputPath3 = "D:/YCB_Video_Dataset/midproduct/colorprob2_" + std::string(timeStr) + ".png";
	//		cv::imwrite(outputPath3, colorProbs[2]);
	//		std::string outputPath4 = "D:/YCB_Video_Dataset/midproduct/colorprob3_" + std::string(timeStr) + ".png";
	//		cv::imwrite(outputPath4, colorProbs[3]);
	//		for (int row = 0; row < curImg.rows; row++)
	//		{
	//			for (int col = 0; col < curImg.cols; col++)
	//			{
	//				float Pcolor = 0.f;
	//				for (int k = 0; k < modelcnt; k++)Pcolor = max(Pcolor, colorProbs[k].at<float>(row, col));
	//				float Ppos = posProb.at<float>(row, col);
	//				float mindist = 1000.f;//640*480这个基本够了
	//				float wmax;
	//				Rect roi;
	//				for (int k = 0; k < modelcnt; k++)
	//				{
	//					Point2f objcenter = _Centers[k];
	//					float curdist = sqrt((objcenter.x - col) * (objcenter.x - col) + (objcenter.y - row) * (objcenter.y - row));
	//					if (curdist < mindist)
	//					{
	//						mindist = curdist;
	//						wmax = Wmax[k];
	//						roi = _rois[k];
	//					}
	//				}
	//				float sigma = (roi.width + roi.height) / 4;
	//				float w = exp(-(mindist * mindist) / (2 * sigma * sigma));
	//				Prob.at<float>(row, col) = w * Pcolor + (1 - w) * Ppos;
	//			}
	//		}
	//		return Prob;
	//	}
	//};

	return 0;
}

 