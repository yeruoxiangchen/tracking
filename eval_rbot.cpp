
#include"stdafx.h"
#include"evaluator.h"

#include"tracker_v1_eccv_final.h"
namespace vx = tracker_v1;

int g_fi = 0;

_STATIC_BEG

std::string  D_RBOT = "D:\\RBOT_dataset\\";

inline Mat1b renderObjMask(CVRModel &model, const Pose& pose, const Matx33f& K, Size dsize)
{
	CVRMats mats;
	mats.mModel = cvrm::fromR33T(pose.R, pose.t);
	mats.mProjection = cvrm::fromK(K,dsize,1,3000);

	CVRender render(model);

	CVRResult rr = render.exec(mats, dsize, CVRM_DEPTH);

	return getRenderMask(rr.depth);
}
class FrameRender
{
	CVRModel  _model;
public:
	FrameRender(CVRModel model)
	{
		_model = model;
	}
	Mat renderResults(const cv::Mat& img, const Matx33f& K, const Pose pose)
	{
		cv::Mat dimg = img.clone();


		{
			CVRMats mats;
			mats.mModel = cvrm::fromR33T(pose.R, pose.t);
			mats.mProjection = cvrm::fromK(K, img.size(), 0.1, 3000);

			//if (_drawContour || _drawBlend)
			{
				CVRModel& m3d = _model;
				CVRender render(m3d);
				auto rr = render.exec(mats, img.size(), CVRM_IMAGE | CVRM_DEPTH, CVRM_DEFAULT, nullptr);
				Mat1b mask = rr.getMaskFromDepth();
				Rect roi = cv::get_mask_roi(DWHS(mask), 127);

				{
					Mat t;
					cv::addWeighted(dimg(roi), 0.6, rr.img(roi), 0.4, 0, t);
					t.copyTo(dimg(roi), mask(roi));
					//rr.img(roi).copyTo(dimg(roi), mask(roi));
				}
				{
					std::vector<std::vector<Point> > cont;
					cv::findContours(mask, cont, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
					drawContours(dimg, cont, -1, Scalar(255, 0, 255), 1, CV_AA);
					//drawContours(dimg, cont, -1, Scalar(255, 0, 0), 2, CV_AA);
				}
			}
		}
		return dimg;
	}
};
static void on_trackers_test_accuracy()
{
	cv::Matx33f K = getRBOT_K();

	std::string dataPath = D_RBOT;
	std::vector<Pose> gtPoses = read_gt_poses(dataPath + "poses_first.txt");

	/*std::vector<std::string> body_names{
		"ape",  "bakingsoda", "benchviseblue", "broccolisoup", "cam",
		"can",  "cat",        "clown",         "cube",         "driller",
		"duck", "eggbox",     "glue",          "iron",         "koalacandy",
		"lamp", "phone",      "squirrel" };*/

	std::string modelName = "can";

	CVRModel model(dataPath + modelName + "/" + modelName + ".obj");
	VisHandler visHdl(model, K);

	std::string imgDir = dataPath + modelName + "/frames/";
	//std::string imgDir = dataPath + modelName + "/mask/";
	auto loadImage = [&imgDir](int fi) {
		auto file = imgDir + ff::StrFormat("a_regular%04d.png", fi);
		//auto file = imgDir + ff::StrFormat("%06d-mask.png", fi);
		//auto file = imgDir + ff::StrFormat("b_dynamiclight%04d.png", fi);
		//auto file = imgDir + ff::StrFormat("c_noisy%04d.png", fi);
		return cv::imread(file, cv::IMREAD_COLOR);
	};
	FrameRender frender(model);
	vx::Tracker tracker;
	tracker.loadModel(model.getFile(),"");

	int start = 0;
	Pose pose = gtPoses[start];

	tracker.reset(loadImage(start), pose, K);

	int nLost = 0, nTotal = 0, totalTime=0;
	const int S = 1;
	for (int fi = start+S; fi <= 1000; fi += S)
	{

		g_fi = fi;
		//tracker.fi = fi;
		cv::Mat3b img = loadImage(fi);
		time_t beg = clock();
		tracker.startUpdate(img, fi);//����reset��֡���ֱ��ͼ����ǰ���ĸ���
		//pose = gtPoses[fi];
		//imshow("curProb", tracker._cur.colorProb);
		//cv::Mat outputImage;
		//tracker._cur.colorProb.convertTo(outputImage, CV_8U, 255.0);
		//imwrite("D:/RBOT_dataset/cube/curProb/1last.png" /*+ ff::StrFormat("/%d-curProb.png", fi)*/, outputImage);
		tracker.update(pose,gtPoses[fi]);

		tracker.endUpdate(pose);
		//imshow("closestProb", tracker._colorHistogram.getProb(loadImage(fi + 1)));
		//tracker._colorHistogram.getProb(loadImage(fi + 1)).convertTo(outputImage, CV_8U, 255.0);
		//imwrite("D:/RBOT_dataset/cube/curProb/2next.png" /*+ ff::StrFormat("/%d-curProb.png", fi)*/, outputImage);
		totalTime += int(clock() - beg);
		printf("fi=%d, time=%dms     \r", fi, int(clock() - beg));
		//Mat dimg = frender.renderResults(img, K, pose);
		//Mat gimg = frender.renderResults(img, K, gtPoses[fi]);
		//cv::imshow("dimg", dimg);
		//cv::imshow("gimg", gimg);
		//waitKey(0);

		if (isLost(pose, gtPoses[fi]))
		{
			printf("E%d",fi);
			printLost(pose.R, pose.t, gtPoses[fi].R, gtPoses[fi].t);
			printf("#%d",fi);
			printLost(gtPoses[fi - S].R, gtPoses[fi - S].t, gtPoses[fi].R, gtPoses[fi].t);
			printf("\n");

			//Mat dimg = frender.renderResults(img, K, pose);
			//Mat gimg = frender.renderResults(img, K, gtPoses[fi - 1]);
			//cv::imshow("dimg", dimg);
			//cv::imshow("gimg", gimg);
			//cv::waitKey(0);
			pose = gtPoses[fi];
			tracker.reset(img, pose, K);
			++nLost;
			
		}
		++nTotal;
		//cv::waitKey(0);
		//imshow("finalProb", tracker._colorHistogram.getProb(loadImage(fi + 1)));
		//tracker._colorHistogram.getProb(loadImage(fi + 1)).convertTo(outputImage, CV_8U, 255.0);
		//imwrite("D:/RBOT_dataset/cube/curProb/3true.png" /*+ ff::StrFormat("/%d-curProb.png", fi)*/, outputImage);
		//if (fi == 417)break;
		
	}
	printf("updatesPerIm=%.2f\n", float(vx::g_totalUpdates) / nTotal);
	printf("acc=%.2f  meanTime=%.2f   \n", 100.f * (nTotal - nLost) / nTotal, float(totalTime)/nTotal);
	
}

CMD_BEG()
CMD0("trackers.test_accuracy", on_trackers_test_accuracy)
CMD_END()



static void on_trackers_test_accuracy_all()
{
	cv::Matx33f K = getRBOT_K();

	std::string dataPath = D_RBOT;
	std::vector<Pose> gtPoses = read_gt_poses(dataPath + "poses_first.txt");

	std::vector<std::string> body_names{
		"ape",  "bakingsoda", "benchviseblue", "broccolisoup", "cam",
		"can",  "cat",        "clown",         "cube",         "driller",
		"duck", "eggbox",     "glue",          "iron",         "koalacandy",
		"lamp", "phone",      "squirrel" };

	struct DModel
	{
		std::string name;
		float       accuracy;
	};

	std::vector<DModel>  accModels;

	uint totalTime = 0, totalFrames = 0;

	for (auto& modelName : body_names)
	{
		vx::Tracker tracker; 

		CVRModel model(dataPath + modelName + "/" + modelName + ".obj");
		VisHandler visHdl(model, K);

		std::string imgDir = dataPath + modelName + "/frames/";
		auto loadImage = [&imgDir](int fi) {
			auto file = imgDir + ff::StrFormat("a_regular%04d.png", fi);
			//auto file = imgDir + ff::StrFormat("b_dynamiclight%04d.png", fi);
			//auto file = imgDir + ff::StrFormat("c_noisy%04d.png", fi);
			return cv::imread(file, cv::IMREAD_COLOR);
		};

		tracker.loadModel(model.getFile(), "");

		int start = 0;
		Pose pose = gtPoses[start];
		tracker.reset(loadImage(start), pose, K);

		printf("\n");
		int nLost = 0, nTotal = 0;
		const int S = 1;
		for (int fi = start + S; fi <= 1000; fi += S)
		{
			g_fi = fi;


			cv::Mat3b img = loadImage(fi);

			time_t beg = clock();

			tracker.startUpdate(img, fi);
			//tracker.update(pose);
			tracker.endUpdate(pose);

			totalTime += uint(clock() - beg);
			++totalFrames;
			printf("fi=%d, time=%dms     \r", fi, int(clock() - beg));

			if (isLost(pose, gtPoses[fi]))
			{
				printf("E%d:", fi);
				printLost(pose.R, pose.t, gtPoses[fi].R, gtPoses[fi].t);

				pose = gtPoses[fi];
				tracker.reset(img, pose, K);
				++nLost;
			}
			++nTotal;
		}

		float acc = 100.f * (nTotal - nLost) / nTotal;
		printf("%s acc=%.2f   \n", modelName.c_str(), acc);
		accModels.push_back({ modelName, acc });
	}

	printf("\n");
	float mean = 0;
	for (auto& v : accModels)
	{
		printf("%s = %.2f\n", v.name.c_str(), v.accuracy);
		mean += v.accuracy;
	}
	mean /= accModels.size();
	printf("updatesPerIm=%.2f\n", float(vx::g_totalUpdates) / totalFrames);
	printf("meanTime=%.2f\n", float(totalTime) / totalFrames);
	printf("mean = %.2f\n", mean);
}

CMD_BEG()
CMD0("trackers.test_accuracy_all", on_trackers_test_accuracy_all)
CMD_END()



static void on_trackers_test_accuracy_show()
{
	cv::Matx33f K = getRBOT_K();

	std::string dataPath = D_RBOT;
	std::vector<Pose> gtPoses = read_gt_poses(dataPath + "poses_first.txt");

	/*std::vector<std::string> body_names{
		"ape",  "bakingsoda", "benchviseblue", "broccolisoup", "cam",
		"can",  "cat",        "clown",         "cube",         "driller",
		"duck", "eggbox",     "glue",          "iron",         "koalacandy",
		"lamp", "phone",      "squirrel" };*/

	std::string modelName = "ape";

	CVRModel model(dataPath + modelName + "/" + modelName + ".obj");
	VisHandler visHdl(model, K);
	FrameRender frender(model);

	std::string imgDir = dataPath + modelName + "/frames/";
	auto loadImage = [&imgDir](int fi) {
		auto file = imgDir + ff::StrFormat("a_regular%04d.png", fi);
		//auto file = imgDir + ff::StrFormat("b_dynamiclight%04d.png", fi);
		//auto file = imgDir + ff::StrFormat("c_noisy%04d.png", fi);
		return cv::imread(file, cv::IMREAD_COLOR);
	};

	vx::Tracker tracker;
	tracker.loadModel(model.getFile(), "");

	int start = 0;
	Pose pose = gtPoses[start];
	tracker.reset(loadImage(start), pose, K);

	int nLost = 0, nTotal = 0, totalTime = 0;
	const int S = 1;
	for (int fi = start + S; fi <= 1000; fi += S)
	{
		g_fi = fi;

		cv::Mat3b img = loadImage(fi);

		time_t beg = clock();
		tracker.startUpdate(img, fi);//����reset��֡���ֱ��ͼ����ǰ���ĸ���

		auto R0 = pose.R;
		//tracker.update(pose);

		tracker.endUpdate(pose);
		totalTime += int(clock() - beg);
		printf("fi=%d, time=%dms     \r", fi, int(clock() - beg));
		Mat dimg = frender.renderResults(img, K, pose);
		Mat gtimg = frender.renderResults(img, K, gtPoses[fi]);
		//cv::namedWindow("dimg", cv::WINDOW_NORMAL);
		cv::imshow("dimg", dimg);
		/*cv::namedWindow("gtimg", cv::WINDOW_NORMAL);
		cv::imshow("gtimg", gtimg);*/
		cv::waitKey(10); // �ȴ������¼�
			// ���� ESC ���˳�ѭ��
		if (waitKey(1) == 27)
			break;
		if (isLost(pose, gtPoses[fi]))
		{

			pose = gtPoses[fi];
			tracker.reset(img, pose, K);
			
		}
	
	}

}

CMD_BEG()
CMD0("trackers.test_show", on_trackers_test_accuracy_show)
CMD_END()

static void on_trackers_test_upd()
{
	cv::Matx33f K = getRBOT_K();

	std::string dataPath = D_RBOT;
	std::vector<Pose> gtPoses = read_gt_poses(dataPath + "poses_first.txt");

	std::vector<std::string> body_names{
		"ape",  "bakingsoda", "benchviseblue", "broccolisoup", "cam",
		"can",  "cat",        "clown",         "cube",         "driller",
		"duck", "eggbox",     "glue",          "iron",         "koalacandy",
		"lamp", "phone",      "squirrel" };

	struct DModel
	{
		std::string name;
		float       accuracy;
	};

	std::vector<DModel>  accModels;

	uint totalTime = 0, totalFrames = 0;

	float meanR = 0.f, meant = 0.f;
	int cnt = 0;
	std::string modelName = "bakingsoda";
	{
		vx::Tracker tracker;

		CVRModel model(dataPath + modelName + "/" + modelName + ".obj");
		VisHandler visHdl(model, K);
		FrameRender frender(model);

		std::string imgDir = dataPath + modelName + "/frames/";
		auto loadImage = [&imgDir](int fi) {
			auto file = imgDir + ff::StrFormat("a_regular%04d.png", fi);
			//auto file = imgDir + ff::StrFormat("b_dynamiclight%04d.png", fi);
			//auto file = imgDir + ff::StrFormat("c_noisy%04d.png", fi);
			return cv::imread(file, cv::IMREAD_COLOR);
		};

		tracker.loadModel(model.getFile(), "");

		int start = 0;
		Pose pose = gtPoses[start];
		tracker.reset(loadImage(start), pose, K);

		printf("\n");
		int nLost = 0, nTotal = 0;
		const int S = 1;
		for (int fi = start + S; fi <= 1000; fi += S)
		{
			g_fi = fi;


			cv::Mat3b img = loadImage(fi);

			time_t beg = clock();

			tracker.startUpdate(img, fi);
			//tracker.update(pose);
			tracker.endUpdate(pose);

			totalTime += uint(clock() - beg);
			++totalFrames;
			printf("fi=%d, time=%dms     \n", fi, int(clock() - beg));
			{
				printf("E%d:", fi);
				printLost(pose.R, pose.t, gtPoses[fi].R, gtPoses[fi].t);
				float errR = get_errorR(pose.R, gtPoses[fi].R);
				float errt = get_errort(pose.t, gtPoses[fi].t);
				meanR += errR;
				meant += errt;
				cnt++;
				pose = gtPoses[fi];
				tracker.reset(img, pose, K);
			}
			printf("fi=%d, time=%dms     \r", fi, int(clock() - beg));
			Mat dimg = frender.renderResults(img, K, pose);
			Mat gtimg = frender.renderResults(img, K, gtPoses[fi]);
			cv::namedWindow("dimg", cv::WINDOW_NORMAL);
			cv::imshow("dimg", dimg);
			cv::namedWindow("gtimg", cv::WINDOW_NORMAL);
			cv::imshow("gtimg", gtimg);
			cv::waitKey(1); // �ȴ������¼�
				// ���� ESC ���˳�ѭ��
			if (waitKey(1) == 27)
				break;
		}

	}
	printf("%f    %f\n", meanR / cnt, meant / cnt);

}

CMD_BEG()
CMD0("trackers.test_upd", on_trackers_test_upd)
CMD_END()

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
std::string ycbvSeqList[] = {
	/*"0005",*/"0048","0049","0050","0051","0052","0053","0054","0055","0056","0057","0058","0059"
};

static void mulitrackers_show()
{
	cv::Matx33f K = cv::Matx33f::zeros();
	K(0, 0) = 1066.778;
	K(0, 2) = 312.9869;
	K(1, 1) = 1067.487;
	K(1, 2) = 241.3109;
	K(2, 2) = 1.0f;
	//����seqlist�е�ÿһ����Ƶ����
	uint totalTime = 0, totalFrames = 0;
	for (auto video : ycbvSeqList)
	{
		printf("video %s is loading...\n", video.c_str());
		std::string dataPath = D_YCBV;
		//gtpose->F:\datasets\YCB_Video_Dataset\evalx\poses_gt.json,һ��model_id��Ӧһ��pose����,�����ڵ�ǰ��Ƶ��ÿһ֡��λ��
		std::map<int, Object_P> gtPoses = read_gt_poses(dataPath + "poses_gt.json", video);
		int framecnt = read_frame_num(gtPoses);
		std::vector<pair<int, string>>ModelList = read_model(gtPoses);//��Ƶ������ģ����
		vx::Tracker tracker;
		std::string imgDir = dataPath + "/data_opt/" + video;
		auto loadImage = [&imgDir](int fi) {
			auto file = imgDir + ff::StrFormat("/%06d-color.png", fi);
			//auto file = imgDir + ff::StrFormat("b_dynamiclight%04d.png", fi);
			//auto file = imgDir + ff::StrFormat("c_noisy%04d.png", fi);
			return cv::imread(file, cv::IMREAD_COLOR);
		};
		//ȷ��Ҫ���ص�ģ�͵��ļ�
		int vi = 1;
		CVRModel models;
		//ModelList.erase(ModelList.begin());
		ModelList.erase(ModelList.begin() + 1, ModelList.end());

		for (auto& model : ModelList)
		{
			CVRModel _model(dataPath + "models/" + model.second + "/textured.obj");
			printf("load models %d/%d    \n", vi++, (int)ModelList.size());
			models = _model;
			tracker.loadModel(_model.getFile(), "");
		}
		FrameRender frender(models);

		//��ָ��֡����������pose�ŵ�poses���棬����get_poses����
		int start = 1;
		vector<Pose>poses = get_poses(start, gtPoses);
		//poses.erase(poses.begin());
		poses.erase(poses.begin() + 1, poses.end());
		tracker.reset(loadImage(start), poses[0], K);

		printf("\n");
		int nLost = 0, nTotal = 0;
		const int S = 1;
		float meanR = 0.f, meant = 0.f;
		int cnterr = 0;
		for (int fi = start + S; fi <= framecnt; fi += S)
		{
			g_fi = fi;

			cv::Mat3b img = loadImage(fi);

			time_t beg = clock();

			tracker.startUpdate(img, fi);
			//tracker.update(poses[0]);
			tracker.endUpdate(poses[0]);
			totalTime += uint(clock() - beg);
			++totalFrames;
			printf("fi=%d, time=%dms:     \r", fi, int(clock() - beg));

			int oi = 0;
			vector<Pose>gtps = get_poses(fi, gtPoses);
			//gtps.erase(gtps.begin());
			gtps.erase(gtps.begin() + 1, gtps.end());

			//Mat dimg = frender.renderResults(img, K, poses[0]);
			//Mat gtimg = frender.renderResults(img, K, gtps[0]);
			//cv::namedWindow("dimg", cv::WINDOW_NORMAL);
			//cv::imshow("dimg", dimg);
			//cv::namedWindow("gtimg", cv::WINDOW_NORMAL);
			//cv::imshow("gtimg", gtimg);
			//cv::waitKey(1); // �ȴ������¼�
			/*cv::namedWindow("dimg", cv::WINDOW_NORMAL);
			cv::imshow("dimg", dimg);
			cv::namedWindow("gtimg", cv::WINDOW_NORMAL);
			cv::imshow("gtimg", gtimg);
			cv::namedWindow("img", cv::WINDOW_NORMAL);
			cv::imshow("img", img);*/

			cv::waitKey(1); // �ȴ������¼�
			// ���� ESC ���˳�ѭ��
			if (waitKey(1) == 27)
				break;
			printLost(poses[0].R, poses[0].t, gtps[0].R, gtps[0].t);

			if (isLost(poses[0], gtps[0]))
			{
				printf("E%d:", fi);
				printLost(poses[0].R, poses[0].t, gtps[0].R, gtps[0].t);

				poses[0] = gtps[0];
				tracker.reset(img, poses[0], K);
				++nLost;
			}
			++nTotal;
		}
		printf("\n");
		float acc = 100.f * (nTotal - nLost) / nTotal;
		printf("%s acc=%.2f   \n", video.c_str(), acc);
	}

}

CMD_BEG()
CMD0("3dmultiobj_show", mulitrackers_show)
CMD_END()

static void video_gen()
{
	string folderPath = "D:/RBOT_dataset/can/py";
	int frameWidth = 640;  // ����ʵ��ͼ�����
	int frameHeight = 512; // ����ʵ��ͼ�����
	double fps = 10;        // �趨֡��

	VideoWriter video("D:/RBOT_dataset/can/output.mp4", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(frameWidth, frameHeight));

	if (!video.isOpened()) {
		cout << "Could not open the output video for write." << endl;
		return;
	}

	for (int i = 1; i <= 1000; i++) {
		for (int j = 0; j <= 4; j++) {
			string filePath = folderPath + "/" + to_string(i) + "_" + to_string(j) + ".png"; // ȷ���������չ����ʵ��ͼ�����
			Mat frame = imread(filePath);
			cout << "the image finished: " << filePath << endl;
			if (frame.empty()) {
				//cout << "Could not open or find the image: " << filePath << endl;
				continue;
			}

			video.write(frame);
		}
	}

	video.release();
	cout << "Video created successfully." << endl;

}

CMD_BEG()
CMD0("videogenerating", video_gen)
CMD_END()

_STATIC_END
