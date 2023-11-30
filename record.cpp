//LM
int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	if (pose.hdl && pose.hdl->test(R, t) <= 0)
		return 0;

	++g_totalUpdates;

	const float fx = K(0, 0), fy = K(1, 1);

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();

	float E_x = 0.f;
	float E_deltax = 0.f;
	float f_x = 0.f;
	float f_deltx = 0.f;
	float E_diff = 0.f;
	float L_diff = 0.f;
	float ρ = 0.f;

	Matx66f JJ = Matx66f::zeros();
	Vec6f J(0, 0, 0, 0, 0, 0);

	for (int i = 0; i < npt; ++i)
	{
		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		/*if (q.z == 0.f)
			continue;
		else*/
		{
			const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
			if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
				continue;

			Point3f qn = K * (R * vcp[i].normal + t);
			Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
			n = normalize(n);

			const float X = Q.x, Y = Q.y, Z = Q.z;
			/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
			dq/dQ = |                     | = |        |
					|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
			*/
			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			auto* dd = this->getDirData(n);
			if (!dd)
				continue;
			int cpi;
			auto* dirLine = dd->getScanLine(pt, cpi);
			if (!dirLine || cpi < 0)
				continue;

			auto& cp = dirLine->vPoints[cpi];

			Vec2f nx = dirLine->xdir;
			float du = (pt - dirLine->xstart).dot(nx);

			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);

			float w = pow(1.f / (fabs(du - cp.x) + 1.f), 2.f - alpha) * cp.w * cp.w;

			E_x += w * (du - cp.x) * (du - cp.x);

			f_x += w * (du - cp.x);

			J += w * (du - cp.x) * j;

			JJ += w * j * j.t();
		}
	}

	/*const float lambda = 5000.f * npt / 200.f;

	for (int i = 0; i < 3; ++i)
		JJ(i, i) += lambda * 100.f;

	for (int i = 3; i < 6; ++i)
		JJ(i, i) += lambda;*/
	if (isinit)
	{
		float tao = 1e-4;
		for (int i = 0; i < 6; i++)
			mu = max(mu, tao * JJ(i, i));
	}
	for (int i = 0; i < 6; i++)
		JJ(i, i) += mu;

	int ec = 0;

	Vec6f p;// = -JJ.inv() * J;
	if (solve(JJ, -J, p))
	{
		cv::Vec3f dt(p[0], p[1], p[2]);
		cv::Vec3f rvec(p[3], p[4], p[5]);
		Matx33f dR;
		cv::Rodrigues(rvec, dR);

		pose.t = pose.R * dt + pose.t;
		pose.R = pose.R * dR;
		R = pose.R;
		t = pose.t;
		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
			if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
				continue;
			Point3f qn = K * (R * vcp[i].normal + t);
			Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
			n = normalize(n);
			const float X = Q.x, Y = Q.y, Z = Q.z;
			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);
			Point2f pt(q.x / q.z, q.y / q.z);
			auto* dd = this->getDirData(n);
			if (!dd)
				continue;
			int cpi;
			auto* dirLine = dd->getScanLine(pt, cpi);
			if (!dirLine || cpi < 0)
				continue;
			auto& cp = dirLine->vPoints[cpi];
			Vec2f nx = dirLine->xdir;
			float du = (pt - dirLine->xstart).dot(nx);
			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);
			auto dt = n_dq_dQ.t() * R;
			auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));
			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);
			float w = pow(1.f / (fabs(du - cp.x) + 1.f), 2.f - alpha) * cp.w * cp.w;
			E_deltax += w * (du - cp.x) * (du - cp.x);
			f_deltx += w * (du - cp.x);
		}
		E_diff = E_deltax - E_x;//F(x;t) - F(x;t+△t)
		L_diff = (-p.t() * J * f_x - 0.5 * p.t() * J * J.t() * p)[0];//L(0) - L(△t)
		ρ = E_diff / L_diff;
		if (ρ > 0)
		{
			float s = 1.f / 3.f;
			v = 2.f;
			float temp = (1 - pow(2 * ρ - 1, 3));
			mu = (temp > s ? mu * temp : mu * s);

		}
		else
		{
			//此时会造成代价函数值增大，不更新参数
			mu = mu * v;
			v = v * 2;
		}


		if (g_uhdl)
			g_uhdl->onUpdate(pose.R, pose.t);

		float diff = p.dot(p);
		//printf("diff=%f\n", sqrt(diff));

		return diff < eps* eps ? 0 : 1;
	}

	return 0;
}
bool update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps)
{
	float mu = 0.f;
	float v = 2.f;
	for (int itr = 0; itr < maxItrs; ++itr)
		if (this->_update(pose, K, cpoints, alpha, eps, mu, v, itr == 0) <= 0)
			return false;
	return true;
}

//GM
int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps)
{
	const Matx33f R = pose.R;
	const Point3f t = pose.t;

	if (pose.hdl && pose.hdl->test(R, t) <= 0)
		return 0;

	++g_totalUpdates;

	const float fx = K(0, 0), fy = K(1, 1);

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();

	Matx66f JJ = Matx66f::zeros();
	Vec6f J(0, 0, 0, 0, 0, 0);

	for (int i = 0; i < npt; ++i)
	{
		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		/*if (q.z == 0.f)
			continue;
		else*/
		{
			const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
			if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
				continue;

			Point3f qn = K * (R * vcp[i].normal + t);
			Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
			n = normalize(n);

			const float X = Q.x, Y = Q.y, Z = Q.z;
			/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
			dq/dQ = |                     | = |        |
					|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
			*/
			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			auto* dd = this->getDirData(n);
			if (!dd)
				continue;
			int cpi;
			auto* dirLine = dd->getScanLine(pt, cpi);
			if (!dirLine || cpi < 0)
				continue;

			auto& cp = dirLine->vPoints[cpi];

			Vec2f nx = dirLine->xdir;
			float du = (pt - dirLine->xstart).dot(nx);

			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);

			float w = pow(1.f / (fabs(du - cp.x) + 1.f), 2.f - alpha) * cp.w * cp.w;

			J += w * (du - cp.x) * j;

			JJ += w * j * j.t();
		}
	}

	const float lambda = 5000.f * npt / 200.f;

	for (int i = 0; i < 3; ++i)
		JJ(i, i) += lambda * 100.f;

	for (int i = 3; i < 6; ++i)
		JJ(i, i) += lambda;

	int ec = 0;

	Vec6f p;// = -JJ.inv() * J;
	if (solve(JJ, -J, p))
	{
		cv::Vec3f dt(p[0], p[1], p[2]);
		cv::Vec3f rvec(p[3], p[4], p[5]);
		Matx33f dR;
		cv::Rodrigues(rvec, dR);

		pose.t = pose.R * dt + pose.t;
		pose.R = pose.R * dR;


		if (g_uhdl)
			g_uhdl->onUpdate(pose.R, pose.t);

		float diff = p.dot(p);
		//printf("diff=%f\n", sqrt(diff));

		return diff < eps* eps ? 0 : 1;
	}

	return 0;
}
bool update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps)
{
	for (int itr = 0; itr < maxItrs; ++itr)
		if (this->_update(pose, K, cpoints, alpha, eps) <= 0)
			return false;
	return true;
}

//新方法,单纯使用均值
struct Optimizer
{
public:
	struct ContourPoint
	{
		float   w; //weight
		float   x; //position on the scan-line
	};
	enum { MAX_POINTS_PER_LINE = 3 };
	struct ScanLine
	{
		float     y;
		Point2f   xdir;
		Point2f   xstart;
		ContourPoint  vPoints[MAX_POINTS_PER_LINE];
		int       nPoints;
		short* cpIndex; //index of the closest contour point for each x position
	public:
		void setCoordinates(const Point2f& start, const Point2f& end, float y)
		{
			this->xstart = start;
			xdir = (Point2f)normalize(Vec2f(end - start));
			this->y = y;
		}
		int getClosestContourPoint(const Point2f& pt, int xsize)
		{
			int x = int((pt - xstart).dot(xdir) + 0.5f);
			if (uint(x) < uint(xsize))
				return cpIndex[x];
			return -1;
		}
	};

	struct DirData
	{
		Vec2f      dir;
		Point2f    ystart;
		Point2f    ydir;
		std::vector<ScanLine>  lines;
		Mat1s         _cpIndexBuf;
	public:
		void setCoordinates(const Point2f& ystart, const Point2f& ypt)
		{
			this->ystart = ystart;
			ydir = (Point2f)normalize(Vec2f(ypt - ystart));
		}
		void resize(int rows, int cols)
		{
			lines.clear();
			lines.resize(rows);
			_cpIndexBuf.create(rows, cols);
			for (int y = 0; y < rows; ++y)
			{
				lines[y].cpIndex = _cpIndexBuf.ptr<short>(y);
			}
		}
		const ScanLine* getScanLine(const Point2f& pt, int& matchedContourPoint)
		{
			int y = int((pt - ystart).dot(ydir) + 0.5f);
			if (uint(y) >= lines.size())
				return nullptr;
			matchedContourPoint = lines[y].getClosestContourPoint(pt, int(_cpIndexBuf.cols));
			return &lines[y];
		}
	};

	std::vector<DirData>  _dirs;
	Rect  _roi;
public:
	static void _gaussianFitting(const float* data, int size, ContourPoint& cp)
	{
		float w = 0.f, wsum = 0.f;
		for (int i = 0; i < size; ++i)
		{
			wsum += data[i] * float(i);
			w += data[i];
		}

		cp.x = wsum / w;
	}
	struct _LineBuilder
	{
		struct LocalMaxima
		{
			int x;
			float val;
		};
		std::vector<LocalMaxima>  _lmBuf;
	public:
		_LineBuilder(int size)
		{
			_lmBuf.resize(size);
		}

		void operator()(ScanLine& line, const float* data, int size, int gaussWindowSizeHalf)
		{
			LocalMaxima* vlm = &_lmBuf[0];
			int nlm = 0;
			for (int i = 1; i < size - 1; ++i)
			{
				if (data[i] > data[i - 1] && data[i] > data[i + 1])
				{
					auto& lm = vlm[nlm++];
					lm.x = i;
					lm.val = data[i];
				}
			}
			if (nlm > MAX_POINTS_PER_LINE)
			{
				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.val > b.val;
					});
				nlm = MAX_POINTS_PER_LINE;

				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.x < b.x;
					});

			}

			for (int i = 0; i < nlm; ++i)
			{
				auto& lm = vlm[i];
				auto& cp = line.vPoints[i];

				const int start = __max(0, lm.x - gaussWindowSizeHalf), end = __min(size, lm.x + gaussWindowSizeHalf);
				_gaussianFitting(data + start, end - start, cp);

				cp.x += (float)start;
				cp.w = lm.val;
			}
			line.nPoints = nlm;

			if (nlm <= 1)
				memset(line.cpIndex, nlm == 0 ? 0xFF : 0, sizeof(short) * size);
			else
			{
				int start = 0;
				for (int pi = 0; pi < nlm - 1; ++pi)
				{
					int end = int(int(line.vPoints[pi].x + line.vPoints[pi + 1].x) / 2 + 0.5f) + 1;
					for (int i = start; i < end; ++i)
						line.cpIndex[i] = pi;
					start = end;
				}
				for (int i = start; i < size; ++i)
					line.cpIndex[i] = nlm - 1;
			}
		}
	};
	static void _calcScanLinesForRows(const Mat1f& prob, DirData& dirPositive, DirData& dirNegative, const Matx23f& invA)
	{
		const int gaussWindowSizeHalf = 3;

		Mat1f edgeProb;
		cv::Sobel(prob, edgeProb, CV_32F, 1, 0, 7);


		{
			Point2f O = transA(Point2f(0.f, 0.f), invA.val), P = transA(Point2f(0.f, float(prob.rows - 1)), invA.val);
			dirPositive.setCoordinates(O, P);
			dirNegative.setCoordinates(P, O);
		}
		dirPositive.resize(prob.rows, prob.cols);
		dirNegative.resize(prob.rows, prob.cols);

		std::unique_ptr<float[]> _rdata(new float[prob.cols * 2]);
		float* posData = _rdata.get(), * negData = posData + prob.cols;
		_LineBuilder buildLine(prob.cols);

		const int xend = int(prob.cols - 1);
		for (int y = 0; y < prob.rows; ++y)
		{
			auto& positiveLine = dirPositive.lines[y];
			auto& negativeLine = dirNegative.lines[prob.rows - 1 - y];

			Point2f O = transA(Point2f(0.f, float(y)), invA.val), P = transA(Point2f(float(prob.cols - 1), float(y)), invA.val);
			positiveLine.setCoordinates(O, P, float(y));
			negativeLine.setCoordinates(P, O, float(prob.rows - 1 - y));

			const float* ep = edgeProb.ptr<float>(y);

			for (int x = 0; x < prob.cols; ++x)
			{
				if (ep[x] > 0)
				{
					posData[x] = ep[x]; negData[xend - x] = 0.f;
				}
				else
				{

					posData[x] = 0.f; negData[xend - x] = -ep[x];
				}
			}

			buildLine(positiveLine, posData, prob.cols, gaussWindowSizeHalf);
			buildLine(negativeLine, negData, prob.cols, gaussWindowSizeHalf);
		}
	}

	//Mat  _prob;
	std::vector<int>  _dirIndex;

	float computeScanLines(const Mat1f& prob_, Rect roi)
	{
		//_prob = prob_.clone(); //save for visualization
		const int N = 8;

		Point2f center(float(roi.x + roi.width / 2), float(roi.y + roi.height / 2));
		Rect_<float> roif(roi);
		std::vector<Point2f>  corners = {
			Point2f(roif.x, roif.y), Point2f(roif.x + roif.width,roif.y), Point2f(roif.x + roif.width,roif.y + roif.height),Point2f(roif.x,roif.y + roif.height)
		};

		struct _DDir
		{
			Vec2f   dir;
			Matx23f A;
		};

		_dirs.clear();
		_dirs.resize(N * 2);

		for (int i = 0; i < N; ++i)
			/*cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				for (int i = r.start; i < r.end; ++i)*/
		{
			double theta = 180.0 / N * i;
			Matx23f A = getRotationMatrix2D(center, theta, 1.0);
			std::vector<Point2f> Acorners;
			cv::transform(corners, Acorners, A);
			cv::Rect droi = getBoundingBox2D(Acorners);
			A = Matx23f(1, 0, -droi.x,
				0, 1, -droi.y) * A;

			Mat1f dirProb;
			cv::warpAffine(prob_, dirProb, A, droi.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

			theta = theta / 180.0 * CV_PI;
			auto dir = Vec2f(cos(theta), sin(theta));
			/*imshow("dirProb", dirProb);
			cv::waitKey();*/

			auto& positiveDir = _dirs[i];
			auto& negativeDir = _dirs[i + N];

			auto invA = invertAffine(A);
			_calcScanLinesForRows(dirProb, positiveDir, negativeDir, invA);

			positiveDir.dir = dir;
			negativeDir.dir = -dir;
		}
		/*});*/

	//normalize weight of contour points
		{
			float wMax = 0;
			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						if (dirLine.vPoints[i].w > wMax)
							wMax = dirLine.vPoints[i].w;
				}
			}

			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						dirLine.vPoints[i].w /= wMax;
				}
			}
		}

		//build index of dirs
		if (_dirIndex.empty())
		{
			_dirIndex.resize(361);
			for (int i = 0; i < (int)_dirIndex.size(); ++i)
			{
				float theta = i * CV_PI / 180.f - CV_PI;
				Vec2f dir(cos(theta), sin(theta));

				float cosMax = -1;
				int jm = -1;
				for (int j = 0; j < (int)_dirs.size(); ++j)
				{
					float vcos = _dirs[j].dir.dot(dir);
					if (vcos > cosMax)
					{
						cosMax = vcos;
						jm = j;
					}
				}
				_dirIndex[i] = jm;
			}
		}

		_roi = roi;
	}
	DirData* getDirData(const Vec2f& ptNormal)
	{
		float theta = atan2(ptNormal[1], ptNormal[0]);
		int i = int((theta + CV_PI) * 180 / CV_PI);
		auto* ddx = uint(i) < _dirIndex.size() ? &_dirs[_dirIndex[i]] : nullptr;
		return ddx;
	}

	struct PoseData
		:public Pose
	{
		int itr = 0;
		DFRHandler* hdl = nullptr;
	};

	float calcError(const PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float err = 0.f;
		float nerr = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;

				float du = (pt - dirLine->xstart).dot(nx);

				float w = cp.w * cp.w;
				err += pow(fabs(du - cp.x), alpha) * w;

				nerr += w;
			}
		}
		return err / nerr;
	}
	float InitV(PoseData pose, const Matx33f& K, const std::vector<CPoint>& cpoints)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float avgE = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);
				avgE += fabs(du - cp.x);
			}
		}
		avgE /= npt;

		return avgE;
	}
	int _update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, float v, float eps)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);

				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);
				//φ=exp(-F^2/2v^2)
				float w = exp(-(du - cp.x) * (du - cp.x) / (2 * v * v)) * cp.w * cp.w;

				J += w * (du - cp.x) * j;

				JJ += w * j * j.t();
			}
		}

		const float lambda = 5000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 100.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		int ec = 0;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			dpose.t = pose.R * dt + pose.t;
			dpose.R = pose.R * dR;

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float v, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, dpose, K, cpoints, v, eps) <= 0)
				return false;
		return true;
	}
};
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	float v;
	int k = 0;
	const float eps = 1e-4;
	float vmax = 0.f, vmin = 0.f;
	Optimizer::PoseData dpose;//T'
	static_cast<Pose&>(dpose) = pose;
	Optimizer::PoseData lpose;//T0
	static_cast<Pose&>(lpose) = pose;
	const int Iv = 10, innerItrs = 3;

	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		std::vector<Point2f>  c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);
	//计算vmax和vmin
	vmax = dfr.InitV(dpose, K, this->views[curView].contourPoints3d) * 3;
	float avgE = dfr.InitV(dpose, K, this->views[curView].contourPoints3d) * 1.5;
	vmax = avgE * 0.5;
	vmin = avgE / 1000;
	v = vmax; //vmin = 0.0001f;
	while (1)
	{
		int kstart = k;
		while (k - kstart < Iv)
		{
			//T'=itr(Tk,v);
			curView = this->_getNearestView(lpose.R, lpose.t);
			//检查在当前v下的收敛性，收敛性好那就跳出去，换一个v再迭代
			if (!dfr.update(lpose, dpose, K, this->views[curView].contourPoints3d, innerItrs, v, eps))
				break;
			k++;
			lpose = dpose;
		}
		if (v == vmin)break;
		//lpose = dpose;
		v = max(v / 2, vmin);
		k++;
	}
	pose = lpose;

	return 1.f;
}

//原版eccv
struct Optimizer
{
public:
	struct ContourPoint
	{
		float   w; //weight
		float   x; //position on the scan-line
	};

	enum { MAX_POINTS_PER_LINE = 3 };
	struct ScanLine
	{
		float     y;
		Point2f   xdir;
		Point2f   xstart;
		ContourPoint  vPoints[MAX_POINTS_PER_LINE];
		int       nPoints;
		short* cpIndex; //index of the closest contour point for each x position
	public:
		void setCoordinates(const Point2f& start, const Point2f& end, float y)
		{
			this->xstart = start;
			xdir = (Point2f)normalize(Vec2f(end - start));
			this->y = y;
		}
		int getClosestContourPoint(const Point2f& pt, int xsize)
		{
			int x = int((pt - xstart).dot(xdir) + 0.5f);
			if (uint(x) < uint(xsize))
				return cpIndex[x];
			return -1;
		}
	};

	struct DirData
	{
		Vec2f      dir;
		Point2f    ystart;
		Point2f    ydir;
		std::vector<ScanLine>  lines;
		Mat1s         _cpIndexBuf;
	public:
		void setCoordinates(const Point2f& ystart, const Point2f& ypt)
		{
			this->ystart = ystart;
			ydir = (Point2f)normalize(Vec2f(ypt - ystart));
		}
		void resize(int rows, int cols)
		{
			lines.clear();
			lines.resize(rows);
			_cpIndexBuf.create(rows, cols);
			for (int y = 0; y < rows; ++y)
			{
				lines[y].cpIndex = _cpIndexBuf.ptr<short>(y);
			}
		}
		const ScanLine* getScanLine(const Point2f& pt, int& matchedContourPoint)
		{
			int y = int((pt - ystart).dot(ydir) + 0.5f);
			if (uint(y) >= lines.size())
				return nullptr;
			matchedContourPoint = lines[y].getClosestContourPoint(pt, int(_cpIndexBuf.cols));
			return &lines[y];
		}
	};

	std::vector<DirData>  _dirs;
	Rect  _roi;
public:
	static void _gaussianFitting(const float* data, int size, ContourPoint& cp)
	{
		float w = 0.f, wsum = 0.f;
		for (int i = 0; i < size; ++i)
		{
			wsum += data[i] * float(i);
			w += data[i];
		}

		cp.x = wsum / w;
	}
	struct _LineBuilder
	{
		struct LocalMaxima
		{
			int x;
			float val;
		};
		std::vector<LocalMaxima>  _lmBuf;
	public:
		_LineBuilder(int size)
		{
			_lmBuf.resize(size);
		}

		void operator()(ScanLine& line, const float* data, int size, int gaussWindowSizeHalf)
		{
			LocalMaxima* vlm = &_lmBuf[0];
			int nlm = 0;
			for (int i = 1; i < size - 1; ++i)
			{
				if (data[i] > data[i - 1] && data[i] > data[i + 1])
				{
					auto& lm = vlm[nlm++];
					lm.x = i;
					lm.val = data[i];
				}
			}
			if (nlm > MAX_POINTS_PER_LINE)
			{
				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.val > b.val;
					});
				nlm = MAX_POINTS_PER_LINE;

				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.x < b.x;
					});
			}
			for (int i = 0; i < nlm; ++i)
			{
				auto& lm = vlm[i];
				auto& cp = line.vPoints[i];

				const int start = __max(0, lm.x - gaussWindowSizeHalf), end = __min(size, lm.x + gaussWindowSizeHalf);
				_gaussianFitting(data + start, end - start, cp);

				cp.x += (float)start;
				cp.w = lm.val;
			}
			line.nPoints = nlm;

			if (nlm <= 1)
				memset(line.cpIndex, nlm == 0 ? 0xFF : 0, sizeof(short) * size);
			else
			{
				int start = 0;
				for (int pi = 0; pi < nlm - 1; ++pi)
				{
					int end = int(int(line.vPoints[pi].x + line.vPoints[pi + 1].x) / 2 + 0.5f) + 1;
					for (int i = start; i < end; ++i)
						line.cpIndex[i] = pi;
					start = end;
				}
				for (int i = start; i < size; ++i)
					line.cpIndex[i] = nlm - 1;
			}
		}
	};
	static void _calcScanLinesForRows(const Mat1f& prob, DirData& dirPositive, DirData& dirNegative, const Matx23f& invA)
	{
		const int gaussWindowSizeHalf = 3;

		Mat1f edgeProb;
		cv::Sobel(prob, edgeProb, CV_32F, 1, 0, 7);


		{
			Point2f O = transA(Point2f(0.f, 0.f), invA.val), P = transA(Point2f(0.f, float(prob.rows - 1)), invA.val);
			dirPositive.setCoordinates(O, P);
			dirNegative.setCoordinates(P, O);
		}
		dirPositive.resize(prob.rows, prob.cols);
		dirNegative.resize(prob.rows, prob.cols);

		std::unique_ptr<float[]> _rdata(new float[prob.cols * 2]);
		float* posData = _rdata.get(), * negData = posData + prob.cols;
		_LineBuilder buildLine(prob.cols);

		const int xend = int(prob.cols - 1);
		for (int y = 0; y < prob.rows; ++y)
		{
			auto& positiveLine = dirPositive.lines[y];
			auto& negativeLine = dirNegative.lines[prob.rows - 1 - y];

			Point2f O = transA(Point2f(0.f, float(y)), invA.val), P = transA(Point2f(float(prob.cols - 1), float(y)), invA.val);
			positiveLine.setCoordinates(O, P, float(y));
			negativeLine.setCoordinates(P, O, float(prob.rows - 1 - y));

			const float* ep = edgeProb.ptr<float>(y);

			for (int x = 0; x < prob.cols; ++x)
			{
				if (ep[x] > 0)
				{
					posData[x] = ep[x]; negData[xend - x] = 0.f;
				}
				else
				{
					posData[x] = 0.f; negData[xend - x] = -ep[x];
				}
			}

			buildLine(positiveLine, posData, prob.cols, gaussWindowSizeHalf);
			buildLine(negativeLine, negData, prob.cols, gaussWindowSizeHalf);
		}
	}

	//Mat  _prob;
	std::vector<int>  _dirIndex;

	void computeScanLines(const Mat1f& prob_, Rect roi)
	{
		//_prob = prob_.clone(); //save for visualization

		const int N = 8;

		Point2f center(float(roi.x + roi.width / 2), float(roi.y + roi.height / 2));
		Rect_<float> roif(roi);
		std::vector<Point2f>  corners = {
			Point2f(roif.x, roif.y), Point2f(roif.x + roif.width,roif.y), Point2f(roif.x + roif.width,roif.y + roif.height),Point2f(roif.x,roif.y + roif.height)
		};

		struct _DDir
		{
			Vec2f   dir;
			Matx23f A;
		};

		_dirs.clear();
		_dirs.resize(N * 2);

		//for (int i = 0; i < N; ++i)
		cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
			for (int i = r.start; i < r.end; ++i)
			{
				double theta = 180.0 / N * i;
				Matx23f A = getRotationMatrix2D(center, theta, 1.0);
				std::vector<Point2f> Acorners;
				cv::transform(corners, Acorners, A);
				cv::Rect droi = getBoundingBox2D(Acorners);
				A = Matx23f(1, 0, -droi.x,
					0, 1, -droi.y) * A;

				Mat1f dirProb;
				cv::warpAffine(prob_, dirProb, A, droi.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

				theta = theta / 180.0 * CV_PI;
				auto dir = Vec2f(cos(theta), sin(theta));

				/*imshow("dirProb", dirProb);
				cv::waitKey();*/

				auto& positiveDir = _dirs[i];
				auto& negativeDir = _dirs[i + N];

				auto invA = invertAffine(A);
				_calcScanLinesForRows(dirProb, positiveDir, negativeDir, invA);

				positiveDir.dir = dir;
				negativeDir.dir = -dir;
			}
			});

		//normalize weight of contour points
		{
			float wMax = 0;
			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						if (dirLine.vPoints[i].w > wMax)
							wMax = dirLine.vPoints[i].w;
				}
			}

			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						dirLine.vPoints[i].w /= wMax;
				}
			}
		}

		//build index of dirs
		if (_dirIndex.empty())
		{
			_dirIndex.resize(361);
			for (int i = 0; i < (int)_dirIndex.size(); ++i)
			{
				float theta = i * CV_PI / 180.f - CV_PI;
				Vec2f dir(cos(theta), sin(theta));

				float cosMax = -1;
				int jm = -1;
				for (int j = 0; j < (int)_dirs.size(); ++j)
				{
					float vcos = _dirs[j].dir.dot(dir);
					if (vcos > cosMax)
					{
						cosMax = vcos;
						jm = j;
					}
				}
				_dirIndex[i] = jm;
			}
		}

		_roi = roi;
	}
	DirData* getDirData(const Vec2f& ptNormal)
	{
		float theta = atan2(ptNormal[1], ptNormal[0]);
		int i = int((theta + CV_PI) * 180 / CV_PI);
		auto* ddx = uint(i) < _dirIndex.size() ? &_dirs[_dirIndex[i]] : nullptr;
		return ddx;
	}

	struct PoseData
		:public Pose
	{
		int itr = 0;
		DFRHandler* hdl = nullptr;
	};

	float calcError(const PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float err = 0.f;
		float nerr = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;

				float du = (pt - dirLine->xstart).dot(nx);

				float w = cp.w * cp.w;
				err += pow(fabs(du - cp.x), alpha) * w;

				nerr += w;
			}
		}
		return err / nerr;
	}

	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);

				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);

				float w = pow(1.f / (fabs(du - cp.x) + 1.f), 2.f - alpha) * cp.w * cp.w;

				J += w * (du - cp.x) * j;

				JJ += w * j * j.t();
			}
		}

		const float lambda = 5000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 100.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		int ec = 0;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			pose.t = pose.R * dt + pose.t;
			pose.R = pose.R * dR;


			if (g_uhdl)
				g_uhdl->onUpdate(pose.R, pose.t);

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, K, cpoints, alpha, eps) <= 0)
				return false;
		return true;
	}
};
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		std::vector<Point2f>  c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);

	//printf("init time=%dms \n", int(clock() - beg));

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;

	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3;

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}

	pose = dpose;

	return 1.f;
}

//每个点一个v
struct Optimizer
{
public:
	struct ContourPoint
	{
		float   w; //weight
		float   x; //position on the scan-line
	};
	enum { MAX_POINTS_PER_LINE = 3 };
	struct ScanLine
	{
		float     y;
		Point2f   xdir;
		Point2f   xstart;
		ContourPoint  vPoints[MAX_POINTS_PER_LINE];
		int       nPoints;
		short* cpIndex; //index of the closest contour point for each x position
	public:
		void setCoordinates(const Point2f& start, const Point2f& end, float y)
		{
			this->xstart = start;
			xdir = (Point2f)normalize(Vec2f(end - start));
			this->y = y;
		}
		int getClosestContourPoint(const Point2f& pt, int xsize)
		{
			int x = int((pt - xstart).dot(xdir) + 0.5f);
			if (uint(x) < uint(xsize))
				return cpIndex[x];
			return -1;
		}
	};

	struct DirData
	{
		Vec2f      dir;
		Point2f    ystart;
		Point2f    ydir;
		std::vector<ScanLine>  lines;
		Mat1s         _cpIndexBuf;
	public:
		void setCoordinates(const Point2f& ystart, const Point2f& ypt)
		{
			this->ystart = ystart;
			ydir = (Point2f)normalize(Vec2f(ypt - ystart));
		}
		void resize(int rows, int cols)
		{
			lines.clear();
			lines.resize(rows);
			_cpIndexBuf.create(rows, cols);
			for (int y = 0; y < rows; ++y)
			{
				lines[y].cpIndex = _cpIndexBuf.ptr<short>(y);
			}
		}
		const ScanLine* getScanLine(const Point2f& pt, int& matchedContourPoint)
		{
			int y = int((pt - ystart).dot(ydir) + 0.5f);
			if (uint(y) >= lines.size())
				return nullptr;
			matchedContourPoint = lines[y].getClosestContourPoint(pt, int(_cpIndexBuf.cols));
			return &lines[y];
		}
	};

	std::vector<DirData>  _dirs;
	Rect  _roi;
public:
	static void _gaussianFitting(const float* data, int size, ContourPoint& cp)
	{
		float w = 0.f, wsum = 0.f;
		for (int i = 0; i < size; ++i)
		{
			wsum += data[i] * float(i);
			w += data[i];
		}

		cp.x = wsum / w;
	}
	struct _LineBuilder
	{
		struct LocalMaxima
		{
			int x;
			float val;
		};
		std::vector<LocalMaxima>  _lmBuf;
	public:
		_LineBuilder(int size)
		{
			_lmBuf.resize(size);
		}

		void operator()(ScanLine& line, const float* data, int size, int gaussWindowSizeHalf)
		{
			LocalMaxima* vlm = &_lmBuf[0];
			int nlm = 0;
			for (int i = 1; i < size - 1; ++i)
			{
				if (data[i] > data[i - 1] && data[i] > data[i + 1])
				{
					auto& lm = vlm[nlm++];
					lm.x = i;
					lm.val = data[i];
				}
			}
			if (nlm > MAX_POINTS_PER_LINE)
			{
				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.val > b.val;
					});
				nlm = MAX_POINTS_PER_LINE;

				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.x < b.x;
					});

			}
			for (int i = 0; i < nlm; ++i)
			{
				auto& lm = vlm[i];
				auto& cp = line.vPoints[i];

				const int start = __max(0, lm.x - gaussWindowSizeHalf), end = __min(size, lm.x + gaussWindowSizeHalf);
				_gaussianFitting(data + start, end - start, cp);

				cp.x += (float)start;
				cp.w = lm.val;
			}
			line.nPoints = nlm;

			if (nlm <= 1)
				memset(line.cpIndex, nlm == 0 ? 0xFF : 0, sizeof(short) * size);
			else
			{
				int start = 0;
				for (int pi = 0; pi < nlm - 1; ++pi)
				{
					int end = int(int(line.vPoints[pi].x + line.vPoints[pi + 1].x) / 2 + 0.5f) + 1;
					for (int i = start; i < end; ++i)
						line.cpIndex[i] = pi;
					start = end;
				}
				for (int i = start; i < size; ++i)
					line.cpIndex[i] = nlm - 1;
			}
		}
	};
	static void _calcScanLinesForRows(const Mat1f& prob, DirData& dirPositive, DirData& dirNegative, const Matx23f& invA)
	{
		const int gaussWindowSizeHalf = 3;

		Mat1f edgeProb;
		cv::Sobel(prob, edgeProb, CV_32F, 1, 0, 7);


		{
			Point2f O = transA(Point2f(0.f, 0.f), invA.val), P = transA(Point2f(0.f, float(prob.rows - 1)), invA.val);
			dirPositive.setCoordinates(O, P);
			dirNegative.setCoordinates(P, O);
		}
		dirPositive.resize(prob.rows, prob.cols);
		dirNegative.resize(prob.rows, prob.cols);

		std::unique_ptr<float[]> _rdata(new float[prob.cols * 2]);
		float* posData = _rdata.get(), * negData = posData + prob.cols;
		_LineBuilder buildLine(prob.cols);

		const int xend = int(prob.cols - 1);
		for (int y = 0; y < prob.rows; ++y)
		{
			auto& positiveLine = dirPositive.lines[y];
			auto& negativeLine = dirNegative.lines[prob.rows - 1 - y];

			Point2f O = transA(Point2f(0.f, float(y)), invA.val), P = transA(Point2f(float(prob.cols - 1), float(y)), invA.val);
			positiveLine.setCoordinates(O, P, float(y));
			negativeLine.setCoordinates(P, O, float(prob.rows - 1 - y));

			const float* ep = edgeProb.ptr<float>(y);

			for (int x = 0; x < prob.cols; ++x)
			{
				if (ep[x] > 0)
				{
					posData[x] = ep[x]; negData[xend - x] = 0.f;
				}
				else
				{

					posData[x] = 0.f; negData[xend - x] = -ep[x];
				}
			}

			buildLine(positiveLine, posData, prob.cols, gaussWindowSizeHalf);
			buildLine(negativeLine, negData, prob.cols, gaussWindowSizeHalf);
		}
	}

	//Mat  _prob;
	std::vector<int>  _dirIndex;

	void computeScanLines(const Mat1f& prob_, Rect roi)
	{
		//_prob = prob_.clone(); //save for visualization
		const int N = 8;

		Point2f center(float(roi.x + roi.width / 2), float(roi.y + roi.height / 2));
		Rect_<float> roif(roi);
		std::vector<Point2f>  corners = {
			Point2f(roif.x, roif.y), Point2f(roif.x + roif.width,roif.y), Point2f(roif.x + roif.width,roif.y + roif.height),Point2f(roif.x,roif.y + roif.height)
		};

		struct _DDir
		{
			Vec2f   dir;
			Matx23f A;
		};

		_dirs.clear();
		_dirs.resize(N * 2);

		for (int i = 0; i < N; ++i)
			/*cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				for (int i = r.start; i < r.end; ++i)*/
		{
			double theta = 180.0 / N * i;
			Matx23f A = getRotationMatrix2D(center, theta, 1.0);
			std::vector<Point2f> Acorners;
			cv::transform(corners, Acorners, A);
			cv::Rect droi = getBoundingBox2D(Acorners);
			A = Matx23f(1, 0, -droi.x,
				0, 1, -droi.y) * A;

			Mat1f dirProb;
			cv::warpAffine(prob_, dirProb, A, droi.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

			theta = theta / 180.0 * CV_PI;
			auto dir = Vec2f(cos(theta), sin(theta));
			/*imshow("dirProb", dirProb);
			cv::waitKey();*/

			auto& positiveDir = _dirs[i];
			auto& negativeDir = _dirs[i + N];

			auto invA = invertAffine(A);
			_calcScanLinesForRows(dirProb, positiveDir, negativeDir, invA);

			positiveDir.dir = dir;
			negativeDir.dir = -dir;
		}
		/*});*/

	//normalize weight of contour points
		{
			float wMax = 0;
			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						if (dirLine.vPoints[i].w > wMax)
							wMax = dirLine.vPoints[i].w;
				}
			}

			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						dirLine.vPoints[i].w /= wMax;
				}
			}
		}

		//build index of dirs
		if (_dirIndex.empty())
		{
			_dirIndex.resize(361);
			for (int i = 0; i < (int)_dirIndex.size(); ++i)
			{
				float theta = i * CV_PI / 180.f - CV_PI;
				Vec2f dir(cos(theta), sin(theta));

				float cosMax = -1;
				int jm = -1;
				for (int j = 0; j < (int)_dirs.size(); ++j)
				{
					float vcos = _dirs[j].dir.dot(dir);
					if (vcos > cosMax)
					{
						cosMax = vcos;
						jm = j;
					}
				}
				_dirIndex[i] = jm;
			}
		}

		_roi = roi;
	}
	DirData* getDirData(const Vec2f& ptNormal)
	{
		float theta = atan2(ptNormal[1], ptNormal[0]);
		int i = int((theta + CV_PI) * 180 / CV_PI);
		auto* ddx = uint(i) < _dirIndex.size() ? &_dirs[_dirIndex[i]] : nullptr;
		return ddx;
	}

	struct PoseData
		:public Pose
	{
		int itr = 0;
		DFRHandler* hdl = nullptr;
	};

	float calcError(const PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float err = 0.f;
		float nerr = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;

				float du = (pt - dirLine->xstart).dot(nx);

				float w = cp.w * cp.w;
				err += pow(fabs(du - cp.x), alpha) * w;

				nerr += w;
			}
		}
		return err / nerr;
	}
	void InitV(PoseData pose, const Matx33f& K, const std::vector<CPoint>& cpoints, vector<float>& means, vector<float>& stddevs, const int neighbor = 10)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		means.resize(npt);
		stddevs.resize(npt);
		vector<float>Func(npt);
		float sumE = 0.f;
		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);
				Func[i] = du - cp.x;
				sumE += du - cp.x;
			}
		}
		for (int i = 0; i < npt; i++)
		{
			auto& mean = means[i];//邻近点的均值
			auto& stddev = stddevs[i];//方差
			for (int j = 0; j < neighbor; j++)
			{
				int idx = (i + j - 2) % neighbor;
				mean += Func[idx];
			}
			mean /= neighbor;
			for (int j = 0; j < neighbor; j++)
			{
				int idx = (i + j - 2) % neighbor;
				stddev += (Func[idx] - mean) * (Func[idx] - mean);
			}
			stddev /= npt;
			stddev = sqrt(stddev);
		}
	}
	int _update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, vector<float> vs, float eps)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);

				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);
				//φ=exp(-F^2/2v^2)
				float v = vs[i];
				float w = exp(-(du - cp.x) * (du - cp.x) / (2 * v * v)) * cp.w * cp.w;

				J += w * (du - cp.x) * j;

				JJ += w * j * j.t();
			}
		}

		const float lambda = 5000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 100.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		int ec = 0;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			dpose.t = pose.R * dt + pose.t;
			dpose.R = pose.R * dR;

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, vector<float> vs, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, dpose, K, cpoints, vs, eps) <= 0)
				return false;
		return true;
	}
};
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	float v;
	int k = 0;
	const float eps = 1e-4;
	Optimizer::PoseData dpose;//T'
	static_cast<Pose&>(dpose) = pose;
	Optimizer::PoseData lpose;//T0
	static_cast<Pose&>(lpose) = pose;
	const int Iv = 10, innerItrs = 3;

	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		std::vector<Point2f>  c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	vector<float>means;
	vector<float>stds;
	vector<float>vmins;//每个点的vmin
	vector<float>vmaxs;//每个点的vmax
	vector<float>vs;//每个点当前的v
	vector<bool>convergent;//每个点是否收敛
	dfr.computeScanLines(curProb, roi);
	//计算vmax和vmin
	curView = this->_getNearestView(lpose.R, lpose.t);
	dfr.InitV(dpose, K, this->views[curView].contourPoints3d, means, stds);
	// 将vector<float>转换为OpenCV的Mat对象
	cv::Mat inputMat(1, stds.size(), CV_32F);
	memcpy(inputMat.ptr<float>(0), stds.data(), stds.size() * sizeof(float));
	// 定义高斯滤波的标准差
	double sigma = 1.0;
	// 对数组进行高斯滤波
	cv::GaussianBlur(inputMat, inputMat, cv::Size(0, 0), sigma);
	// 将滤波后的结果复制回vector<float>
	std::vector<float> smoothedStddevs;
	smoothedStddevs.resize(stds.size());
	memcpy(smoothedStddevs.data(), inputMat.ptr<float>(0), stds.size() * sizeof(float));

	int npt = means.size();
	vmins.resize(npt);
	vmaxs.resize(npt);
	convergent.resize(npt);
	for (int i = 0; i < npt; i++)
	{
		vmins[i] = stds[i] / 1000;
		vmaxs[i] = stds[i] * 0.5;
		convergent[i] = 0;
	}
	vs = vmaxs; //vmin = 0.0001f;
	while (1)
	{
		int kstart = k;
		while (k - kstart < Iv)
		{
			//T'=itr(Tk,v);
			curView = this->_getNearestView(lpose.R, lpose.t);
			//检查在当前v下的收敛性，收敛性好那就跳出去，换一个v再迭代
			if (!dfr.update(lpose, dpose, K, this->views[curView].contourPoints3d, innerItrs, vs, eps))
				break;
			k++;
			lpose = dpose;
		}
		bool conver = true;
		for (int i = 0; i < npt; i++)
		{
			if (vs[i] != vmins[i])
			{
				vs[i] = max(vs[i] / 2, vmins[i]);
				conver = false;
			}
			else convergent[i] = true;
		}
		if (conver)break;
		k++;
	}
	pose = lpose;

	return 1.f;
}

//用iou
struct Optimizer
{
public:
	struct ContourPoint
	{
		float   w; //weight
		float   x; //position on the scan-line
	};
	enum { MAX_POINTS_PER_LINE = 3 };
	struct ScanLine
	{
		float     y;
		Point2f   xdir;
		Point2f   xstart;
		ContourPoint  vPoints[MAX_POINTS_PER_LINE];
		int       nPoints;
		short* cpIndex; //index of the closest contour point for each x position
	public:
		void setCoordinates(const Point2f& start, const Point2f& end, float y)
		{
			this->xstart = start;
			xdir = (Point2f)normalize(Vec2f(end - start));
			this->y = y;
		}
		int getClosestContourPoint(const Point2f& pt, int xsize)
		{
			int x = int((pt - xstart).dot(xdir) + 0.5f);
			if (uint(x) < uint(xsize))
				return cpIndex[x];
			return -1;
		}
	};

	struct DirData
	{
		Vec2f      dir;
		Point2f    ystart;
		Point2f    ydir;
		std::vector<ScanLine>  lines;
		Mat1s         _cpIndexBuf;
	public:
		void setCoordinates(const Point2f& ystart, const Point2f& ypt)
		{
			this->ystart = ystart;
			ydir = (Point2f)normalize(Vec2f(ypt - ystart));
		}
		void resize(int rows, int cols)
		{
			lines.clear();
			lines.resize(rows);
			_cpIndexBuf.create(rows, cols);
			for (int y = 0; y < rows; ++y)
			{
				lines[y].cpIndex = _cpIndexBuf.ptr<short>(y);
			}
		}
		const ScanLine* getScanLine(const Point2f& pt, int& matchedContourPoint)
		{
			int y = int((pt - ystart).dot(ydir) + 0.5f);
			if (uint(y) >= lines.size())
				return nullptr;
			matchedContourPoint = lines[y].getClosestContourPoint(pt, int(_cpIndexBuf.cols));
			return &lines[y];
		}
	};

	std::vector<DirData>  _dirs;
	Rect  _roi;
public:
	static void _gaussianFitting(const float* data, int size, ContourPoint& cp)
	{
		float w = 0.f, wsum = 0.f;
		for (int i = 0; i < size; ++i)
		{
			wsum += data[i] * float(i);
			w += data[i];
		}

		cp.x = wsum / w;
	}
	struct _LineBuilder
	{
		struct LocalMaxima
		{
			int x;
			float val;
		};
		std::vector<LocalMaxima>  _lmBuf;
	public:
		_LineBuilder(int size)
		{
			_lmBuf.resize(size);
		}

		void operator()(ScanLine& line, const float* data, int size, int gaussWindowSizeHalf)
		{
			LocalMaxima* vlm = &_lmBuf[0];
			int nlm = 0;
			for (int i = 1; i < size - 1; ++i)
			{
				if (data[i] > data[i - 1] && data[i] > data[i + 1])
				{
					auto& lm = vlm[nlm++];
					lm.x = i;
					lm.val = data[i];
				}
			}
			if (nlm > MAX_POINTS_PER_LINE)
			{
				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.val > b.val;
					});
				nlm = MAX_POINTS_PER_LINE;

				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.x < b.x;
					});

			}

			for (int i = 0; i < nlm; ++i)
			{
				auto& lm = vlm[i];
				auto& cp = line.vPoints[i];

				const int start = __max(0, lm.x - gaussWindowSizeHalf), end = __min(size, lm.x + gaussWindowSizeHalf);
				_gaussianFitting(data + start, end - start, cp);

				cp.x += (float)start;
				cp.w = lm.val;
			}
			line.nPoints = nlm;

			if (nlm <= 1)
				memset(line.cpIndex, nlm == 0 ? 0xFF : 0, sizeof(short) * size);
			else
			{
				int start = 0;
				for (int pi = 0; pi < nlm - 1; ++pi)
				{
					int end = int(int(line.vPoints[pi].x + line.vPoints[pi + 1].x) / 2 + 0.5f) + 1;
					for (int i = start; i < end; ++i)
						line.cpIndex[i] = pi;
					start = end;
				}
				for (int i = start; i < size; ++i)
					line.cpIndex[i] = nlm - 1;
			}
		}
	};
	static void _calcScanLinesForRows(const Mat1f& prob, DirData& dirPositive, DirData& dirNegative, const Matx23f& invA)
	{
		const int gaussWindowSizeHalf = 3;

		Mat1f edgeProb;
		cv::Sobel(prob, edgeProb, CV_32F, 1, 0, 7);


		{
			Point2f O = transA(Point2f(0.f, 0.f), invA.val), P = transA(Point2f(0.f, float(prob.rows - 1)), invA.val);
			dirPositive.setCoordinates(O, P);
			dirNegative.setCoordinates(P, O);
		}
		dirPositive.resize(prob.rows, prob.cols);
		dirNegative.resize(prob.rows, prob.cols);

		std::unique_ptr<float[]> _rdata(new float[prob.cols * 2]);
		float* posData = _rdata.get(), * negData = posData + prob.cols;
		_LineBuilder buildLine(prob.cols);

		const int xend = int(prob.cols - 1);
		for (int y = 0; y < prob.rows; ++y)
		{
			auto& positiveLine = dirPositive.lines[y];
			auto& negativeLine = dirNegative.lines[prob.rows - 1 - y];

			Point2f O = transA(Point2f(0.f, float(y)), invA.val), P = transA(Point2f(float(prob.cols - 1), float(y)), invA.val);
			positiveLine.setCoordinates(O, P, float(y));
			negativeLine.setCoordinates(P, O, float(prob.rows - 1 - y));

			const float* ep = edgeProb.ptr<float>(y);

			for (int x = 0; x < prob.cols; ++x)
			{
				if (ep[x] > 0)
				{
					posData[x] = ep[x]; negData[xend - x] = 0.f;
				}
				else
				{

					posData[x] = 0.f; negData[xend - x] = -ep[x];
				}
			}

			buildLine(positiveLine, posData, prob.cols, gaussWindowSizeHalf);
			buildLine(negativeLine, negData, prob.cols, gaussWindowSizeHalf);
		}
	}

	//Mat  _prob;
	std::vector<int>  _dirIndex;

	void computeScanLines(const Mat1f& prob_, Rect roi)
	{
		//_prob = prob_.clone(); //save for visualization
		const int N = 8;

		Point2f center(float(roi.x + roi.width / 2), float(roi.y + roi.height / 2));
		Rect_<float> roif(roi);
		std::vector<Point2f>  corners = {
			Point2f(roif.x, roif.y), Point2f(roif.x + roif.width,roif.y), Point2f(roif.x + roif.width,roif.y + roif.height),Point2f(roif.x,roif.y + roif.height)
		};

		struct _DDir
		{
			Vec2f   dir;
			Matx23f A;
		};

		_dirs.clear();
		_dirs.resize(N * 2);

		for (int i = 0; i < N; ++i)
			/*cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				for (int i = r.start; i < r.end; ++i)*/
		{
			double theta = 180.0 / N * i;
			Matx23f A = getRotationMatrix2D(center, theta, 1.0);
			std::vector<Point2f> Acorners;
			cv::transform(corners, Acorners, A);
			cv::Rect droi = getBoundingBox2D(Acorners);
			A = Matx23f(1, 0, -droi.x,
				0, 1, -droi.y) * A;

			Mat1f dirProb;
			cv::warpAffine(prob_, dirProb, A, droi.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

			theta = theta / 180.0 * CV_PI;
			auto dir = Vec2f(cos(theta), sin(theta));
			/*imshow("dirProb", dirProb);
			cv::waitKey();*/

			auto& positiveDir = _dirs[i];
			auto& negativeDir = _dirs[i + N];

			auto invA = invertAffine(A);
			_calcScanLinesForRows(dirProb, positiveDir, negativeDir, invA);

			positiveDir.dir = dir;
			negativeDir.dir = -dir;
		}
		/*});*/

	//normalize weight of contour points
		{
			float wMax = 0;
			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						if (dirLine.vPoints[i].w > wMax)
							wMax = dirLine.vPoints[i].w;
				}
			}

			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						dirLine.vPoints[i].w /= wMax;
				}
			}
		}

		//build index of dirs
		if (_dirIndex.empty())
		{
			_dirIndex.resize(361);
			for (int i = 0; i < (int)_dirIndex.size(); ++i)
			{
				float theta = i * CV_PI / 180.f - CV_PI;
				Vec2f dir(cos(theta), sin(theta));

				float cosMax = -1;
				int jm = -1;
				for (int j = 0; j < (int)_dirs.size(); ++j)
				{
					float vcos = _dirs[j].dir.dot(dir);
					if (vcos > cosMax)
					{
						cosMax = vcos;
						jm = j;
					}
				}
				_dirIndex[i] = jm;
			}
		}

		_roi = roi;
	}
	DirData* getDirData(const Vec2f& ptNormal)
	{
		float theta = atan2(ptNormal[1], ptNormal[0]);
		int i = int((theta + CV_PI) * 180 / CV_PI);
		auto* ddx = uint(i) < _dirIndex.size() ? &_dirs[_dirIndex[i]] : nullptr;
		return ddx;
	}

	struct PoseData
		:public Pose
	{
		int itr = 0;
		DFRHandler* hdl = nullptr;
	};

	float calcError(const PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float err = 0.f;
		float nerr = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;

				float du = (pt - dirLine->xstart).dot(nx);

				float w = cp.w * cp.w;
				err += pow(fabs(du - cp.x), alpha) * w;

				nerr += w;
			}
		}
		return err / nerr;
	}
	float InitV(PoseData pose, const Matx33f& K, const std::vector<CPoint>& cpoints)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float avgE = 0.f;
		vector<float>Func(npt);
		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);
				avgE += fabs(du - cp.x);
				Func[i] = fabs(du - cp.x);
			}
		}
		avgE /= npt;
		return avgE;
	}
	int _update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, float v, float eps)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);

				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);
				//φ=exp(-F^2/2v^2)
				float w = exp(-(du - cp.x) * (du - cp.x) / (2 * v * v)) * cp.w * cp.w;

				J += w * (du - cp.x) * j;

				JJ += w * j * j.t();
			}
		}

		const float lambda = 5000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 100.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		int ec = 0;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			dpose.t = pose.R * dt + pose.t;
			dpose.R = pose.R * dR;

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float v, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, dpose, K, cpoints, v, eps) <= 0)
				return false;
		return true;
	}
};
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	float v;
	int k = 0;
	const float eps = 1e-4;
	float vmax = 0.f, vmin = 0.f;
	Optimizer::PoseData dpose;//T'
	static_cast<Pose&>(dpose) = pose;
	Optimizer::PoseData lpose;//T0
	static_cast<Pose&>(lpose) = pose;
	const int Iv = 10, innerItrs = 3;

	Rect curROI; std::vector<Point2f>  c2d;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;

	//前景和mask的iou
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	float Iou = 0.f; int numPixels = 0;
	for (int i = roi.x; i <= roi.x + roi.width; i++)
	{
		for (int j = roi.y; j <= roi.y + roi.height; j++)
		{
			if (pointPolygonTest(c2d, Point2f(i, j), false) >= 0)
			{
				Iou += curProb.at<float>(j, i);
				numPixels++;
			}
		}
	}
	/*Mat1f roiMat = curProb(roi);
	float Iou = sum(roiMat)[0];
	int numPixels = roi.width * roi.height;*/
	/*用来评价上一帧到这一帧位姿变化程度是否比较大，metricIou越大说明重合度越高，位姿变化越小、分割越好。
	如果MetircIou越大，说明：1.分割比较好，匹配的点确实是最近的，匹配的点都很好，可以把收敛域扩大
	2.位姿变化小，，因此收敛域要缩小 */
	float metricIou = Iou / numPixels;
	float metric = 0.f;
	/*if (metricIou <= 0.65)metric = 0.3;
	else
	{
		metric = 2.8 * metricIou - 1.3;
	}*/
	metric = 1 / (1 + exp(-10 * metricIou + 6.5)) + 0.2f;
	g_avgMetric += metricIou;
	g_maxMetric = max(metricIou, g_maxMetric);
	g_minMetric = min(metricIou, g_minMetric);
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);
	//计算vmax和vmin
	curView = this->_getNearestView(lpose.R, lpose.t);

	float avgE = dfr.InitV(dpose, K, this->views[curView].contourPoints3d);
	//avgE *= (1 - metric);
	avgE *= metric;
	vmax = avgE;
	vmin = avgE / 1000;//迭代log1000次
	v = vmax; vmin = 0.0001f;
	while (1)
	{
		int kstart = k;
		while (k - kstart < Iv)
		{
			//T'=itr(Tk,v);
			curView = this->_getNearestView(lpose.R, lpose.t);
			//检查在当前v下的收敛性，收敛性好那就跳出去，换一个v再迭代
			if (!dfr.update(lpose, dpose, K, this->views[curView].contourPoints3d, innerItrs, v, eps))
				break;
			k++;
			lpose = dpose;
		}
		if (v == vmin)break;
		//lpose = dpose;
		v = max(v / 2, vmin);
		k++;
	}
	pose = lpose;

	return 1.f;
}

//用均值-α标准差
struct Optimizer
{
public:
	struct ContourPoint
	{
		float   w; //weight
		float   x; //position on the scan-line
	};
	enum { MAX_POINTS_PER_LINE = 3 };
	struct ScanLine
	{
		float     y;
		Point2f   xdir;
		Point2f   xstart;
		ContourPoint  vPoints[MAX_POINTS_PER_LINE];
		int       nPoints;
		short* cpIndex; //index of the closest contour point for each x position
	public:
		void setCoordinates(const Point2f& start, const Point2f& end, float y)
		{
			this->xstart = start;
			xdir = (Point2f)normalize(Vec2f(end - start));
			this->y = y;
		}
		int getClosestContourPoint(const Point2f& pt, int xsize)
		{
			int x = int((pt - xstart).dot(xdir) + 0.5f);
			if (uint(x) < uint(xsize))
				return cpIndex[x];
			return -1;
		}
	};

	struct DirData
	{
		Vec2f      dir;
		Point2f    ystart;
		Point2f    ydir;
		std::vector<ScanLine>  lines;
		Mat1s         _cpIndexBuf;
	public:
		void setCoordinates(const Point2f& ystart, const Point2f& ypt)
		{
			this->ystart = ystart;
			ydir = (Point2f)normalize(Vec2f(ypt - ystart));
		}
		void resize(int rows, int cols)
		{
			lines.clear();
			lines.resize(rows);
			_cpIndexBuf.create(rows, cols);
			for (int y = 0; y < rows; ++y)
			{
				lines[y].cpIndex = _cpIndexBuf.ptr<short>(y);
			}
		}
		const ScanLine* getScanLine(const Point2f& pt, int& matchedContourPoint)
		{
			int y = int((pt - ystart).dot(ydir) + 0.5f);
			if (uint(y) >= lines.size())
				return nullptr;
			matchedContourPoint = lines[y].getClosestContourPoint(pt, int(_cpIndexBuf.cols));
			return &lines[y];
		}
	};

	std::vector<DirData>  _dirs;
	Rect  _roi;
public:
	static void _gaussianFitting(const float* data, int size, ContourPoint& cp)
	{
		float w = 0.f, wsum = 0.f;
		for (int i = 0; i < size; ++i)
		{
			wsum += data[i] * float(i);
			w += data[i];
		}

		cp.x = wsum / w;
	}
	struct _LineBuilder
	{
		struct LocalMaxima
		{
			int x;
			float val;
		};
		std::vector<LocalMaxima>  _lmBuf;
	public:
		_LineBuilder(int size)
		{
			_lmBuf.resize(size);
		}

		void operator()(ScanLine& line, const float* data, int size, int gaussWindowSizeHalf)
		{
			LocalMaxima* vlm = &_lmBuf[0];
			int nlm = 0;
			for (int i = 1; i < size - 1; ++i)
			{
				if (data[i] > data[i - 1] && data[i] > data[i + 1])
				{
					auto& lm = vlm[nlm++];
					lm.x = i;
					lm.val = data[i];
				}
			}
			if (nlm > MAX_POINTS_PER_LINE)
			{
				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.val > b.val;
					});
				nlm = MAX_POINTS_PER_LINE;

				std::sort(vlm, vlm + nlm, [](const LocalMaxima& a, const LocalMaxima& b) {
					return a.x < b.x;
					});

			}

			for (int i = 0; i < nlm; ++i)
			{
				auto& lm = vlm[i];
				auto& cp = line.vPoints[i];

				const int start = __max(0, lm.x - gaussWindowSizeHalf), end = __min(size, lm.x + gaussWindowSizeHalf);
				_gaussianFitting(data + start, end - start, cp);

				cp.x += (float)start;
				cp.w = lm.val;
			}
			line.nPoints = nlm;

			if (nlm <= 1)
				memset(line.cpIndex, nlm == 0 ? 0xFF : 0, sizeof(short) * size);
			else
			{
				int start = 0;
				for (int pi = 0; pi < nlm - 1; ++pi)
				{
					int end = int(int(line.vPoints[pi].x + line.vPoints[pi + 1].x) / 2 + 0.5f) + 1;
					for (int i = start; i < end; ++i)
						line.cpIndex[i] = pi;
					start = end;
				}
				for (int i = start; i < size; ++i)
					line.cpIndex[i] = nlm - 1;
			}
		}
	};
	static void _calcScanLinesForRows(const Mat1f& prob, DirData& dirPositive, DirData& dirNegative, const Matx23f& invA)
	{
		const int gaussWindowSizeHalf = 3;

		Mat1f edgeProb;
		cv::Sobel(prob, edgeProb, CV_32F, 1, 0, 7);


		{
			Point2f O = transA(Point2f(0.f, 0.f), invA.val), P = transA(Point2f(0.f, float(prob.rows - 1)), invA.val);
			dirPositive.setCoordinates(O, P);
			dirNegative.setCoordinates(P, O);
		}
		dirPositive.resize(prob.rows, prob.cols);
		dirNegative.resize(prob.rows, prob.cols);

		std::unique_ptr<float[]> _rdata(new float[prob.cols * 2]);
		float* posData = _rdata.get(), * negData = posData + prob.cols;
		_LineBuilder buildLine(prob.cols);

		const int xend = int(prob.cols - 1);
		for (int y = 0; y < prob.rows; ++y)
		{
			auto& positiveLine = dirPositive.lines[y];
			auto& negativeLine = dirNegative.lines[prob.rows - 1 - y];

			Point2f O = transA(Point2f(0.f, float(y)), invA.val), P = transA(Point2f(float(prob.cols - 1), float(y)), invA.val);
			positiveLine.setCoordinates(O, P, float(y));
			negativeLine.setCoordinates(P, O, float(prob.rows - 1 - y));

			const float* ep = edgeProb.ptr<float>(y);

			for (int x = 0; x < prob.cols; ++x)
			{
				if (ep[x] > 0)
				{
					posData[x] = ep[x]; negData[xend - x] = 0.f;
				}
				else
				{

					posData[x] = 0.f; negData[xend - x] = -ep[x];
				}
			}

			buildLine(positiveLine, posData, prob.cols, gaussWindowSizeHalf);
			buildLine(negativeLine, negData, prob.cols, gaussWindowSizeHalf);
		}
	}

	//Mat  _prob;
	std::vector<int>  _dirIndex;

	void computeScanLines(const Mat1f& prob_, Rect roi)
	{
		//_prob = prob_.clone(); //save for visualization
		const int N = 8;

		Point2f center(float(roi.x + roi.width / 2), float(roi.y + roi.height / 2));
		Rect_<float> roif(roi);
		std::vector<Point2f>  corners = {
			Point2f(roif.x, roif.y), Point2f(roif.x + roif.width,roif.y), Point2f(roif.x + roif.width,roif.y + roif.height),Point2f(roif.x,roif.y + roif.height)
		};

		struct _DDir
		{
			Vec2f   dir;
			Matx23f A;
		};

		_dirs.clear();
		_dirs.resize(N * 2);

		for (int i = 0; i < N; ++i)
			/*cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				for (int i = r.start; i < r.end; ++i)*/
		{
			double theta = 180.0 / N * i;
			Matx23f A = getRotationMatrix2D(center, theta, 1.0);
			std::vector<Point2f> Acorners;
			cv::transform(corners, Acorners, A);
			cv::Rect droi = getBoundingBox2D(Acorners);
			A = Matx23f(1, 0, -droi.x,
				0, 1, -droi.y) * A;

			Mat1f dirProb;
			cv::warpAffine(prob_, dirProb, A, droi.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

			theta = theta / 180.0 * CV_PI;
			auto dir = Vec2f(cos(theta), sin(theta));
			/*imshow("dirProb", dirProb);
			cv::waitKey();*/

			auto& positiveDir = _dirs[i];
			auto& negativeDir = _dirs[i + N];

			auto invA = invertAffine(A);
			_calcScanLinesForRows(dirProb, positiveDir, negativeDir, invA);

			positiveDir.dir = dir;
			negativeDir.dir = -dir;
		}
		/*});*/

	//normalize weight of contour points
		{
			float wMax = 0;
			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						if (dirLine.vPoints[i].w > wMax)
							wMax = dirLine.vPoints[i].w;
				}
			}

			for (auto& dir : _dirs)
			{
				for (auto& dirLine : dir.lines)
				{
					for (int i = 0; i < dirLine.nPoints; ++i)
						dirLine.vPoints[i].w /= wMax;
				}
			}
		}

		//build index of dirs
		if (_dirIndex.empty())
		{
			_dirIndex.resize(361);
			for (int i = 0; i < (int)_dirIndex.size(); ++i)
			{
				float theta = i * CV_PI / 180.f - CV_PI;
				Vec2f dir(cos(theta), sin(theta));

				float cosMax = -1;
				int jm = -1;
				for (int j = 0; j < (int)_dirs.size(); ++j)
				{
					float vcos = _dirs[j].dir.dot(dir);
					if (vcos > cosMax)
					{
						cosMax = vcos;
						jm = j;
					}
				}
				_dirIndex[i] = jm;
			}
		}

		_roi = roi;

	}
	DirData* getDirData(const Vec2f& ptNormal)
	{
		float theta = atan2(ptNormal[1], ptNormal[0]);
		int i = int((theta + CV_PI) * 180 / CV_PI);
		auto* ddx = uint(i) < _dirIndex.size() ? &_dirs[_dirIndex[i]] : nullptr;
		return ddx;
	}

	struct PoseData
		:public Pose
	{
		int itr = 0;
		DFRHandler* hdl = nullptr;
	};

	float calcError(const PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		float err = 0.f;
		float nerr = 0.f;

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;

				float du = (pt - dirLine->xstart).dot(nx);

				float w = cp.w * cp.w;
				err += pow(fabs(du - cp.x), alpha) * w;

				nerr += w;
			}
		}
		return err / nerr;
	}
	void InitV(PoseData pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float& meanE, float& stdE)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		meanE = 0.f;
		stdE = 0.f;
		vector<float>Func(npt);
		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);
				meanE += fabs(du - cp.x);
				Func[i] = fabs(du - cp.x);
			}
		}
		meanE /= npt;
		for (int i = 0; i < npt; i++)
		{
			stdE += pow(meanE - Func[i], 2);
		}
		stdE /= npt;
		stdE = sqrt(stdE);
	}
	int _update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, float v, float eps)
	{
		const Matx33f R = pose.R;
		const Point3f t = pose.t;

		if (pose.hdl && pose.hdl->test(R, t) <= 0)
			return 0;

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; ++i)
		{
			Point3f Q = R * vcp[i].center + t;
			Point3f q = K * Q;
			/*if (q.z == 0.f)
				continue;
			else*/
			{
				const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f qn = K * (R * vcp[i].normal + t);
				Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
				n = normalize(n);

				const float X = Q.x, Y = Q.y, Z = Q.z;
				/*      |fx/Z   0   -fx*X/Z^2 |   |a  0   b|
				dq/dQ = |                     | = |        |
						|0    fy/Z  -fy*Y/Z^2 |   |0  c   d|
				*/
				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;

				auto& cp = dirLine->vPoints[cpi];

				Vec2f nx = dirLine->xdir;
				float du = (pt - dirLine->xstart).dot(nx);

				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = vcp[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);
				//φ=exp(-F^2/2v^2)
				float w = exp(-(du - cp.x) * (du - cp.x) / (2 * v * v)) * cp.w * cp.w;

				J += w * (du - cp.x) * j;

				JJ += w * j * j.t();
			}
		}

		const float lambda = 5000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 100.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		int ec = 0;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			dpose.t = pose.R * dt + pose.t;
			dpose.R = pose.R * dR;

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData pose, PoseData& dpose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float v, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, dpose, K, cpoints, v, eps) <= 0)
				return false;
		return true;
	}
};
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	float v;
	int k = 0;
	const float eps = 1e-4;
	float vmax = 0.f, vmin = 0.f;
	Optimizer::PoseData dpose;//T'
	static_cast<Pose&>(dpose) = pose;
	Optimizer::PoseData lpose;//T0
	static_cast<Pose&>(lpose) = pose;
	const int Iv = 10, innerItrs = 3;

	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		std::vector<Point2f>  c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);
	//计算vmax和vmin
	float mean = 0.f, std = 0.f; float Vspace = 0.f;
	dfr.InitV(dpose, K, this->views[curView].contourPoints3d, mean, std);
	Vspace = mean - 0.3 * std;//取前百分之三十左右的点
	vmax = Vspace;
	vmin = Vspace / 1000;
	v = vmax; //vmin = 0.0001f;
	while (1)
	{
		int kstart = k;
		while (k - kstart < Iv)
		{
			//T'=itr(Tk,v);
			curView = this->_getNearestView(lpose.R, lpose.t);
			//检查在当前v下的收敛性，收敛性好那就跳出去，换一个v再迭代
			if (!dfr.update(lpose, dpose, K, this->views[curView].contourPoints3d, innerItrs, v, eps))
				break;
			k++;
			lpose = dpose;
		}
		if (v == vmin)break;
		//lpose = dpose;
		v = max(v / 2, vmin);
		k++;
	}
	pose = lpose;

	return 1.f;
}

//改正iou：
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT, float iou)
{
	float v;
	int k = 0;
	const float eps = 1e-4;
	float vmax = 0.f, vmin = 0.f;
	Optimizer::PoseData dpose;//T'
	static_cast<Pose&>(dpose) = pose;
	Optimizer::PoseData lpose;//T0
	static_cast<Pose&>(lpose) = pose;
	const int Iv = 10, innerItrs = 3;

	Rect curROI; std::vector<Point2f>  c2d;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;


	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);
	//计算vmax和vmin
	curView = this->_getNearestView(lpose.R, lpose.t);

	float avgE = dfr.InitV(dpose, K, this->views[curView].contourPoints3d);
	//avgE *= (1 - metric);
	avgE *= iou;
	vmax = avgE;
	vmin = avgE / 1000;//迭代log1000次
	v = vmax; vmin = 0.0001f;
	while (1)
	{
		int kstart = k;
		while (k - kstart < Iv)
		{
			//T'=itr(Tk,v);
			curView = this->_getNearestView(lpose.R, lpose.t);
			//检查在当前v下的收敛性，收敛性好那就跳出去，换一个v再迭代
			if (!dfr.update(lpose, dpose, K, this->views[curView].contourPoints3d, innerItrs, v, eps))
				break;
			k++;
			lpose = dpose;
		}
		if (v == vmin)break;
		//lpose = dpose;
		v = max(v / 2, vmin);
		k++;
	}
	pose = lpose;

	return 1.f;
}
class Tracker
	:public ITracker
{
	float       _modelScale = 0.001f;
	float       _histogramLearningRate = 0.2f;
	cv::Matx33f _K;
	cv::Matx44f _mProj;
	Frame       _prev;
	Frame       _cur;
	Object  _obj;
	ColorHistogram _colorHistogram;
	bool    _isLocalTracking = false;
	float _iou = 1.f;
	struct FrameInfo
	{
		float   theta;
		float   err;
	};

	std::deque<FrameInfo>  _frameInfo;

	Pose _scalePose(const Pose& pose) const
	{
		return { pose.R, pose.t * _modelScale };
	}
	Pose _descalePose(const Pose& pose) const
	{
		return { pose.R, pose.t * (1.f / _modelScale) };
	}
public:
	virtual void loadModel(const std::string& modelFile, const std::string& argstr)
	{
		_obj.loadModel(modelFile, _modelScale);

		ff::CommandArgSet args(argstr);
		_isLocalTracking = args.getd<bool>("local", false);
	}
	virtual void setUpdateHandler(UpdateHandler* hdl)
	{
		g_uhdl = hdl;
	}
	virtual void reset(const Mat& img, const Pose& pose, const Matx33f& K)
	{
		_cur.img = img.clone();
		_cur.pose = _scalePose(pose);
		_cur.objMask = Mat1b();
		_mProj = cvrm::fromK(K, img.size(), 0.1, 3);
		_K = K;

		_colorHistogram.update(_obj, _cur.img, _cur.pose, _K, 1.f);
		_cur.colorProb = _colorHistogram.getProb(_cur.img);

		Iou();

	}
	virtual void startUpdate(const Mat& img, int fi, Pose gtPose = Pose())
	{
		if (!_prev.img.empty())
		{
			float	theta = getRDiff(_prev.pose.R, _cur.pose.R);
			_frameInfo.push_back({ theta, _cur.err });
			while (_frameInfo.size() > 30)
				_frameInfo.pop_front();
		}

		_prev = _cur;
		_cur.img = img;
		_cur.colorProb = _colorHistogram.getProb(_cur.img);
	}

	template<typename _GetValT>
	static float _getMedianOfLastN(const std::deque<FrameInfo>& frameInfo, int nMax, _GetValT getVal)
	{
		int n = __min((int)frameInfo.size(), nMax);
		std::vector<float> tmp(n);
		int i = 0;
		for (auto itr = frameInfo.rbegin(); i < n; ++itr)
		{
			tmp[i++] = getVal(*itr);
		}
		std::sort(tmp.begin(), tmp.end());
		return tmp[n / 2];
	}

	virtual float update(Pose& pose)
	{
		_cur.pose = _scalePose(pose);

		float thetaT = CV_PI / 8, errT = 1.f; //default values used for only the first 2 frames

		if (!_frameInfo.empty())
		{//estimate from previous frames
			thetaT = _getMedianOfLastN(_frameInfo, 5, [](const FrameInfo& v) {return v.theta; });
			errT = _getMedianOfLastN(_frameInfo, 15, [](const FrameInfo& v) {return v.err; });
		}

		_cur.err = _obj.templ.pro(_K, _cur.pose, _cur.colorProb, thetaT, _isLocalTracking ? FLT_MAX : errT, _iou);
		pose = _descalePose(_cur.pose);
		return _cur.err;
	}

	void Iou()
	{
		int curView = _obj.templ._getNearestView(_cur.pose.R, _cur.pose.t);
		Projector prj(_K, _cur.pose.R, _cur.pose.t);
		vector<Point2f>c2d = prj(_obj.templ.views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });//pose投影的轮廓点
		Rect_<float> rectf = getBoundingBox2D(c2d);
		cv::Mat projectionMask = cv::Mat::zeros(_cur.colorProb.size(), CV_8UC1);
		std::vector<std::vector<cv::Point>> contours;
		contours.push_back(std::vector<cv::Point>(c2d.begin(), c2d.end()));
		cv::fillPoly(projectionMask, contours, cv::Scalar(255));//轮廓点生成mask
		int dilationSize = 5;
		cv::Mat dilatedProjectionMask;
		//dilate
		cv::dilate(projectionMask, dilatedProjectionMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1)));
		double thresholdValue = 0.5;//大于0.5认定为前景
		cv::Mat binaryProbMask;
		cv::threshold(_cur.colorProb, binaryProbMask, thresholdValue, 255, cv::THRESH_BINARY);
		dilatedProjectionMask.convertTo(dilatedProjectionMask, CV_8UC1);
		binaryProbMask.convertTo(binaryProbMask, CV_8UC1);
		cv::Mat intersection = dilatedProjectionMask & binaryProbMask;
		cv::Mat unionArea = dilatedProjectionMask | binaryProbMask;
		int intersectionArea = cv::countNonZero(intersection);
		int unionAreaValue = cv::countNonZero(unionArea);
		_iou = static_cast<double>(intersectionArea) / static_cast<double>(unionAreaValue);
	}

	virtual void endUpdate(const Pose& pose)
	{
		_cur.pose = _scalePose(pose);
		_colorHistogram.update(_obj, _cur.img, _cur.pose, _K, _histogramLearningRate);
	}
};

//IRLS ICP
int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	if (pose.hdl && pose.hdl->test(R, t) <= 0)
		return 0;

	++g_totalUpdates;

	const float fx = K(0, 0), fy = K(1, 1);
	const float cx = K(0, 2), cy = K(1, 2);
	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();

	//构建线性方程组
	Mat1f A(npt * 3, 6);
	Mat1f b(npt * 3, 1);
	int irow = 0;
	std::ofstream fileA("A.txt");
	std::ofstream fileB("B.txt");
	for (int i = 0; i < npt; ++i)
	{
		Point3f P = R * vcp[i].center + t;
		Point3f p = K * P;

		const int x = int(p.x / p.z + 0.5), y = int(p.y / p.z + 0.5);
		const float z = p.z;
		if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
			continue;

		Point3f pn = K * (R * vcp[i].normal + t);
		Vec2f n(pn.x / pn.z - p.x / p.z, pn.y / pn.z - p.y / p.z);
		n = normalize(n);
		const float X = P.x, Y = P.y, Z = P.z;
		Point2f pt(p.x / p.z, p.y / p.z);
		auto* dd = this->getDirData(n);
		if (!dd)
			continue;
		int cpi;
		auto* dirLine = dd->getScanLine(pt, cpi);
		if (!dirLine || cpi < 0)
			continue;
		auto& cp = dirLine->vPoints[cpi];
		Point2f qt = dirLine->xstart + dirLine->xdir * cp.x;
		Point3f q(qt.x, qt.y, 1);//转齐次
		Point3f Q(z * (q.x - cx) / fx, z * (q.y - cy) / fy, z);
		Point3f PQ = P - Q;
		float w = 1 /*/ (pow(fabs((P - Q).x), 2 - alpha) + pow(fabs((P - Q).y), 2 - alpha) + pow(fabs((P - Q).z), 2 - alpha))*/;
		//w *= (cp.w * cp.w);
		float* _b = b.ptr<float>(irow);
		_b[0] = (Q.x - P.x);
		float* a = A.ptr<float>(irow++);
		a[0] = 0; a[1] = w * P.z; a[2] = -w * P.y; a[3] = w; a[4] = 0; a[5] = 0;
		for (int j = 0; j < 6; j++)
			fileA << A.at<float>(irow - 1, j) << " ";
		fileA << "\n";
		fileB << b.at<float>(irow - 1, 0) << " ";
		fileB << "\n";
		_b = b.ptr<float>(irow);
		_b[0] = (Q.y - P.y);
		a = A.ptr<float>(irow++);
		a[0] = -w * P.z; a[1] = 0; a[2] = w * P.x; a[3] = 0; a[4] = w; a[5] = 0;
		for (int j = 0; j < 6; j++)
			fileA << A.at<float>(irow - 1, j) << " ";
		fileA << "\n";
		fileB << b.at<float>(irow - 1, 0) << " ";
		fileB << "\n";
		_b = b.ptr<float>(irow);
		_b[0] = (z - P.z);//q.z=z=p.z
		a = A.ptr<float>(irow++);
		a[0] = w * P.y; a[1] = -w * P.x; a[2] = 0; a[3] = 0; a[4] = 0; a[5] = w;
		for (int j = 0; j < 6; j++)
			fileA << A.at<float>(irow - 1, j) << " ";
		fileA << "\n";
		fileB << b.at<float>(irow - 1, 0) << " ";
		fileB << "\n";

	}
	fileA.close();
	fileB.close();
	// 解线性方程组
	Mat p_;
	if (solve(A, b, p_, DECOMP_SVD))
	{
		// 断言确保解的维度正确
		CV_Assert(p_.rows == 6 && p_.cols == 1);

		// 获取解向量
		const float* p = p_.ptr<float>();
		// 使用罗德里格斯公式更新旋转矩阵
		Matx33f R_ = Matx33f::eye();
		R_(0, 1) = -p[2];
		R_(1, 0) = p[2];
		R_(0, 2) = p[1];
		R_(2, 0) = -p[1];
		R_(1, 2) = -p[0];
		R_(2, 1) = p[0];

		// 使用SVD确保旋转矩阵保持正交
		Matx33f U, Vt;
		Mat W;
		cv::SVDecomp(R_, W, U, Vt);
		R_ = U * Vt;

		// 检查旋转矩阵的行列式，保证其为正值
		float det = cv::determinant(R_);
		if (det < 0) {
			Matx33f B = Matx33f::eye();
			B(2, 2) = det;
			R_ = Vt.t() * B * U.t();
		}

		// 获取平移向量
		Point3f t_(p[3], p[4], p[5]);

		// 更新旋转矩阵和平移向量
		R = Matx33f(R_) * R;
		t = Matx33f(R_) * t + t_;
		pose.R = R;
		pose.t = t;
		float diff = p_.dot(p_);
		return diff < eps ? 0 : 1;
	}
	return 0;
}
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	{
		Projector prj(K, pose.R, pose.t);
		std::vector<Point2f>  c2d = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		Rect_<float> rectf = getBoundingBox2D(c2d);
		curROI = Rect(rectf);
	}

	Optimizer dfr;
	Rect roi = curROI;
	const int dW = 100;
	rectAppend(roi, dW, dW, dW, dW);
	roi = rectOverlapped(roi, Rect(0, 0, curProb.cols, curProb.rows));
	dfr.computeScanLines(curProb, roi);

	//printf("init time=%dms \n", int(clock() - beg));

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;

	//const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int maxItrs = 30;

	dfr._update(dpose, K, this->views[curView].contourPoints3d, maxItrs);

	pose = dpose;

	return 1.f;
}


//gtmask
_cur.pose = pose;
Projector prj(curK, _cur.pose.R, _cur.pose.t);
vector<Point2f>cPts;
int curView = _obj.templ._getNearestView(_cur.pose.R, _cur.pose.t);
if (uint(curView) < _obj.templ.views.size())
{
	Point2f objCenter = prj(_obj.templ.modelCenter);
	auto& view = _obj.templ.views[curView];

	for (auto& cp : view.contourPoints3d)
	{
		Point2f c = prj(cp.center);
		cPts.push_back(c);
	}
	Rect_<float> rectf = getBoundingBox2D(cPts);
	Rect roi = Rect(rectf);
}
cv::Mat foreground = cv::Mat::zeros(cv::Size(640, 512), CV_8UC1);
vector<Point> contourPoints(cPts.begin(), cPts.end());
vector<vector<Point>> contours = { contourPoints };
drawContours(foreground, contours, -1, Scalar(255, 255, 255), FILLED);
_cur.colorProb = foreground;
imwrite("D:/RBOT_dataset/ape/mask" + ff::StrFormat("/%06d-mask.png", fi), foreground);

string maskfile = "D:/RBOT_dataset/ape/mask" + ff::StrFormat("/%06d-mask.png", fi);
Mat mask = imread(maskfile);
cv::Mat coloredMask;
cv::cvtColor(mask, coloredMask, CV_BGR2GRAY);


//icp
int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps)
		{
			Matx33f R = pose.R;
			Point3f t = pose.t;

			if (pose.hdl && pose.hdl->test(R, t) <= 0)
				return 0;

			++g_totalUpdates;

			const float fx = K(0, 0), fy = K(1, 1);
			const float cx = K(0, 2), cy = K(1, 2);
			auto* vcp = &cpoints[0];
			int npt = (int)cpoints.size();

			//构建线性方程组
			Mat1f A(npt * 3, 6);
			Mat1f b(npt * 3, 1);
			int irow = 0;
			for (int i = 0; i < npt; ++i)
			{
				Point3f P = R * vcp[i].center + t;
				Point3f p = K * P;

				const int x = int(p.x / p.z + 0.5), y = int(p.y / p.z + 0.5);
				const float z = p.z;
				if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
					continue;

				Point3f pn = K * (R * vcp[i].normal + t);
				Vec2f n(pn.x / pn.z - p.x / p.z, pn.y / pn.z - p.y / p.z);
				n = normalize(n);
				const float X = P.x, Y = P.y, Z = P.z;
				Point2f pt(p.x / p.z, p.y / p.z);
				auto* dd = this->getDirData(n);
				if (!dd)
					continue;
				int cpi;
				auto* dirLine = dd->getScanLine(pt, cpi);
				if (!dirLine || cpi < 0)
					continue;
				auto& cp = dirLine->vPoints[cpi];
				Point2f qt = dirLine->xstart + dirLine->xdir * cp.x;
				Point3f q(qt.x, qt.y, 1);//转齐次
				Point3f Q(z * (q.x - cx) / fx, z * (q.y - cy) / fy, z);
				//pt.x*z+n.x*z=pn.x   qt.x*z+dirLine->xdir.x*z=qn.x
				Point3f qn(qt.x * z + dirLine->xdir.x * z, qt.y * z + dirLine->xdir.y * z, z);
				Point3f Qn(z * (qn.x - cx) / fx, z * (qn.y - cy) / fy, z);//K*Qn=qn,qn/z-q.x/z=n.x
				Point3f QN = R.inv() * (Qn - t);//三维点
				Vec3f Qnormal(QN - Q);
				Qnormal = normalize(Qnormal);//法向量拿到了

				//Q = P;
				Point3f PQ = P - Q;
				float w = 1 / (pow(fabs((P - Q).x), 2 - alpha) + pow(fabs((P - Q).y), 2 - alpha) + 1e-3);
				w *= (cp.w * cp.w);

				float* _b = b.ptr<float>(irow);
				_b[0] = w * (Q.x - P.x);
				float* a = A.ptr<float>(irow++);
				a[0] = 0; a[1] = w * P.z; a[2] = -w * P.y; a[3] = w; a[4] = 0; a[5] = 0;
				_b = b.ptr<float>(irow);
				_b[0] = w * (Q.y - P.y);
				a = A.ptr<float>(irow++);
				a[0] = -w * P.z; a[1] = 0; a[2] = w * P.x; a[3] = 0; a[4] = w; a[5] = 0;

				_b = b.ptr<float>(irow);
				_b[0] = w * (z - P.z);//q.z=z=p.z
				a = A.ptr<float>(irow++);
				a[0] = w * P.y; a[1] = -w * P.x; a[2] = 0; a[3] = 0; a[4] = 0; a[5] = w;

			}

			A = A(Rect(0, 0, 6, irow)).clone();
			b = b(Rect(0, 0, 1, irow)).clone();
			//cout << A << endl;
			//cout << A1 << endl;
			// 解线性方程组
			Mat p_;
			if (solve(A, b, p_, DECOMP_SVD))
			{
				// 断言确保解的维度正确
				CV_Assert(p_.rows == 6 && p_.cols == 1);

				// 获取解向量
				const float* p = p_.ptr<float>();
				// 使用罗德里格斯公式更新旋转矩阵
				Matx33f R_ = Matx33f::eye();
				R_(0, 1) = -p[2];
				R_(1, 0) = p[2];
				R_(0, 2) = p[1];
				R_(2, 0) = -p[1];
				R_(1, 2) = -p[0];
				R_(2, 1) = p[0];

				// 使用SVD确保旋转矩阵保持正交
				Matx33f U, Vt;
				Mat W;
				cv::SVDecomp(R_, W, U, Vt);
				R_ = U * Vt;

				// 检查旋转矩阵的行列式，保证其为正值
				float det = cv::determinant(R_);
				if (det < 0) {
					Matx33f B = Matx33f::eye();
					B(2, 2) = det;
					R_ = Vt.t() * B * U.t();
				}

				// 获取平移向量
				Point3f t_(p[3], p[4], p[5]);

				// 更新旋转矩阵和平移向量
				R = Matx33f(R_) * R;
				t = Matx33f(R_) * t + t_;
				pose.R = R;
				pose.t = t;
				float diff = p_.dot(p_);

				return diff < eps ? 0 : 1;
			}
			return 0;
		}
//point - plane
int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	if (pose.hdl && pose.hdl->test(R, t) <= 0)
		return 0;

	++g_totalUpdates;

	const float fx = K(0, 0), fy = K(1, 1);
	const float cx = K(0, 2), cy = K(1, 2);
	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();

	//构建线性方程组
	Mat1f A(npt * 3, 6);
	Mat1f b(npt * 3, 1);
	int irow = 0;
	for (int i = 0; i < npt; ++i)
	{
		Point3f P = R * vcp[i].center + t;
		Point3f p = K * P;

		const int x = int(p.x / p.z + 0.5), y = int(p.y / p.z + 0.5);
		const float z = p.z;
		if (uint(x - _roi.x) >= uint(_roi.width) || uint(y - _roi.y) >= uint(_roi.height))
			continue;
		Point3f Pn = R * vcp[i].normal + t;
		Point3f pn = K * (R * vcp[i].normal + t);
		Vec2f n(pn.x / pn.z - p.x / p.z, pn.y / pn.z - p.y / p.z);
		n = normalize(n);
		const float X = P.x, Y = P.y, Z = P.z;
		Point2f pt(p.x / p.z, p.y / p.z);
		auto* dd = this->getDirData(n);
		if (!dd)
			continue;
		int cpi;
		auto* dirLine = dd->getScanLine(pt, cpi);
		if (!dirLine || cpi < 0)
			continue;
		auto& cp = dirLine->vPoints[cpi];
		Point2f qt = dirLine->xstart + dirLine->xdir * cp.x;
		Point3f q(qt.x, qt.y, 1);//转齐次
		Point3f Q(z * (q.x - cx) / fx, z * (q.y - cy) / fy, z);
		////pt.x*z+n.x*z=pn.x   qt.x*z+dirLine->xdir.x*z=qn.x
		//Point3f qn(qt.x * z + dirLine->xdir.x * z, qt.y * z + dirLine->xdir.y * z, z);
		//Point3f Qn(z * (qn.x - cx) / fx, z * (qn.y - cy) / fy, z);//K*Qn=qn,qn/z-q.x/z=n.x
		//Point3f QN = R.inv() * (Qn - t);//三维点
		//Vec3f Qnormal(QN - Q);
		//Qnormal = normalize(Qnormal);//法向量拿到了
		Vec3f Pnormal(Pn - P);
		Pnormal = normalize(Pnormal);
		//Q = P;
		Point3f PQ = P - Q;
		float w = 1 / (pow(fabs((P - Q).x), 2 - alpha) + pow(fabs((P - Q).y), 2 - alpha) + 1e-3);
		w *= (cp.w * cp.w);
		const float nx = Pnormal[0], ny = Pnormal[1], nz = Pnormal[2];
		float* _b = b.ptr<float>(irow);
		_b[0] = w * (nx * P.x + ny * P.y + nz * P.z - nx * Q.x - ny * Q.y - nz * Q.z);
		float* a = A.ptr<float>(irow++);
		a[0] = (nz * Q.y - ny * Q.z) * w;
		a[1] = (nx * Q.z - nz * Q.x) * w;
		a[2] = (ny * Q.x - nx * Q.y) * w;
		a[3] = nx * w;
		a[4] = ny * w;
		a[5] = nz * w;

	}

	A = A(Rect(0, 0, 6, irow)).clone();
	b = b(Rect(0, 0, 1, irow)).clone();
	Mat1f A1 = A.t() * A;
	cout << A << endl;
	cout << A1 << endl;
	// 解线性方程组
	Mat p_;
	if (solve(A, b, p_, DECOMP_SVD))
	{
		// 断言确保解的维度正确
		CV_Assert(p_.rows == 6 && p_.cols == 1);

		// 获取解向量
		const float* p = p_.ptr<float>();
		// 使用罗德里格斯公式更新旋转矩阵
		Matx33f R_ = Matx33f::eye();
		R_(0, 1) = -p[2];
		R_(1, 0) = p[2];
		R_(0, 2) = p[1];
		R_(2, 0) = -p[1];
		R_(1, 2) = -p[0];
		R_(2, 1) = p[0];

		// 使用SVD确保旋转矩阵保持正交
		Matx33f U, Vt;
		Mat W;
		cv::SVDecomp(R_, W, U, Vt);
		R_ = U * Vt;

		// 检查旋转矩阵的行列式，保证其为正值
		float det = cv::determinant(R_);
		if (det < 0) {
			Matx33f B = Matx33f::eye();
			B(2, 2) = det;
			R_ = Vt.t() * B * U.t();
		}

		// 获取平移向量
		Point3f t_(p[3], p[4], p[5]);

		// 更新旋转矩阵和平移向量
		/*R = Matx33f(R_) * R;
		t = Matx33f(R_) * t + t_;*/
		R = Matx33f(R_).inv() * R;
		t = Matx33f(R_).inv() * (t - t_);
		pose.R = R;
		pose.t = t;
		float diff = p_.dot(p_);

		return diff < eps ? 0 : 1;
	}
	return 0;
}