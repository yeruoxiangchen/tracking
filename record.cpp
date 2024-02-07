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
		float tao = 1e-3;
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

//先乘扰动
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

		//Q = P;
		Point3f PQ = P - Q;
		float w = 1 / (pow(fabs((P - Q).x), 2 - alpha) + pow(fabs((P - Q).y), 2 - alpha) + 1e-3);
		w *= (cp.w * cp.w);
		w = 1;
		Point3f Xx = vcp[i].center;
		float* _b = b.ptr<float>(irow);
		_b[0] = w * (Q.x - P.x);
		float* a = A.ptr<float>(irow++);
		//a[0] = 0; a[1] = w * Xx.z; a[2] = -w * Xx.y; a[3] = w; a[4] = 0; a[5] = 0;
		a[0] = R(0, 2) * Xx.y - R(0, 1) * Xx.z;
		a[1] = R(0, 0) * Xx.z - R(0, 2) * Xx.x;
		a[2] = R(0, 1) * Xx.x - R(0, 0) * Xx.y;
		a[3] = R(0, 0);
		a[4] = R(0, 1);
		a[5] = R(0, 2);
		_b = b.ptr<float>(irow);
		_b[0] = w * (Q.y - P.y);
		a = A.ptr<float>(irow++);
		//a[0] = -w * Xx.z; a[1] = 0; a[2] = w * Xx.x; a[3] = 0; a[4] = w; a[5] = 0;
		a[0] = R(1, 2) * Xx.y - R(1, 1) * Xx.z;
		a[1] = R(1, 0) * Xx.z - R(1, 2) * Xx.x;
		a[2] = R(1, 1) * Xx.x - R(1, 0) * Xx.y;
		a[3] = R(1, 0);
		a[4] = R(1, 1);
		a[5] = R(1, 2);
		_b = b.ptr<float>(irow);
		_b[0] = w * (z - P.z);//q.z=z=p.z
		a = A.ptr<float>(irow++);
		//a[0] = w * Xx.y; a[1] = -w * Xx.x; a[2] = 0; a[3] = 0; a[4] = 0; a[5] = w;
		a[0] = R(2, 2) * Xx.y - R(2, 1) * Xx.z;
		a[1] = R(2, 0) * Xx.z - R(2, 2) * Xx.x;
		a[2] = R(2, 1) * Xx.x - R(2, 0) * Xx.y;
		a[3] = R(2, 0);
		a[4] = R(2, 1);
		a[5] = R(2, 2);
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
		/*R =  R * Matx33f(R_);
		t = R * t_;*/
		pose.R = R;
		pose.t = t;
		float diff = p_.dot(p_);

		return diff < eps ? 0 : 1;
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




//LM完整
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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
		if (isinit)
		{
			float tao = 1e-3;
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
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, float thetaT, float errT);
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

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
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
		/*Mat foreground = imread("D:/RBOT_dataset/ape/mask" + ff::StrFormat("/%06d-mask.png", fi), IMREAD_GRAYSCALE);
		foreground.convertTo(foreground, CV_32F, 1 / 255.0);
		_cur.colorProb = foreground;*/
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
	int fi;
	virtual float update(Pose& pose)
	{

		_cur.pose = _scalePose(pose);

		float thetaT = CV_PI / 8, errT = 1.f; //default values used for only the first 2 frames

		if (!_frameInfo.empty())
		{//estimate from previous frames
			thetaT = _getMedianOfLastN(_frameInfo, 5, [](const FrameInfo& v) {return v.theta; });
			errT = _getMedianOfLastN(_frameInfo, 15, [](const FrameInfo& v) {return v.err; });
		}

		_cur.err = _obj.templ.pro(_K, _cur.pose, _cur.colorProb, thetaT, _isLocalTracking ? FLT_MAX : errT);
		pose = _descalePose(_cur.pose);
		return _cur.err;
	}

	virtual void endUpdate(const Pose& pose)
	{
		_cur.pose = _scalePose(pose);
		_colorHistogram.update(_obj, _cur.img, _cur.pose, _K, _histogramLearningRate);
	}
};

//Ransac 简单
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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

	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit, int interval, int idx)
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

		for (int i = idx; i < npt; i += interval)
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
		if (isinit)
		{
			float tao = 1e-3;
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

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps, int interval, int idx)
	{
		float mu = 0.f;
		float v = 2.f;
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_update(pose, K, cpoints, alpha, eps, mu, v, itr == 0, interval, idx) <= 0)
				return false;
		return true;
	}
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat img, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT);

	float eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints, int interval, int idx);
};

inline float Templates::eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints, int interval, int idx)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();
	Mat gray;
	cvtColor(curImg, gray, COLOR_BGR2GRAY);
	Mat Sobelx, Sobely;
	Sobel(gray, Sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, Sobely, CV_32F, 0, 1, 3);
	/*imshow("gray", gray);
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);
	waitKey(0);*/
	Mat img = curImg.clone();

	float res = 0; int continuePt = 0; int intervalnpt = 0;
	for (int i = idx; i < npt; i += interval)
	{

		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
		if (x >= curImg.cols || y >= curImg.rows || x < 0 || y < 0)
		{
			continuePt++;
			continue;
		}
		Point p(x, y);
		Point3f qn = K * (R * vcp[i].normal + t);
		Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
		n = normalize(n);
		float edgex = Sobelx.at<float>(p);
		float edgey = Sobely.at<float>(p);
		Vec2f pixeln(edgex, edgey);
		if (norm(pixeln) == 0)
		{
			continuePt++;
			continue;
		}
		pixeln = normalize(pixeln);
		float cosAngle = n.dot(pixeln) / (norm(n) * norm(pixeln));
		res += fabs(cosAngle);
		intervalnpt++;
		/*line(img, p, p + Point(n)*10, Scalar(0, 255, 0));
		line(img, p, p + Point(pixeln)*10, Scalar(0, 0, 255));
		imshow("img", img);
		waitKey(0);*/
	}
	/*int intervalnpt = npt / interval;
	intervalnpt = (npt % interval > idx ? intervalnpt + 1 : intervalnpt);
	intervalnpt -= continuePt;*/
	res /= intervalnpt;
	return res;
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	//eval(K, pose, curImg, this->views[curView].contourPoints3d, 1, 0);
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

	/*Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;*/
	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3; int interval = 3;
	vector<Optimizer::PoseData>dposes(interval);
	float maxAngle = 0.f; int maxidx = -1;

	for (int i = 0; i < interval; i++)
	{
		auto& dpose = dposes[i];
		static_cast<Pose&>(dpose) = pose;
		for (int itr = 0; itr < outerItrs; ++itr)
		{
			curView = this->_getNearestView(dpose.R, dpose.t);
			if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps, interval, i))
				break;
		}
		curView = this->_getNearestView(dpose.R, dpose.t);
		float evaluate = eval(K, dpose, curImg, this->views[curView].contourPoints3d, interval, i);
		if (maxAngle < evaluate)
		{
			maxAngle = evaluate;
			maxidx = i;
		}
	}

	auto& dpose = dposes[maxidx];

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
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

		_colorHistogram.clear();
		_colorHistogram.update(_obj, _cur.img, _cur.pose, _K, 1.f);
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
		/*Mat foreground = imread("D:/RBOT_dataset/ape/mask" + ff::StrFormat("/%06d-mask.png", fi), IMREAD_GRAYSCALE);
		foreground.convertTo(foreground, CV_32F, 1 / 255.0);
		_cur.colorProb = foreground;*/
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
	int fi;
	virtual float update(Pose& pose)
	{

		_cur.pose = _scalePose(pose);

		float thetaT = CV_PI / 8, errT = 1.f; //default values used for only the first 2 frames

		if (!_frameInfo.empty())
		{//estimate from previous frames
			thetaT = _getMedianOfLastN(_frameInfo, 5, [](const FrameInfo& v) {return v.theta; });
			errT = _getMedianOfLastN(_frameInfo, 15, [](const FrameInfo& v) {return v.err; });
		}

		_cur.err = _obj.templ.pro(_K, _cur.pose, _cur.colorProb, _cur.img, thetaT, _isLocalTracking ? FLT_MAX : errT);
		pose = _descalePose(_cur.pose);
		return _cur.err;
	}

	virtual void endUpdate(const Pose& pose)
	{
		_cur.pose = _scalePose(pose);
		_colorHistogram.update(_obj, _cur.img, _cur.pose, _K, _histogramLearningRate);
	}
};


//Ransac 四种、考虑全部curView
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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

	Mat img;
	struct Compare {
		bool operator()(const pair<float, int>& left, const pair<float, int>& right) {
			return left.first < right.first;  // 保持最大堆，当堆中超过规定的点个数之后把最大的弹出去
		};
	};
	static const int NHeaps = 4;//需要计算的轮廓点组数
	/*
	* int是完整cpoints中点的索引
	* 0 du-cp
	* 1 向量模
	* 2 方向差，cos角度
	* 3 w
	*/
	std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Compare> MaxHeaps[NHeaps];
	vector<int>EvalPoints[NHeaps];
	void extractcps(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float ratio = 0.5f, int idx = -1)
	{
		if (idx == -1)//粗位姿估计的时候，idx=-1，全部EvalPoints清零
		{
			for (int i = 0; i < NHeaps; i++)
			{
				EvalPoints[i].clear();
				while (!MaxHeaps[i].empty())MaxHeaps[i].pop();
			}
		}
		else {//refine过程中，当前ransac的部分位姿变化导致视角改变，需要重新更新当前idx的EvalPoints
			EvalPoints[idx].clear();
			while (!MaxHeaps[idx].empty())MaxHeaps[idx].pop();
		}

		Matx33f R = pose.R;
		Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		int RansacNpt = npt * ratio;//用于后续refine的点数

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; i += 1)
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

				/*
				* 0 du-cp
				* 1 向量模
				* 2 方向差，角度差
				* 3 w
				* 都是越小越好
				*/

				Point2f cpt = dirLine->xstart + cp.x * dirLine->xdir;//轮廓点
				float heapval[NHeaps] = {
					du - cp.x,//0
					norm(pt - cpt),//1
					-(n.dot(dirLine->xdir) / (norm(n) * norm(dirLine->xdir))),//2
					-cp.w * cp.w//3
				};
				if (idx == -1)
				{
					for (int hp = 0; hp < NHeaps; hp++)
					{
						auto& heap = MaxHeaps[hp];
						auto& val = heapval[hp];
						heap.push({ val,i });
						if (heap.size() > RansacNpt)
							heap.pop();
					}
				}
				else
				{
					auto& heap = MaxHeaps[idx];
					auto& val = heapval[idx];
					heap.push({ val,i });
					if (heap.size() > RansacNpt)
						heap.pop();
				}

			}
		}
		if (idx == -1)
		{
			for (int i = 0; i < NHeaps; i++)
			{
				auto& heap = MaxHeaps[i];
				while (!heap.empty())
				{
					EvalPoints[i].push_back(heap.top().second);
					heap.pop();
				}
			}
		}
		else
		{
			auto& heap = MaxHeaps[idx];
			while (!heap.empty())
			{
				EvalPoints[idx].push_back(heap.top().second);
				heap.pop();
			}
		}


	}
	//完整cpoints计算初始位姿
	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		Mat imgclone = img.clone();

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

		for (int i = 0; i < npt; i += 1)
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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

	//四组轮廓点
	int _RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit, int idx)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		//Mat imgclone = img.clone();

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		auto& heap = MaxHeaps[idx];//当前的最大堆
		auto EvalPoint = EvalPoints[idx];

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);
		vector<int>nptidx;
		for (int k = 0; k < EvalPoint.size(); k++)
		{
			int i = EvalPoint[k];
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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
			for (int k = 0; k < EvalPoint.size(); k++)
			{
				auto i = EvalPoint[k];
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

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps, int idx, bool ViewChange)
	{
		float mu = 0.f;
		float v = 2.f;
		if (!ViewChange)//视角没变，之前的轮廓点可以继续用，
		{
			for (int itr = 0; itr < maxItrs; ++itr)
				if (this->_RefineUpdate(pose, K, cpoints, alpha, eps, mu, v, itr == 0, idx) <= 0)
					return false;
		}
		else//视角变了
		{
			for (int itr = 0; itr < maxItrs; ++itr) {
				if (this->_update(pose, K, cpoints, alpha, eps, mu, v, itr == 0) <= 0) {
					extractcps(pose, K, cpoints, idx);
					return false;
				}
			}
		}
		return true;
	}
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat img, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT);

	float eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline float Templates::eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();
	Mat gray;
	cvtColor(curImg, gray, COLOR_BGR2GRAY);
	Mat Sobelx, Sobely;
	Sobel(gray, Sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, Sobely, CV_32F, 0, 1, 3);
	/*imshow("gray", gray);
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);s
	waitKey(0);*/
	Mat img = curImg.clone();

	float res = 0; int continuePt = 0;
	for (int i = 0; i < npt; i += 1)
	{

		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
		if (x >= curImg.cols || y >= curImg.rows || x < 0 || y < 0)
		{
			continuePt++;
			continue;
		}
		Point p(x, y);
		Point3f qn = K * (R * vcp[i].normal + t);
		Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
		n = normalize(n);
		float edgex = Sobelx.at<float>(p);
		float edgey = Sobely.at<float>(p);
		Vec2f pixeln(edgex, edgey);
		if (norm(pixeln) == 0)
		{
			continuePt++;
			continue;
		}
		pixeln = normalize(pixeln);
		float cosAngle = n.dot(pixeln) / (norm(n) * norm(pixeln));
		res += fabs(cosAngle);
		/*line(img, p, p + Point(n)*10, Scalar(0, 255, 0));
		line(img, p, p + Point(pixeln)*10, Scalar(0, 0, 255));
		imshow("img", img);
		waitKey(0);*/
	}
	int finalnpt = npt - continuePt;
	res /= finalnpt;
	return res;
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	//eval(K, pose, curImg, this->views[curView].contourPoints3d, 1, 0);
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
	dfr.img = curImg.clone();

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;
	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3; ;
	vector<Optimizer::PoseData>dposes(dfr.NHeaps);
	float maxAngle = 0.f; int maxidx = -1;

	//全部轮廓点
	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	dfr.extractcps(dpose, K, this->views[curView].contourPoints3d);
	//开始refine
	for (int i = 0; i < dfr.NHeaps; i++)
	{
		auto& posed = dposes[i];
		posed = dpose;
		for (int itr = 0; itr < outerItrs; ++itr)
		{
			bool ViewChange = 0;
			int nowView = this->_getNearestView(posed.R, posed.t);
			if (curView != nowView) {
				ViewChange = 1;
				curView = nowView;
			}
			if (!dfr.RefineUpdate(posed, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps, i, ViewChange))
				break;
		}
		curView = this->_getNearestView(posed.R, posed.t);
		float evaluate = eval(K, posed, curImg, this->views[curView].contourPoints3d);
		if (evaluate > maxAngle)
		{
			maxAngle = evaluate;
			maxidx = i;
		}
	}

	dpose = dposes[maxidx];

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
}

//Ransac 四种，完全不考虑curView
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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

	Mat img;
	struct Compare {
		bool operator()(const pair<float, int>& left, const pair<float, int>& right) {
			return left.first < right.first;  // 保持最大堆，当堆中超过规定的点个数之后把最大的弹出去
		};
	};
	static const int NHeaps = 4;//需要计算的轮廓点组数
	/*
	* int是完整cpoints中点的索引
	* 0 du-cp
	* 1 向量模
	* 2 方向差，cos角度
	* 3 w
	*/
	std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Compare> MaxHeaps[NHeaps];
	vector<int>EvalPoints[NHeaps];
	void extractcps(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float ratio = 0.5f)
	{
		Matx33f R = pose.R;
		Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		int RansacNpt = npt * ratio;//用于后续refine的点数

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; i += 1)
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

				/*
				* 0 du-cp
				* 1 向量模
				* 2 方向差，角度差
				* 3 w
				* 都是越小越好,w需要取反
				*/
				/*if (i==68)
				{
					cout << 1 << endl;
				}*/
				Point2f cpt = dirLine->xstart + cp.x * dirLine->xdir;//轮廓点
				float heapval[NHeaps] = {
					du - cp.x,//0
					norm(pt - cpt),//1
					-(n.dot(dirLine->xdir) / (norm(n) * norm(dirLine->xdir))),//2
					-cp.w * cp.w//3
				};
				for (int hp = 0; hp < NHeaps; hp++)
				{
					auto& heap = MaxHeaps[hp];
					auto& val = heapval[hp];
					heap.push({ val,i });
					if (heap.size() > RansacNpt)
						heap.pop();
				}

			}
		}
		for (int i = 0; i < NHeaps; i++)
		{
			auto& heap = MaxHeaps[i];
			while (!heap.empty())
			{
				EvalPoints[i].push_back(heap.top().second);
				heap.pop();
			}
		}

	}
	//完整cpoints计算初始位姿
	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		Mat imgclone = img.clone();

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

		for (int i = 0; i < npt; i += 1)
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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

	//四组轮廓点
	int _RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit, int idx)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		//Mat imgclone = img.clone();

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		auto& heap = MaxHeaps[idx];//当前的最大堆
		auto EvalPoint = EvalPoints[idx];

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);
		vector<int>nptidx;
		for (int k = 0; k < EvalPoint.size(); k++)
		{
			int i = EvalPoint[k];
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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
			for (int k = 0; k < EvalPoint.size(); k++)
			{
				auto i = EvalPoint[k];
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

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps, int idx)
	{
		float mu = 0.f;
		float v = 2.f;
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_RefineUpdate(pose, K, cpoints, alpha, eps, mu, v, itr == 0, idx) <= 0)
				return false;
		return true;
	}
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat img, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT);

	float eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline float Templates::eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();
	Mat gray;
	cvtColor(curImg, gray, COLOR_BGR2GRAY);
	Mat Sobelx, Sobely;
	Sobel(gray, Sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, Sobely, CV_32F, 0, 1, 3);
	/*imshow("gray", gray);
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);s
	waitKey(0);*/
	Mat img = curImg.clone();

	float res = 0; int continuePt = 0;
	for (int i = 0; i < npt; i += 1)
	{

		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
		if (x >= curImg.cols || y >= curImg.rows || x < 0 || y < 0)
		{
			continuePt++;
			continue;
		}
		Point p(x, y);
		Point3f qn = K * (R * vcp[i].normal + t);
		Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
		n = normalize(n);
		float edgex = Sobelx.at<float>(p);
		float edgey = Sobely.at<float>(p);
		Vec2f pixeln(edgex, edgey);
		if (norm(pixeln) == 0)
		{
			continuePt++;
			continue;
		}
		pixeln = normalize(pixeln);
		float cosAngle = n.dot(pixeln) / (norm(n) * norm(pixeln));
		res += fabs(cosAngle);
		/*line(img, p, p + Point(n)*10, Scalar(0, 255, 0));
		line(img, p, p + Point(pixeln)*10, Scalar(0, 0, 255));
		imshow("img", img);
		waitKey(0);*/
	}
	int finalnpt = npt - continuePt;
	res /= finalnpt;
	return res;
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	//eval(K, pose, curImg, this->views[curView].contourPoints3d, 1, 0);
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
	dfr.img = curImg.clone();

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;
	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3; ;
	vector<Optimizer::PoseData>dposes(dfr.NHeaps);
	float maxAngle = 0.f; int maxidx = -1;

	//全部轮廓点
	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	dfr.extractcps(dpose, K, this->views[curView].contourPoints3d);
	//开始refine
	for (int i = 0; i < dfr.NHeaps; i++)
	{
		auto& posed = dposes[i];
		posed = dpose;
		for (int itr = 0; itr < outerItrs; ++itr)
		{
			curView = this->_getNearestView(posed.R, posed.t);
			if (!dfr.RefineUpdate(posed, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps, i))
				break;
		}
		curView = this->_getNearestView(posed.R, posed.t);
		float evaluate = eval(K, posed, curImg, this->views[curView].contourPoints3d);
		if (evaluate > maxAngle)
		{
			maxAngle = evaluate;
			maxidx = i;
		}
	}

	dpose = dposes[maxidx];

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
}

//Ransac 四种，考虑curView V2
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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

	Mat img;
	struct Compare {
		bool operator()(const pair<float, int>& left, const pair<float, int>& right) {
			return left.first < right.first;  // 保持最大堆，当堆中超过规定的点个数之后把最大的弹出去
		};
	};
	static const int NHeaps = 4;//需要计算的轮廓点组数
	/*
	* int是完整cpoints中点的索引
	* 0 du-cp
	* 1 向量模
	* 2 方向差，cos角度
	* 3 w
	*/
	std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Compare> MaxHeaps[NHeaps];
	vector<int>EvalPoints[NHeaps];
	void extractcps(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float ratio = 0.5f)
	{
		Matx33f R = pose.R;
		Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		int RansacNpt = npt * ratio;//用于后续refine的点数

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; i += 1)
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

				/*
				* 0 du-cp
				* 1 向量模
				* 2 方向差，角度差
				* 3 w
				* 都是越小越好,w需要取反
				*/
				/*if (i==68)
				{
					cout << 1 << endl;
				}*/
				Point2f cpt = dirLine->xstart + cp.x * dirLine->xdir;//轮廓点
				float heapval[NHeaps] = {
					du - cp.x,//0
					norm(pt - cpt),//1
					-(n.dot(dirLine->xdir) / (norm(n) * norm(dirLine->xdir))),//2
					-cp.w * cp.w//3
				};
				for (int hp = 0; hp < NHeaps; hp++)
				{
					auto& heap = MaxHeaps[hp];
					auto& val = heapval[hp];
					heap.push({ val,i });
					if (heap.size() > RansacNpt)
						heap.pop();
				}

			}
		}
		for (int i = 0; i < NHeaps; i++)
		{
			auto& heap = MaxHeaps[i];
			while (!heap.empty())
			{
				EvalPoints[i].push_back(heap.top().second);
				heap.pop();
			}
		}

	}
	//完整cpoints计算初始位姿
	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		Mat imgclone = img.clone();

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

		for (int i = 0; i < npt; i += 1)
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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

	//四组轮廓点
	int _RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit, int idx)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		//Mat imgclone = img.clone();

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		auto& heap = MaxHeaps[idx];//当前的最大堆
		auto EvalPoint = EvalPoints[idx];

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);
		vector<int>nptidx;
		for (int k = 0; k < EvalPoint.size(); k++)
		{
			int i = EvalPoint[k];
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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
			for (int k = 0; k < EvalPoint.size(); k++)
			{
				auto i = EvalPoint[k];
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

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps, int idx, bool ViewChange)
	{
		float mu = 0.f;
		float v = 2.f;
		if (!ViewChange) {
			for (int itr = 0; itr < maxItrs; ++itr)
				if (this->_RefineUpdate(pose, K, cpoints, alpha, eps, mu, v, itr == 0, idx) <= 0)
					return false;
		}
		else {
			for (int itr = 0; itr < maxItrs; ++itr)
				if (this->_update(pose, K, cpoints, alpha, eps, mu, v, itr == 0) <= 0)
					extractcps(pose, K, cpoints);
			return false;
		}

		return true;
	}
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat img, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT);

	float eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline float Templates::eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();
	Mat gray;
	cvtColor(curImg, gray, COLOR_BGR2GRAY);
	Mat Sobelx, Sobely;
	Sobel(gray, Sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, Sobely, CV_32F, 0, 1, 3);
	/*imshow("gray", gray);
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);s
	waitKey(0);*/
	Mat img = curImg.clone();

	float res = 0; int continuePt = 0;
	for (int i = 0; i < npt; i += 1)
	{

		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
		if (x >= curImg.cols || y >= curImg.rows || x < 0 || y < 0)
		{
			continuePt++;
			continue;
		}
		Point p(x, y);
		Point3f qn = K * (R * vcp[i].normal + t);
		Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
		n = normalize(n);
		float edgex = Sobelx.at<float>(p);
		float edgey = Sobely.at<float>(p);
		Vec2f pixeln(edgex, edgey);
		if (norm(pixeln) == 0)
		{
			continuePt++;
			continue;
		}
		pixeln = normalize(pixeln);
		float cosAngle = n.dot(pixeln) / (norm(n) * norm(pixeln));
		res += fabs(cosAngle);
		/*line(img, p, p + Point(n)*10, Scalar(0, 255, 0));
		line(img, p, p + Point(pixeln)*10, Scalar(0, 0, 255));
		imshow("img", img);
		waitKey(0);*/
	}
	int finalnpt = npt - continuePt;
	res /= finalnpt;
	return res;
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	//eval(K, pose, curImg, this->views[curView].contourPoints3d, 1, 0);
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
	dfr.img = curImg.clone();

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;
	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3; ;
	vector<Optimizer::PoseData>dposes(dfr.NHeaps);
	float maxAngle = 0.f; int maxidx = -1;

	//全部轮廓点
	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	dfr.extractcps(dpose, K, this->views[curView].contourPoints3d);
	//开始refine
	for (int i = 0; i < dfr.NHeaps; i++)
	{
		auto& posed = dposes[i];
		posed = dpose;
		for (int itr = 0; itr < outerItrs; ++itr)
		{
			bool ViewChange = 0;
			int nowView = this->_getNearestView(posed.R, posed.t);
			if (curView != nowView)
			{
				ViewChange = 1;
				curView = nowView;
			}
			//curView = this->_getNearestView(posed.R, posed.t);
			if (!dfr.RefineUpdate(posed, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps, i, ViewChange))
				break;
		}
		curView = this->_getNearestView(posed.R, posed.t);
		float evaluate = eval(K, posed, curImg, this->views[curView].contourPoints3d);
		if (evaluate > maxAngle)
		{
			maxAngle = evaluate;
			maxidx = i;
		}
	}

	dpose = dposes[maxidx];

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
}


//Ransac 四种，考虑curView v3
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
					//if ((data[i] != data[i - 1] && data[i]==255) || data[i] > data[i + 1])
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
		/*Mat1f showprob = abs(edgeProb) / maxElem((Mat1f)abs(edgeProb));
		imshow("asd", showprob);
		waitKey(0);*/
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
			//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
				//for (int i = r.start; i < r.end; ++i)
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

	Mat img;
	struct Compare {
		bool operator()(const pair<float, int>& left, const pair<float, int>& right) {
			return left.first < right.first;  // 保持最大堆，当堆中超过规定的点个数之后把最大的弹出去
		};
	};
	static const int NHeaps = 4;//需要计算的轮廓点组数
	/*
	* int是完整cpoints中点的索引
	* 0 du-cp
	* 1 向量模
	* 2 方向差，cos角度
	* 3 w
	*/
	std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Compare> MaxHeaps[NHeaps];
	vector<int>EvalPoints[NHeaps];
	void extractcps(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int idx = -1, float ratio = 0.5f)
	{
		if (idx == -1)//粗位姿估计的时候，idx=-1，全部EvalPoints清零
		{
			for (int i = 0; i < NHeaps; i++)
			{
				EvalPoints[i].clear();
				while (!MaxHeaps[i].empty())MaxHeaps[i].pop();
			}
		}
		else {//refine过程中，当前ransac的部分位姿变化导致视角改变，需要重新更新当前idx的EvalPoints
			EvalPoints[idx].clear();
			while (!MaxHeaps[idx].empty())MaxHeaps[idx].pop();
		}

		Matx33f R = pose.R;
		Point3f t = pose.t;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		int RansacNpt = npt * ratio;//用于后续refine的点数

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (int i = 0; i < npt; i += 1)
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

				/*
				* 0 du-cp
				* 1 向量模
				* 2 方向差，角度差
				* 3 w
				* 都是越小越好
				*/

				Point2f cpt = dirLine->xstart + cp.x * dirLine->xdir;//轮廓点
				float heapval[NHeaps] = {
					du - cp.x,//0
					norm(pt - cpt),//1
					-(n.dot(dirLine->xdir) / (norm(n) * norm(dirLine->xdir))),//2
					-cp.w * cp.w//3
				};
				if (idx == -1)
				{
					for (int hp = 0; hp < NHeaps; hp++)
					{
						auto& heap = MaxHeaps[hp];
						auto& val = heapval[hp];
						heap.push({ val,i });
						if (heap.size() > RansacNpt)
							heap.pop();
					}
				}
				else
				{
					auto& heap = MaxHeaps[idx];
					auto& val = heapval[idx];
					heap.push({ val,i });
					if (heap.size() > RansacNpt)
						heap.pop();
				}

			}
		}
		if (idx == -1)
		{
			for (int i = 0; i < NHeaps; i++)
			{
				auto& heap = MaxHeaps[i];
				while (!heap.empty())
				{
					EvalPoints[i].push_back(heap.top().second);
					heap.pop();
				}
			}
		}
		else
		{
			auto& heap = MaxHeaps[idx];
			while (!heap.empty())
			{
				EvalPoints[idx].push_back(heap.top().second);
				heap.pop();
			}
		}


	}
	//完整cpoints计算初始位姿
	int _update(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		Mat imgclone = img.clone();

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

		for (int i = 0; i < npt; i += 1)
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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

	//四组轮廓点
	int _RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float alpha, float eps, float mu, float v, bool isinit, int idx)
	{

		Matx33f R = pose.R;
		Point3f t = pose.t;

		//Mat imgclone = img.clone();

		++g_totalUpdates;

		const float fx = K(0, 0), fy = K(1, 1);

		auto* vcp = &cpoints[0];
		int npt = (int)cpoints.size();
		auto& heap = MaxHeaps[idx];//当前的最大堆
		auto EvalPoint = EvalPoints[idx];

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);
		vector<int>nptidx;
		for (int k = 0; k < EvalPoint.size(); k++)
		{
			int i = EvalPoint[k];
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

				/*{
					Point2f cP = dirLine->xstart + dirLine->xdir * cp.x;
					circle(imgclone, cP, 1, Scalar(0, 255, 0));
					circle(imgclone, pt, 1, Scalar(255, 0, 0));
				}*/

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

		/*imshow("img", imgclone);
		waitKey(0);*/

		if (isinit)
		{
			float tao = 1e-3;
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
			for (int k = 0; k < EvalPoint.size(); k++)
			{
				auto i = EvalPoint[k];
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

			return diff < eps* eps ? 0 : 1;
		}

		return 0;
	}
	bool RefineUpdate(PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float alpha, float eps, int idx, bool ViewChange)
	{
		float mu = 0.f;
		float v = 2.f;
		if (!ViewChange)//视角没变，之前的轮廓点可以继续用，
		{
			for (int itr = 0; itr < maxItrs; ++itr)
				if (this->_RefineUpdate(pose, K, cpoints, alpha, eps, mu, v, itr == 0, idx) <= 0)
					return false;
		}
		else//视角变了
		{
			for (int itr = 0; itr < maxItrs; ++itr) {
				if (this->_update(pose, K, cpoints, alpha, eps, mu, v, itr == 0) <= 0) {
					extractcps(pose, K, cpoints, idx);
					return false;
				}
			}
		}
		return true;
	}
};



struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;

	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat img, float thetaT, float errT)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT);

	float eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline float Templates::eval(const Matx33f& K, Pose& pose, const Mat curImg, vector<CPoint>cpoints)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;

	auto* vcp = &cpoints[0];
	int npt = (int)cpoints.size();
	Mat gray;
	cvtColor(curImg, gray, COLOR_BGR2GRAY);
	Mat Sobelx, Sobely;
	Sobel(gray, Sobelx, CV_32F, 1, 0, 3);
	Sobel(gray, Sobely, CV_32F, 0, 1, 3);
	/*imshow("gray", gray);
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);s
	waitKey(0);*/
	Mat img = curImg.clone();

	float res = 0; int continuePt = 0;
	for (int i = 0; i < npt; i += 1)
	{

		Point3f Q = R * vcp[i].center + t;
		Point3f q = K * Q;
		const int x = int(q.x / q.z + 0.5), y = int(q.y / q.z + 0.5);
		if (x >= curImg.cols || y >= curImg.rows || x < 0 || y < 0)
		{
			continuePt++;
			continue;
		}
		Point p(x, y);
		Point3f qn = K * (R * vcp[i].normal + t);
		Vec2f n(qn.x / qn.z - q.x / q.z, qn.y / qn.z - q.y / q.z);
		n = normalize(n);
		float edgex = Sobelx.at<float>(p);
		float edgey = Sobely.at<float>(p);
		Vec2f pixeln(edgex, edgey);
		if (norm(pixeln) == 0)
		{
			continuePt++;
			continue;
		}
		pixeln = normalize(pixeln);
		float cosAngle = n.dot(pixeln) / (norm(n) * norm(pixeln));
		res += fabs(cosAngle);
		/*line(img, p, p + Point(n)*10, Scalar(0, 255, 0));
		line(img, p, p + Point(pixeln)*10, Scalar(0, 0, 255));
		imshow("img", img);
		waitKey(0);*/
	}
	int finalnpt = npt - continuePt;
	res /= finalnpt;
	return res;
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat curImg, float thetaT, float errT)
{
	Rect curROI;
	int curView = this->_getNearestView(pose.R, pose.t);
	//eval(K, pose, curImg, this->views[curView].contourPoints3d, 1, 0);
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
	dfr.img = curImg.clone();

	Optimizer::PoseData dpose;
	static_cast<Pose&>(dpose) = pose;
	const float alpha = 0.125f, alphaNonLocal = 0.75f, eps = 1e-4f;
	const int outerItrs = 10, innerItrs = 3; ;
	vector<Optimizer::PoseData>dposes(dfr.NHeaps);
	float maxAngle = 0.f; int maxidx = -1;

	//全部轮廓点
	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	dfr.extractcps(dpose, K, this->views[curView].contourPoints3d);
	//开始refine
	for (int i = 0; i < dfr.NHeaps; i++)
	{
		auto& posed = dposes[i];
		posed = dpose;
		for (int itr = 0; itr < outerItrs; ++itr)
		{
			bool ViewChange = 0;
			int nowView = this->_getNearestView(posed.R, posed.t);
			if (curView != nowView) {
				ViewChange = 1;
				curView = nowView;
			}
			if (!dfr.RefineUpdate(posed, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps, i, ViewChange))
				break;
		}
		curView = this->_getNearestView(posed.R, posed.t);
		float evaluate = eval(K, posed, curImg, this->views[curView].contourPoints3d);
		if (evaluate > maxAngle)
		{
			maxAngle = evaluate;
			maxidx = i;
		}
	}

	dpose = dposes[maxidx];

	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);

	pose = dpose;

	return errMin;
}

//更改评分方法的Nonlocal
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


struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;
	enum { N_LAYERS = 2 };
	Mat2f         _grad;
	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void getProjectedContours(const Matx33f& K, const Pose& pose, std::vector<Point2f>& points, std::vector<Point2f>* normals)
	{
		int curView = this->_getNearestView(pose.R, pose.t);

		Projector prj(K, pose.R, pose.t);
		points = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		if (normals)
		{
			*normals = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.normal; });
			for (int i = 0; i < (int)points.size(); ++i)
			{
				auto n = (*normals)[i] - points[i];
				(*normals)[i] = normalize(Vec2f(n));
			}
		}
	}

public:
	void initgrad(const Mat& img);
	float getScore(const Pose& pose, const Matx33f& K, float dotT = 0.9f);
	void build(CVRModel& model);

	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float angleMax)
	{
		return this->pro1(K, pose, curProb, img, thetaT, angleMax);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float angleMax);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline void Templates::initgrad(const Mat& img)
{
	Mat1b gray = cv::convertBGRChannels(img, 1);

	Mat1f dx, dy;

	for (int l = 0; l < N_LAYERS; ++l)
	{
		Mat1f ldx, ldy;
		cv::Sobel(gray, ldx, CV_32F, 1, 0);
		cv::Sobel(gray, ldy, CV_32F, 0, 1);

		if (l != 0)
		{
			dx += imscale(ldx, img.size(), INTER_LINEAR);
			dy += imscale(ldy, img.size(), INTER_LINEAR);
		}
		else
		{
			dx = ldx; dy = ldy;
		}
		if (l != N_LAYERS - 1)
			gray = imscale(gray, 0.5);
	}
	_grad = cv::mergeChannels(dx, dy);
}
inline float Templates::getScore(const Pose& pose, const Matx33f& K, float dotT)
{
	std::vector<Point2f>  points, normals;
	getProjectedContours(K, pose, points, &normals);

	float wsum = 1e-6, score = 0;
	int   nmatch = 0;
	for (size_t i = 0; i < points.size(); ++i)
	{
		int x = int(points[i].x + 0.5f), y = int(points[i].y + 0.5f);
		if (uint(x) < uint(_grad.cols) && uint(y) < uint(_grad.rows))
		{
			const float* g = _grad.ptr<float>(y, x);
			float w = sqrt(g[0] * g[0] + g[1] * g[1]) + 1e-6f;
			float dot = (g[0] * normals[i].x + g[1] * normals[i].y) / w;
			dot = fabs(dot);
			if (dot > dotT)
			{
				score += dot * w;
				wsum += w;
				nmatch++;
			}
		}
	}
	return score / wsum * (float(nmatch) / float(points.size()));
}
inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float angleMax)
{
	initgrad(img);
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

	//float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);
	float evaluate = getScore(dpose, K);
	if (evaluate < angleMax)
	{
		const auto R0 = dpose.R;

		const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;

		const float dc = thetaT / N;

		const int subDiv = 3;
		const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		RegionTrajectory traj(Size(subRegionSize, subRegionSize), dc / subDiv);

		Mat1b label = Mat1b::zeros(2 * N + 1, 2 * N + 1);
		struct DSeed
		{
			Point coord;
			//bool  isLocalMinima;
		};
		std::deque<DSeed>  seeds;
		seeds.push_back({ Point(N,N)/*,true*/ });
		label(N, N) = 1;


		auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			{
				if (label(y, x) == 0)
				{
					label(y, x) = 1;
					seeds.push_back({ Point(x, y)/*, false*/ });
				}
			}
		};

		while (!seeds.empty())
		{
			auto curSeed = seeds.front();
			seeds.pop_front();

			checkAdd(curSeed, 0, -1);
			checkAdd(curSeed, -1, 0);
			checkAdd(curSeed, 1, 0);
			checkAdd(curSeed, 0, 1);

			auto dR = theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

			auto dposex = dpose;
			dposex.R = dR * R0;

			Point2f start = dir2Theta(viewDirFromR(dposex.R * R0.t()));
			for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			{
				if (itr % innerItrs == 0)
					curView = this->_getNearestView(dposex.R, dposex.t);

				if (!dfr.update(dposex, K, this->views[curView].contourPoints3d, 1, alphaNonLocal, eps))
					break;

				Point2f end = dir2Theta(viewDirFromR(dposex.R * R0.t()));
				if (traj.addStep(start, end))
					break;
				start = end;
			}
			{
				curView = this->_getNearestView(dposex.R, dposex.t);
				//float err = dfr.calcError(dposex, K, this->views[curView].contourPoints3d, alpha);
				float angle = getScore(dposex, K);
				if (angle > evaluate)
				{
					evaluate = angle;
					dpose = dposex;
				}
				if (angleMax < evaluate)
					break;
			}
		}
	}

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}

	pose = dpose;

	return evaluate;
}

//加上边缘响应的Nonlocal
struct EdgeResponse
{
	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
	Mat1f gradf;
public:
	EdgeResponse(Mat img)
	{
		gradf = getImageField(img);
	}
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float eps)
	{
		Mat1f f = gradf;

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		const Matx33f R = pose.R;
		const Point3f t = pose.t;
		int npt = cpoints.size();

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			J += wf * j;

			JJ += w * j * j.t();
		}

		const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			pose.t = pose.R * dt + pose.t;
			pose.R = pose.R * dR;


			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float eps)
	{
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_edgeupdate(pose, K, cpoints, eps) <= 0)
				return false;
		return true;
	}
};


struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;
	enum { N_LAYERS = 2 };
	Mat2f         _grad;
	Mat img0;
	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void getProjectedContours(const Matx33f& K, const Pose& pose, std::vector<Point2f>& points, std::vector<Point2f>* normals)
	{
		int curView = this->_getNearestView(pose.R, pose.t);

		Projector prj(K, pose.R, pose.t);
		points = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		if (normals)
		{
			*normals = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.normal; });
			for (int i = 0; i < (int)points.size(); ++i)
			{
				auto n = (*normals)[i] - points[i];
				(*normals)[i] = normalize(Vec2f(n));
			}
		}
	}

public:
	void build(CVRModel& model);
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT, fi, gtpose);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
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

	/******************************************************************************/

	//std::ofstream file("pose_matrix2.txt", std::ios::app); // std::ios::app 用于追加到文件末尾

	//// 设置输出格式
	//file << std::fixed << std::setprecision(2);

	//// 写入矩阵
	//file << "init matrix:" << endl;
	//file << "R:" << endl;
	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		file << std::setw(6) << dpose.R(i, j) << " ";
	//	}
	//	file << std::endl;
	//}
	//file << "t:" << endl;
	//for (int i = 0; i < 3; i++) {
	//	file << std::setw(6) << dpose.t[i] << " ";
	//}
	//file << std::endl;

	//// 关闭文件
	//file.close();

	/****************************************************************************************/


	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);
	if (errMin > errT)
	{
		const auto R0 = dpose.R;

		const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;

		const float dc = thetaT / N;

		const int subDiv = 3;
		const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		RegionTrajectory traj(Size(subRegionSize, subRegionSize), dc / subDiv);

		Mat1b label = Mat1b::zeros(2 * N + 1, 2 * N + 1);
		struct DSeed
		{
			Point coord;
			//bool  isLocalMinima;
		};
		std::deque<DSeed>  seeds;
		seeds.push_back({ Point(N,N)/*,true*/ });
		label(N, N) = 1;


		auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			{
				if (label(y, x) == 0)
				{
					label(y, x) = 1;
					seeds.push_back({ Point(x, y)/*, false*/ });
				}
			}
		};

		while (!seeds.empty())
		{
			auto curSeed = seeds.front();
			seeds.pop_front();

			checkAdd(curSeed, 0, -1);
			checkAdd(curSeed, -1, 0);
			checkAdd(curSeed, 1, 0);
			checkAdd(curSeed, 0, 1);

			auto dR = theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

			auto dposex = dpose;
			dposex.R = dR * R0;

			Point2f start = dir2Theta(viewDirFromR(dposex.R * R0.t()));
			for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			{
				if (itr % innerItrs == 0)
					curView = this->_getNearestView(dposex.R, dposex.t);

				if (!dfr.update(dposex, K, this->views[curView].contourPoints3d, 1, alphaNonLocal, eps))
					break;

				Point2f end = dir2Theta(viewDirFromR(dposex.R * R0.t()));
				if (traj.addStep(start, end))
					break;
				start = end;
			}
			{
				curView = this->_getNearestView(dposex.R, dposex.t);
				float err = dfr.calcError(dposex, K, this->views[curView].contourPoints3d, alpha);
				if (err < errMin)
				{
					errMin = err;
					dpose = dposex;
				}
				if (errMin < errT)
					break;
			}
		}
	}

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	pose = dpose;

	auto RR0 = this->views[curView].R;
	float errR = get_errorR(pose.R, gtpose.R);
	float errR0 = get_errorR(pose.R, RR0);

	EdgeResponse edger(img);
	for (int itr = 0; itr < outerItrs; itr++)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!edger.edgeupdate(dpose, K, this->views[curView].contourPoints3d, innerItrs, eps))
			break;
		RR0 = this->views[curView].R;
		errR = get_errorR(dpose.R, gtpose.R);
		errR0 = get_errorR(dpose.R, RR0);
	}
	curView = this->_getNearestView(dpose.R, dpose.t);
	pose = dpose;

	errR = get_errorR(pose.R, gtpose.R);
	//errR0 = get_errorR(pose.R, RR0);

	return errMin;
}


//边缘响应，带内部点

struct CPoint
{
	Point3f    center;
	Point3f    normal;
	float      edgew;

	//DEFINE_BFS_IO_2(CPoint, center, normal)
	DEFINE_BFS_IO_3(CPoint, center, normal, edgew)
};



class EdgeSampler
{
public:
	static void _sample(Rect roiRect, const Mat1f& depth, CVRProjector& prj, std::vector<CPoint>& c3d, int nSamples, const Mat1s& dx, const Mat1s& dy)
	{
		Mat1b depthMask = getRenderMask(depth);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(depthMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		const int dstride = stepC(depth);
		auto get3D = [&depthMask, &depth, dstride, &prj, roiRect](int x, int y, Point3f& P, float& _z) {
			float z = 0;
			if (depthMask(y, x) != 0)
				z = depth(y, x);
			else
			{
				CV_Assert(false);
			}
			P = prj.unproject(float(x + roiRect.x), float(y + roiRect.y), z);
			_z = z;

			return true;
		};

		size_t npt = 0;
		for (auto& c : contours)
			npt += c.size();
		int nSkip = (int)npt / nSamples;

		const int smoothWSZ = 15, hwsz = smoothWSZ / 2;
		float maxDiff = 1.f;
		//cv::Mat visualization = cv::Mat::zeros(depth.size(), CV_8UC1);
		for (auto& c : contours)
		{
			double area = cv::contourArea(c, true);
			if (area < 0) //if is clockwise
				std::reverse(c.begin(), c.end()); //make it counter-clockwise

			Mat2i  cimg(1, c.size(), (Vec2i*)&c[0], int(c.size()) * sizeof(Vec2f));
			Mat2f  smoothed;
			boxFilter(cimg, smoothed, CV_32F, Size(smoothWSZ, 1));
			const Point2f* smoothedPts = smoothed.ptr<Point2f>();

			for (int i = nSkip / 2; i < (int)c.size(); i += nSkip)
			{
				CPoint P;
				float depth;
				if (get3D(c[i].x, c[i].y, P.center, depth))
				{
					Point2f n(0, 0);
					if (i - hwsz >= 0)
						n += smoothedPts[i] - smoothedPts[i - hwsz];
					if (i + hwsz < smoothed.cols)
						n += smoothedPts[i + hwsz] - smoothedPts[i];
					n = normalize(Vec2f(n));
					n = Point2f(-n.y, n.x);
					Point2f q = Point2f(c[i]) + n + Point2f(roiRect.x, roiRect.y);
					P.normal = prj.unproject(q.x, q.y, depth);
					P.edgew = fabs(dx(c[i].y, c[i].x)) + fabs(dy(c[i].y, c[i].x));
					c3d.push_back(P);
					//cv::line(visualization, c[i], Point2f(c[i]) + 100*n + Point2f(roiRect.x, roiRect.y), cv::Scalar(255, 255, 255));
					//cv::circle(visualization, c[i], 5, cv::Scalar(255, 255, 0), -1);

				}
			}
		}
		//imwrite("D:\\RBOT_dataset\\can\\c.png", visualization);
		float wsum = 0;
		for (auto& cd : c3d) {
			wsum += cd.edgew;
		}
		wsum /= c3d.size();
		for (auto& cd : c3d) {
			cd.edgew /= wsum;
		}
	}

	static void sample(std::vector<CPoint>& c3d, CVRResult& rr, int nSamples, Rect roiRect = Rect(0, 0, 0, 0), Size imgSize = Size(0, 0))
	{
		if (imgSize.width == 0 || imgSize.height == 0)
			imgSize = rr.img.size();

		if (roiRect.width == 0 || roiRect.height == 0)
			roiRect = Rect(0, 0, imgSize.width, imgSize.height);

		CVRProjector prj(rr.mats, imgSize);

		Mat rgray = cv::convertBGRChannels(rr.img, 1);
		Mat1s dx, dy;
		cv::Sobel(rgray, dx, CV_16S, 1, 0);
		cv::Sobel(rgray, dy, CV_16S, 0, 1);

		Mat1b fgMask = rr.getMaskFromDepth();

		_sample(roiRect, rr.depth, prj, c3d, nSamples, dx, dy);
	}
};


struct IPoint
{
	Point3f    center;
	float      edgew;

	DEFINE_BFS_IO_2(IPoint, center, edgew)
};



class InnerSampler
{
public:
	static void _sample(Rect roiRect, const Mat1f& depth, CVRProjector& prj, std::vector<IPoint>& c3d, int nSamples, const Mat1b fgMask, const Mat1s& dx, const Mat1s& dy)
	{
		Mat1b depthMask = getRenderMask(depth);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(depthMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		const int dstride = stepC(depth);
		auto get3D = [&depthMask, &depth, dstride, &prj, roiRect](int x, int y, Point3f& P, float& _z) {
			float z = 0;
			if (depthMask(y, x) != 0)
				z = depth(y, x);
			else
			{
				return false;
			}
			P = prj.unproject(float(x + roiRect.x), float(y + roiRect.y), z);
			_z = z;

			return true;
		};

		size_t npt = 0;
		for (auto& c : contours)
			npt += c.size();
		int nSkip = (int)npt / nSamples;

		const int smoothWSZ = 15, hwsz = smoothWSZ / 2;
		float maxDiff = 1.f;

		for (auto& c : contours)
		{
			double area = cv::contourArea(c, true);
			if (area < 0) //if is clockwise
				std::reverse(c.begin(), c.end()); //make it counter-clockwise

			Mat2i  cimg(1, c.size(), (Vec2i*)&c[0], int(c.size()) * sizeof(Vec2f));
			Mat2f  smoothed;
			boxFilter(cimg, smoothed, CV_32F, Size(smoothWSZ, 1));
			const Point2f* smoothedPts = smoothed.ptr<Point2f>();
			std::set<std::pair<int, int>> sampledPoints;//存储
			for (int i = nSkip / 2; i < (int)c.size(); i += nSkip)
			{
				IPoint P;
				float depth;
				if (get3D(c[i].x, c[i].y, P.center, depth))
				{
					P.edgew = fabs(dx(c[i].y, c[i].x)) + fabs(dy(c[i].y, c[i].x));
					c3d.push_back(P);
					//采样到轮廓点后，沿着法向量方向向内部搜索点
					Point2f n(0, 0);
					if (i - hwsz >= 0)
						n += smoothedPts[i] - smoothedPts[i - hwsz];
					if (i + hwsz < smoothed.cols)
						n += smoothedPts[i + hwsz] - smoothedPts[i];
					n = normalize(Vec2f(n));
					n = Point2f(-n.y, n.x);

					int step = 100; int nxt = 50;
					while (1)
					{
						int nx = c[i].x + step * n.x;
						int ny = c[i].y + step * n.y;
						if (nx < 0 || nx >= depthMask.cols ||
							ny < 0 || ny >= depthMask.rows ||
							fgMask(ny, nx) & 255 != 255) {
							break;
						}
						if (sampledPoints.count({ nx, ny }) > 0) {
							step += nxt;
							continue;
						}
						// 添加采样点
						step += nxt;
						if (get3D(nx, ny, P.center, depth))
						{
							P.edgew = fabs(dx(ny, nx)) + fabs(dy(ny, nx));
							sampledPoints.insert({ nx, ny }); // 标记为已采样
							if (P.edgew == 0)continue;
							c3d.push_back(P);
							//cnt++;
						}
					}
				}
			}
		}
		float wsum = 0;
		for (auto& cd : c3d) {
			wsum += cd.edgew;
		}
		wsum /= c3d.size();
		for (auto& cd : c3d) {
			cd.edgew /= wsum;
		}
	}

	static void sample(std::vector<IPoint>& c3d, CVRResult& rr, int nSamples, Rect roiRect = Rect(0, 0, 0, 0), Size imgSize = Size(0, 0))
	{
		if (imgSize.width == 0 || imgSize.height == 0)
			imgSize = rr.img.size();

		if (roiRect.width == 0 || roiRect.height == 0)
			roiRect = Rect(0, 0, imgSize.width, imgSize.height);

		CVRProjector prj(rr.mats, imgSize);

		Mat rgray = cv::convertBGRChannels(rr.img, 1);
		Mat1s dx, dy;
		cv::Sobel(rgray, dx, CV_16S, 1, 0);
		cv::Sobel(rgray, dy, CV_16S, 0, 1);

		Mat1b fgMask = rr.getMaskFromDepth();

		_sample(roiRect, rr.depth, prj, c3d, nSamples, fgMask, dx, dy);
	}
};

struct DView
{
	Vec3f       viewDir;
	cv::Matx33f R;

	std::vector<CPoint>  contourPoints3d;
	std::vector<IPoint>  innerPoints3d;

	//DEFINE_BFS_IO_3(DView, viewDir, R, contourPoints3d)
	DEFINE_BFS_IO_4(DView, viewDir, R, contourPoints3d, innerPoints3d)
};


struct EdgeResponse
{

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
	Mat1f gradf;
public:
	EdgeResponse(Mat img)
	{
		gradf = getImageField(img);
	}
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		Mat1f f = gradf;

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 500;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, int maxItrs, float eps)
	{
		float mu = 0.f;
		float v = 2.f;
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_edgeupdate(pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		return true;
	}
};


struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;
	enum { N_LAYERS = 2 };
	Mat2f         _grad;
	Mat img0;
	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void getProjectedContours(const Matx33f& K, const Pose& pose, std::vector<Point2f>& points, std::vector<Point2f>* normals)
	{
		int curView = this->_getNearestView(pose.R, pose.t);

		Projector prj(K, pose.R, pose.t);
		points = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		if (normals)
		{
			*normals = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.normal; });
			for (int i = 0; i < (int)points.size(); ++i)
			{
				auto n = (*normals)[i] - points[i];
				(*normals)[i] = normalize(Vec2f(n));
			}
		}
	}

public:
	void build(CVRModel& model);
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT, fi, gtpose);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

void projimg(const Matx33f K, Pose pose, vector<CPoint>cpoints, Mat img, int step = 1)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;
	int npt = cpoints.size();
	const float fx = K(0, 0), fy = K(1, 1);

	for (int i = 0; i < cpoints.size(); i += step)
	{
		//	continue;
		const Point3f Q = R * cpoints[i].center + t;
		const Point3f q = K * Q;

		const float X = Q.x, Y = Q.y, Z = Q.z;

		const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

		Point2f pt(q.x / q.z, q.y / q.z);

		//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
		if (uint(int(pt.x + 0.5)) >= uint(img.cols) || uint(int(pt.y + 0.5)) >= uint(img.rows))
			continue;
		circle(img, pt, 1, Scalar(255, 0, 255), -1);
	}
	imshow("img", img);
	//waitKey(0);
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
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

	/******************************************************************************/

	//std::ofstream file("pose_matrix2.txt", std::ios::app); // std::ios::app 用于追加到文件末尾

	//// 设置输出格式
	//file << std::fixed << std::setprecision(2);

	//// 写入矩阵
	//file << "init matrix:" << endl;
	//file << "R:" << endl;
	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		file << std::setw(6) << dpose.R(i, j) << " ";
	//	}
	//	file << std::endl;
	//}
	//file << "t:" << endl;
	//for (int i = 0; i < 3; i++) {
	//	file << std::setw(6) << dpose.t[i] << " ";
	//}
	//file << std::endl;

	//// 关闭文件
	//file.close();

	/****************************************************************************************/


	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);
	if (errMin > errT)
	{
		const auto R0 = dpose.R;

		const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;

		const float dc = thetaT / N;

		const int subDiv = 3;
		const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		RegionTrajectory traj(Size(subRegionSize, subRegionSize), dc / subDiv);

		Mat1b label = Mat1b::zeros(2 * N + 1, 2 * N + 1);
		struct DSeed
		{
			Point coord;
			//bool  isLocalMinima;
		};
		std::deque<DSeed>  seeds;
		seeds.push_back({ Point(N,N)/*,true*/ });
		label(N, N) = 1;


		auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			{
				if (label(y, x) == 0)
				{
					label(y, x) = 1;
					seeds.push_back({ Point(x, y)/*, false*/ });
				}
			}
		};

		while (!seeds.empty())
		{
			auto curSeed = seeds.front();
			seeds.pop_front();

			checkAdd(curSeed, 0, -1);
			checkAdd(curSeed, -1, 0);
			checkAdd(curSeed, 1, 0);
			checkAdd(curSeed, 0, 1);

			auto dR = theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

			auto dposex = dpose;
			dposex.R = dR * R0;

			Point2f start = dir2Theta(viewDirFromR(dposex.R * R0.t()));
			for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			{
				if (itr % innerItrs == 0)
					curView = this->_getNearestView(dposex.R, dposex.t);

				if (!dfr.update(dposex, K, this->views[curView].contourPoints3d, 1, alphaNonLocal, eps))
					break;

				Point2f end = dir2Theta(viewDirFromR(dposex.R * R0.t()));
				if (traj.addStep(start, end))
					break;
				start = end;
			}
			{
				curView = this->_getNearestView(dposex.R, dposex.t);
				float err = dfr.calcError(dposex, K, this->views[curView].contourPoints3d, alpha);
				if (err < errMin)
				{
					errMin = err;
					dpose = dposex;
				}
				if (errMin < errT)
					break;
			}
		}
	}

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	pose = dpose;

	EdgeResponse edger(img);
	for (int itr = 0; itr < outerItrs; itr++)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!edger.edgeupdate(dpose, K, this->views[curView].innerPoints3d, innerItrs, eps))
			break;
	}

	pose = dpose;
	curView = this->_getNearestView(dpose.R, dpose.t);
	projimg(K, dpose, this->views[curView].contourPoints3d, img.clone(), 2);

	return errMin;
}

inline void Templates::build(CVRModel& model)
{
	std::vector<Vec3f>  viewDirs;
	cvrm::sampleSphere(viewDirs, 3000);

	auto center = model.getCenter();
	auto sizeBB = model.getSizeBB();
	float maxBBSize = __max(sizeBB[0], __max(sizeBB[1], sizeBB[2]));
	float eyeDist = 0.8f;
	float fscale = 2.5f;
	Size  viewSize(2000, 2000);

	this->modelCenter = center;

	std::vector<DView> dviews;
	dviews.reserve(viewDirs.size());

	CVRender render(model);

	auto vertices = model.getVertices();

	int vi = 1;
	for (auto& viewDir : viewDirs)
	{
		printf("build templates %d/%d    \r", vi++, (int)viewDirs.size());
		{
			auto eyePos = center + viewDir * eyeDist;

			CVRMats mats;
			mats.mModel = cvrm::lookat(eyePos[0], eyePos[1], eyePos[2], center[0], center[1], center[2], 0.1f, 1.1f, 0.1f);
			mats.mProjection = cvrm::perspective(viewSize.height * fscale, viewSize, __max(0.01, eyeDist - maxBBSize), eyeDist + maxBBSize);

			auto rr = render.exec(mats, viewSize);
			Mat1b fgMask = getRenderMask(rr.depth);

			{
				auto roi = get_mask_roi(DWHS(fgMask), 127);
				int bwx = __min(roi.x, fgMask.cols - roi.x - roi.width);
				int bwy = __min(roi.y, fgMask.rows - roi.y - roi.height);
				if (__min(bwx, bwy) < 5/* || __max(roi.width,roi.height)<fgMask.cols/4*/)
				{
					imshow("mask", fgMask);
					cv::waitKey();
				}
			}

			dviews.push_back(DView());
			DView& dv = dviews.back();

			dv.viewDir = viewDir;

			Vec3f t;
			cvrm::decomposeRT(mats.mModel, dv.R, t);

			EdgeSampler::sample(dv.contourPoints3d, rr, 200);
			InnerSampler::sample(dv.innerPoints3d, rr, 200);
		}
	}
	this->views.swap(dviews);
	this->viewIndex.build(this->views);
}


//金字塔初版
struct EdgeResponse
{
	Mat originimg;
	int pylayers;
	vector<Mat1f> gauss_pyramid;
	Mat1f gradf;
	int downscale;

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
public:
	EdgeResponse() {}
	EdgeResponse(Mat img) {

		originimg = img;
		gradf = getImageField(originimg);
		int height = gradf.rows;
		int width = gradf.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(gradf);
		downscale = 2;

		// 循环进行下采样
		for (int i = 0; i < pylayers; i++) {
			// 创建一个临时变量存储下采样后的图像
			Mat1f temp(height / downscale, width / downscale);
			// 调用pyrDown函数进行下采样
			pyrDown(gauss_pyramid[i], temp);
			// 将下采样后的图像添加到向量中
			gauss_pyramid.push_back(temp);
			downscale *= 2;
		}
	}
	int _edgeupdate(int layer, Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		Mat1f f = gauss_pyramid[layer];

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 500;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f K, const std::vector<IPoint>& cpoints, float eps)
	{
		float mu = 0.f;
		float v = 2.f;
		Matx33f nK = K;
		downscale /= 2;
		//for (int itr = pylayers; itr >= 0 ; --itr) {
		//	nK(0, 0) = K(0, 0) / downscale; // f_x
		//	nK(1, 1) = K(1, 1) / downscale; // f_y
		//	nK(0, 2) = K(0, 2) / downscale; // c_x
		//	nK(1, 2) = K(1, 2) / downscale; // c_y
		//	if (this->_edgeupdate(itr, pose, nK, cpoints, eps, mu, v, itr == 0) <= 0)
		//		return false;
		//	downscale /= 2;
		//}
		int ds = 1;
		for (int itr = 0; itr <= pylayers; ++itr) {
			nK(0, 0) = K(0, 0) / ds; // f_x
			nK(1, 1) = K(1, 1) / ds; // f_y
			nK(0, 2) = K(0, 2) / ds; // c_x
			nK(1, 2) = K(1, 2) / ds; // c_y
			if (this->_edgeupdate(itr, pose, nK, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
			ds *= 2;
		}

		return true;
	}
};

//金字塔v2
struct EdgeResponse
{
	Mat originimg;
	int pylayers;
	vector<Mat1f> gauss_pyramid;
	vector<Mat1f> laplace_pyramid;
	Mat1f gradf;
	int downscale;

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
public:
	EdgeResponse() {}
	EdgeResponse(Mat img) {

		originimg = img;
		gradf = getImageField(originimg);
		int height = gradf.rows;
		int width = gradf.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(gradf);

		// 循环进行下采样
		for (int i = 0; i < pylayers; i++) {

			Mat1f temp(height / 2, width / 2);
			Mat1f tempUp(height, width);
			Mat1f laplace;
			pyrDown(gauss_pyramid[i], temp);
			pyrUp(temp, tempUp);
			gauss_pyramid.push_back(tempUp);
			laplace = gauss_pyramid[i] - tempUp;
			laplace_pyramid.push_back(laplace);
		}
		laplace_pyramid.push_back(gauss_pyramid[pylayers]);
	}
	int _edgeupdate(int layer, Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		Mat1f f = gauss_pyramid[layer];
		//Mat1f f = laplace_pyramid[layer];

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 600;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f K, const std::vector<IPoint>& cpoints, float eps)
	{
		float mu = 0.f;
		float v = 2.f;

		for (int itr = pylayers; itr >= 0; --itr) {
			if (this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		}

		return true;
	}
};

//金字塔v3 拉普拉斯
struct EdgeResponse
{
	Mat originimg;
	int pylayers;
	vector<Mat1f> gauss_pyramid;
	vector<Mat1f> laplace_pyramid;
	Mat1f gradf;
	int fi;

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
public:
	EdgeResponse() {}
	EdgeResponse(Mat img) {

		originimg = img;
		gradf = getImageField(originimg);
		int height = gradf.rows;
		int width = gradf.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(gradf);

		// 循环进行下采样
		for (int i = 0; i < pylayers; i++) {

			Mat1f temp(height / 2, width / 2);
			Mat1f tempUp(height, width);
			Mat1f laplace;
			pyrDown(gauss_pyramid[i], temp);
			pyrUp(temp, tempUp);
			gauss_pyramid.push_back(tempUp);
			laplace = gauss_pyramid[i] - tempUp;
			laplace_pyramid.push_back(laplace);
		}
		laplace_pyramid.push_back(gauss_pyramid[pylayers]);
	}
	EdgeResponse(Mat img, int fi) {
		this->fi = fi;
		originimg = img;
		gradf = getImageField(originimg);
		int height = gradf.rows;
		int width = gradf.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(gradf);

		// 循环进行下采样
		for (int i = 0; i < pylayers; i++) {

			Mat1f temp(height / 2, width / 2);
			Mat1f tempUp(height, width);
			Mat1f laplace;
			pyrDown(gauss_pyramid[i], temp);
			pyrUp(temp, tempUp);
			gauss_pyramid.push_back(tempUp);
			laplace = gauss_pyramid[i] - tempUp;
			laplace_pyramid.push_back(laplace);
		}
		//laplace_pyramid.push_back(gauss_pyramid[pylayers]);

	}
	int _edgeupdate(int layer, Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		//Mat1f f = gauss_pyramid[layer];
		Mat1f f = laplace_pyramid[layer];
		//cv::Mat normalizedF;
		//cv::normalize(f, normalizedF, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//cv::imshow("Display" + to_string(layer), normalizedF);
		//cv::waitKey(0); // Wait for a keystroke in the window
		//imwrite("D:\\RBOT_dataset\\driller\\py\\" + to_string(fi) + "_" + to_string(layer) + ".png", normalizedF);
		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 2000;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f K, const std::vector<IPoint>& cpoints, float eps)
	{
		float mu = 0.f;
		float v = 2.f;

		for (int itr = pylayers - 3; itr >= 0; --itr) {
			if (this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		}
		/*int convergent = 0;
		for (int i = 0; i < 10; i++) {
			for (int itr = pylayers; itr >= 0; --itr) {
				convergent = this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0);
			}
			if (convergent <= 0)return false;
		}*/


		return true;
	}
};

//金字塔 高斯
struct EdgeResponse
{
	Mat originimg;
	int pylayers;
	vector<Mat> gauss_pyramid;
	vector<Mat1f> gauss_edge;
	vector<Mat1f> laplace_pyramid;
	Mat1f gradf;
	int fi;

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
public:
	EdgeResponse() {}
	EdgeResponse(Mat img) {

		originimg = img;
		int height = originimg.rows;
		int width = originimg.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(originimg);
		gradf = getImageField(originimg);
		gauss_edge.push_back(gradf);

		for (int i = 0; i < pylayers; i++)
		{
			Mat gaussDown;
			Mat1f edgeDown(height / 2, width / 2);
			Mat gaussUp;
			Mat1f edgeUp(height, width);
			Mat1f laplace;

			pyrDown(gauss_pyramid[i], gaussDown);
			//edgeDown = getImageField(gaussDown);
			pyrUp(gaussDown, gaussUp);
			//pyrUp(edgeDown, edgeUp);
			edgeDown = getImageField(gaussDown);

			laplace = gauss_edge[i] - edgeUp;

			gauss_pyramid.push_back(gaussUp);
			gauss_edge.push_back(edgeUp);
			laplace_pyramid.push_back(laplace);

		}
	}
	EdgeResponse(Mat img, int fi) {
		this->fi = fi;
		originimg = img;
		int height = originimg.rows;
		int width = originimg.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(originimg);
		gradf = getImageField(originimg);
		gauss_edge.push_back(gradf);

		for (int i = 0; i < pylayers; i++)
		{
			Mat gaussDown;
			Mat1f edgeDown(height / 2, width / 2);
			Mat gaussUp;
			Mat1f edgeUp(height, width);
			Mat1f laplace;

			pyrDown(gauss_pyramid[i], gaussDown);
			//edgeDown = getImageField(gaussDown);
			pyrUp(gaussDown, gaussUp);
			//pyrUp(edgeDown, edgeUp);
			edgeDown = getImageField(gaussDown);

			laplace = gauss_edge[i] - edgeUp;

			gauss_pyramid.push_back(gaussUp);
			gauss_edge.push_back(edgeUp);
			laplace_pyramid.push_back(laplace);

		}

	}
	int _edgeupdate(int layer, Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		//Mat1f f = gauss_pyramid[layer];
		//Mat1f f = laplace_pyramid[layer];
		Mat1f f = gauss_edge[layer];
		//cv::Mat normalizedF;
		//cv::normalize(f, normalizedF, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//cv::imshow("Display" + to_string(layer), normalizedF);
		//cv::waitKey(0); // Wait for a keystroke in the window
		//imwrite("D:\\RBOT_dataset\\driller\\py\\" + to_string(fi) + "_" + to_string(layer) + ".png", normalizedF);
		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 10000;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f K, const std::vector<IPoint>& cpoints, float eps)
	{
		float mu = 0.f;
		float v = 2.f;

		for (int itr = pylayers; itr >= 0; --itr) {
			if (this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		}
		/*int convergent = 0;
		for (int i = 0; i < 10; i++) {
			for (int itr = pylayers; itr >= 0; --itr) {
				convergent = this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0);
			}
			if (convergent <= 0)return false;
		}*/


		return true;
	}
};

//金字塔 final
struct EdgeResponse
{
	Mat originimg;
	int pylayers;
	vector<Mat> gauss_pyramid;
	vector<Mat1f> gauss_edge;
	vector<Mat1f> laplace_pyramid;
	Mat1f gradf;
	int fi;

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
public:
	EdgeResponse() {}
	EdgeResponse(Mat img) {

		originimg = img;
		int height = originimg.rows;
		int width = originimg.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(originimg);
		gradf = getImageField(originimg);
		gauss_edge.push_back(gradf);

		for (int i = 0; i < pylayers; i++)
		{
			Mat gaussDown;
			Mat1f edgeDown(height / 2, width / 2);
			Mat gaussUp;
			Mat1f edgeUp(height, width);
			Mat1f laplace;

			pyrDown(gauss_pyramid[i], gaussDown);
			//edgeDown = getImageField(gaussDown);
			pyrUp(gaussDown, gaussUp);
			//pyrUp(edgeDown, edgeUp);
			edgeDown = getImageField(gaussDown);

			laplace = gauss_edge[i] - edgeUp;

			gauss_pyramid.push_back(gaussUp);
			gauss_edge.push_back(edgeUp);
			laplace_pyramid.push_back(laplace);

		}
	}
	EdgeResponse(Mat img, int fi) {
		this->fi = fi;
		originimg = img;
		int height = originimg.rows;
		int width = originimg.cols;
		// pylayers=log2(min(w,h)/32),向上取整
		pylayers = ceil(log2(min(height, width) / 32));
		gauss_pyramid.push_back(originimg);
		gradf = getImageField(originimg);
		gauss_edge.push_back(gradf);

		for (int i = 0; i < pylayers; i++)
		{
			Mat gaussDown;
			Mat1f edgeDown(height / 2, width / 2);
			Mat gaussUp;
			Mat1f edgeUp(height, width);
			Mat1f laplace;

			pyrDown(gauss_pyramid[i], gaussDown);
			//edgeDown = getImageField(gaussDown);
			pyrUp(gaussDown, gaussUp);
			//pyrUp(edgeDown, edgeUp);
			edgeDown = getImageField(gaussDown);

			laplace = gauss_edge[i] - edgeUp;

			gauss_pyramid.push_back(gaussUp);
			gauss_edge.push_back(edgeUp);
			laplace_pyramid.push_back(laplace);

		}

	}
	int _edgeupdate(int layer, Optimizer::PoseData& pose, const Matx33f& K, const std::vector<IPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		//Mat1f f = gauss_pyramid[layer];
		//Mat1f f = laplace_pyramid[layer];
		Mat1f f = gauss_edge[layer];
		//cv::Mat normalizedF;
		//cv::normalize(f, normalizedF, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//cv::imshow("Display" + to_string(layer), normalizedF);
		//cv::waitKey(0); // Wait for a keystroke in the window
		//imwrite("D:\\RBOT_dataset\\driller\\py\\" + to_string(fi) + "_" + to_string(layer) + ".png", normalizedF);
		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 1100;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f K, const std::vector<IPoint>& cpoints, float eps)
	{
		float mu = 0.f;
		float v = 2.f;

		for (int itr = pylayers; itr >= 0; --itr) {
			if (this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		}
		/*int convergent = 0;
		for (int i = 0; i < 10; i++) {
			for (int itr = pylayers; itr >= 0; --itr) {
				convergent = this->_edgeupdate(itr, pose, K, cpoints, eps, mu, v, itr == 0);
			}
			if (convergent <= 0)return false;
		}*/


		return true;
	}
};


//实时渲染   加内部点

struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;
	CVRModel              _model;
	enum { N_LAYERS = 2 };
	Mat2f         _grad;
	Mat img0;
	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void getProjectedContours(const Matx33f& K, const Pose& pose, std::vector<Point2f>& points, std::vector<Point2f>* normals)
	{
		int curView = this->_getNearestView(pose.R, pose.t);

		Projector prj(K, pose.R, pose.t);
		points = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		if (normals)
		{
			*normals = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.normal; });
			for (int i = 0; i < (int)points.size(); ++i)
			{
				auto n = (*normals)[i] - points[i];
				(*normals)[i] = normalize(Vec2f(n));
			}
		}
	}

public:
	void build(CVRModel& model);
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}
	void realrender(Pose pose, vector<IPoint>& innerPoints)
	{
		float eyeDist = 0.8f;
		float fscale = 2.5f;
		Size  viewSize(2000, 2000);
		auto center = _model.getCenter();
		auto sizeBB = _model.getSizeBB();
		float maxBBSize = __max(sizeBB[0], __max(sizeBB[1], sizeBB[2]));
		CVRender render(_model);

		Vec3f viewDir = _getViewDir(pose.R, pose.t);
		auto eyePos = center + viewDir * eyeDist;

		CVRMats mats;
		mats.mModel = cvrm::lookat(eyePos[0], eyePos[1], eyePos[2], center[0], center[1], center[2], 0.1f, 1.1f, 0.1f);
		mats.mProjection = cvrm::perspective(viewSize.height * fscale, viewSize, __max(0.01, eyeDist - maxBBSize), eyeDist + maxBBSize);

		auto rr = render.exec(mats, viewSize);

		InnerSampler::sample(innerPoints, rr, 200);
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT, fi, gtpose);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

void projimg(const Matx33f K, Pose pose, vector<CPoint>cpoints, Mat img, int step = 1)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;
	int npt = cpoints.size();
	const float fx = K(0, 0), fy = K(1, 1);

	for (int i = 0; i < cpoints.size(); i += step)
	{
		//	continue;
		const Point3f Q = R * cpoints[i].center + t;
		const Point3f q = K * Q;

		const float X = Q.x, Y = Q.y, Z = Q.z;

		const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

		Point2f pt(q.x / q.z, q.y / q.z);

		//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
		if (uint(int(pt.x + 0.5)) >= uint(img.cols) || uint(int(pt.y + 0.5)) >= uint(img.rows))
			continue;
		circle(img, pt, 1, Scalar(255, 0, 255), -1);
	}
	imshow("img", img);
	//waitKey(0);
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
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

	/******************************************************************************/

	//std::ofstream file("pose_matrix2.txt", std::ios::app); // std::ios::app 用于追加到文件末尾

	//// 设置输出格式
	//file << std::fixed << std::setprecision(2);

	//// 写入矩阵
	//file << "init matrix:" << endl;
	//file << "R:" << endl;
	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		file << std::setw(6) << dpose.R(i, j) << " ";
	//	}
	//	file << std::endl;
	//}
	//file << "t:" << endl;
	//for (int i = 0; i < 3; i++) {
	//	file << std::setw(6) << dpose.t[i] << " ";
	//}
	//file << std::endl;

	//// 关闭文件
	//file.close();

	/****************************************************************************************/


	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);
	if (errMin > errT)
	{
		const auto R0 = dpose.R;

		const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;

		const float dc = thetaT / N;

		const int subDiv = 3;
		const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		RegionTrajectory traj(Size(subRegionSize, subRegionSize), dc / subDiv);

		Mat1b label = Mat1b::zeros(2 * N + 1, 2 * N + 1);
		struct DSeed
		{
			Point coord;
			//bool  isLocalMinima;
		};
		std::deque<DSeed>  seeds;
		seeds.push_back({ Point(N,N)/*,true*/ });
		label(N, N) = 1;


		auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			{
				if (label(y, x) == 0)
				{
					label(y, x) = 1;
					seeds.push_back({ Point(x, y)/*, false*/ });
				}
			}
		};

		while (!seeds.empty())
		{
			auto curSeed = seeds.front();
			seeds.pop_front();

			checkAdd(curSeed, 0, -1);
			checkAdd(curSeed, -1, 0);
			checkAdd(curSeed, 1, 0);
			checkAdd(curSeed, 0, 1);

			auto dR = theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

			auto dposex = dpose;
			dposex.R = dR * R0;

			Point2f start = dir2Theta(viewDirFromR(dposex.R * R0.t()));
			for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			{
				if (itr % innerItrs == 0)
					curView = this->_getNearestView(dposex.R, dposex.t);

				if (!dfr.update(dposex, K, this->views[curView].contourPoints3d, 1, alphaNonLocal, eps))
					break;

				Point2f end = dir2Theta(viewDirFromR(dposex.R * R0.t()));
				if (traj.addStep(start, end))
					break;
				start = end;
			}
			{
				curView = this->_getNearestView(dposex.R, dposex.t);
				float err = dfr.calcError(dposex, K, this->views[curView].contourPoints3d, alpha);
				if (err < errMin)
				{
					errMin = err;
					dpose = dposex;
				}
				if (errMin < errT)
					break;
			}
		}
	}

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	pose = dpose;
	EdgeResponse edger(img);
	for (int itr = 0; itr < outerItrs; itr++)
	{

		//curView = this->_getNearestView(dpose.R, dpose.t);
		/*if (!edger.edgeupdate(dpose, K, this->views[curView].innerPoints3d, eps))
			break;*/
		vector<IPoint>innerPoints3d;
		realrender(dpose, innerPoints3d);
		if (!edger.edgeupdate(dpose, K, innerPoints3d, innerItrs, eps))
			break;
	}

	pose = dpose;
	/*curView = this->_getNearestView(dpose.R, dpose.t);
	projimg(K, dpose, this->views[curView].contourPoints3d, img.clone(), 2);*/

	return errMin;
}

inline void Templates::build(CVRModel& model)
{
	std::vector<Vec3f>  viewDirs;
	cvrm::sampleSphere(viewDirs, 3000);

	auto center = model.getCenter();
	auto sizeBB = model.getSizeBB();
	float maxBBSize = __max(sizeBB[0], __max(sizeBB[1], sizeBB[2]));
	float eyeDist = 0.8f;
	float fscale = 2.5f;
	Size  viewSize(2000, 2000);

	this->modelCenter = center;
	this->_model = model;

	std::vector<DView> dviews;
	dviews.reserve(viewDirs.size());

	CVRender render(model);

	auto vertices = model.getVertices();

	int vi = 1;
	for (auto& viewDir : viewDirs)
	{
		printf("build templates %d/%d    \r", vi++, (int)viewDirs.size());
		{
			auto eyePos = center + viewDir * eyeDist;

			CVRMats mats;
			mats.mModel = cvrm::lookat(eyePos[0], eyePos[1], eyePos[2], center[0], center[1], center[2], 0.1f, 1.1f, 0.1f);
			mats.mProjection = cvrm::perspective(viewSize.height * fscale, viewSize, __max(0.01, eyeDist - maxBBSize), eyeDist + maxBBSize);

			auto rr = render.exec(mats, viewSize);
			Mat1b fgMask = getRenderMask(rr.depth);

			{
				auto roi = get_mask_roi(DWHS(fgMask), 127);
				int bwx = __min(roi.x, fgMask.cols - roi.x - roi.width);
				int bwy = __min(roi.y, fgMask.rows - roi.y - roi.height);
				if (__min(bwx, bwy) < 5/* || __max(roi.width,roi.height)<fgMask.cols/4*/)
				{
					imshow("mask", fgMask);
					cv::waitKey();
				}
			}

			dviews.push_back(DView());
			DView& dv = dviews.back();

			dv.viewDir = viewDir;

			Vec3f t;
			cvrm::decomposeRT(mats.mModel, dv.R, t);
			EdgeSampler::sample(dv.contourPoints3d, rr, 200);
			//InnerSampler::sample(dv.innerPoints3d, rr, 200);
		}
	}
	this->views.swap(dviews);
	this->viewIndex.build(this->views);
}


//实时渲染  只有轮廓点
struct EdgeResponse
{

	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
	Mat1f gradf;
public:
	EdgeResponse(Mat img)
	{
		gradf = getImageField(img);
	}
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float eps, float mu, float v, bool isinit)
	{
		Mat1f f = gradf;

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		Matx33f R = pose.R;
		Point3f t = pose.t;
		int npt = cpoints.size();

		float E_x = 0.f;
		float E_deltax = 0.f;
		float f_x = 0.f;
		float f_deltx = 0.f;
		float E_diff = 0.f;
		float L_diff = 0.f;
		float ρ = 0.f;

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));

			E_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			f_x += w * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;*/

		if (isinit)
		{
			float tao = 500;
			for (int i = 0; i < 6; i++)
				mu = max(mu, tao * JJ(i, i));
		}
		for (int i = 0; i < 6; i++)
			JJ(i, i) += mu;

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

			for (size_t i = 0; i < cpoints.size(); ++i)
			{
				//	continue;
				const Point3f Q = R * cpoints[i].center + t;
				const Point3f q = K * Q;

				const float X = Q.x, Y = Q.y, Z = Q.z;

				const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

				Point2f pt(q.x / q.z, q.y / q.z);

				//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
				if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
					continue;

				//Vec2f nx(dfx(y, x), dfy(y, x));
				Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
				nx = normalize(nx);


				Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

				auto dt = n_dq_dQ.t() * R;
				auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

				Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


				const float w = cpoints[i].edgew;
				const float wf = (w * _resample(f, pt.x, pt.y));

				E_deltax += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
				f_deltx += w * _resample(f, pt.x, pt.y);
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

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float eps)
	{
		float mu = 0.f;
		float v = 2.f;
		for (int itr = 0; itr < maxItrs; ++itr)
			if (this->_edgeupdate(pose, K, cpoints, eps, mu, v, itr == 0) <= 0)
				return false;
		return true;
	}
};


struct Templates
{
	Point3f               modelCenter;
	std::vector<DView>   views;
	ViewIndex             viewIndex;
	CVRModel              _model;
	enum { N_LAYERS = 2 };
	Mat2f         _grad;
	Mat img0;
	DEFINE_BFS_IO_2(Templates, modelCenter, views)

public:
	void getProjectedContours(const Matx33f& K, const Pose& pose, std::vector<Point2f>& points, std::vector<Point2f>* normals)
	{
		int curView = this->_getNearestView(pose.R, pose.t);

		Projector prj(K, pose.R, pose.t);
		points = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.center; });
		if (normals)
		{
			*normals = prj(views[curView].contourPoints3d, [](const CPoint& p) {return p.normal; });
			for (int i = 0; i < (int)points.size(); ++i)
			{
				auto n = (*normals)[i] - points[i];
				(*normals)[i] = normalize(Vec2f(n));
			}
		}
	}

public:
	void build(CVRModel& model);
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, Mat1f gradf, float eps);
	void save(const std::string& file)
	{
		ff::OBFStream os(file);
		os << (*this);
	}
	void load(const std::string& file)
	{
		ff::IBFStream is(file);
		is >> (*this);
		viewIndex.build(this->views);
	}
	void showInfo()
	{
		int minSamplePoints = INT_MAX;
		for (auto& v : views)
			if (v.contourPoints3d.size() < minSamplePoints)
				minSamplePoints = v.contourPoints3d.size();
		printf("minSamplePoints=%d\n", minSamplePoints);
	}
public:
	Vec3f _getViewDir(const Matx33f& R, const Vec3f& t)
	{
		return normalize(-R.inv() * t - Vec3f(this->modelCenter));
	}
	int  _getNearestView(const Vec3f& viewDir)
	{
		CV_Assert(fabs(viewDir.dot(viewDir) - 1.f) < 1e-3f);

		return viewIndex.getViewInDir(viewDir);
	}
	int  _getNearestView(const Matx33f& R, const Vec3f& t)
	{
		return _getNearestView(this->_getViewDir(R, t));
	}
	void realrender(Pose pose, vector<CPoint>& innerPoints)
	{
		float eyeDist = 0.8f;
		float fscale = 2.5f;
		Size  viewSize(2000, 2000);
		auto center = _model.getCenter();
		auto sizeBB = _model.getSizeBB();
		float maxBBSize = __max(sizeBB[0], __max(sizeBB[1], sizeBB[2]));
		CVRender render(_model);

		Vec3f viewDir = _getViewDir(pose.R, pose.t);
		auto eyePos = center + viewDir * eyeDist;

		CVRMats mats;
		mats.mModel = cvrm::lookat(eyePos[0], eyePos[1], eyePos[2], center[0], center[1], center[2], 0.1f, 1.1f, 0.1f);
		mats.mProjection = cvrm::perspective(viewSize.height * fscale, viewSize, __max(0.01, eyeDist - maxBBSize), eyeDist + maxBBSize);

		auto rr = render.exec(mats, viewSize);
		Mat1b fgMask = getRenderMask(rr.depth);

		{
			auto roi = get_mask_roi(DWHS(fgMask), 127);
			int bwx = __min(roi.x, fgMask.cols - roi.x - roi.width);
			int bwy = __min(roi.y, fgMask.rows - roi.y - roi.height);
			if (__min(bwx, bwy) < 5/* || __max(roi.width,roi.height)<fgMask.cols/4*/)
			{
				imshow("mask", fgMask);
				cv::waitKey();
			}
		}
		EdgeSampler::sample(innerPoints, rr, 200);
	}

	float pro(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
	{
		return this->pro1(K, pose, curProb, img, thetaT, errT, fi, gtpose);
	}

	float pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose);
};


class RegionTrajectory
{
	Mat1b  _pathMask;
	float  _delta;

	Point2f _uv2Pt(const Point2f& uv)
	{
		return Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(Size regionSize, float delta)
	{
		_pathMask = Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(Point2f start, Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

void projimg(const Matx33f K, Pose pose, vector<CPoint>cpoints, Mat img, int step = 1)
{
	Matx33f R = pose.R;
	Point3f t = pose.t;
	int npt = cpoints.size();
	const float fx = K(0, 0), fy = K(1, 1);

	for (int i = 0; i < cpoints.size(); i += step)
	{
		//	continue;
		const Point3f Q = R * cpoints[i].center + t;
		const Point3f q = K * Q;

		const float X = Q.x, Y = Q.y, Z = Q.z;

		const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

		Point2f pt(q.x / q.z, q.y / q.z);

		//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
		if (uint(int(pt.x + 0.5)) >= uint(img.cols) || uint(int(pt.y + 0.5)) >= uint(img.rows))
			continue;
		circle(img, pt, 1, Scalar(255, 0, 255), -1);
	}
	imshow("img", img);
	//waitKey(0);
}

inline float Templates::pro1(const Matx33f& K, Pose& pose, const Mat1f& curProb, const Mat& img, float thetaT, float errT, int fi, Pose gtpose)
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

	/******************************************************************************/

	//std::ofstream file("pose_matrix2.txt", std::ios::app); // std::ios::app 用于追加到文件末尾

	//// 设置输出格式
	//file << std::fixed << std::setprecision(2);

	//// 写入矩阵
	//file << "init matrix:" << endl;
	//file << "R:" << endl;
	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		file << std::setw(6) << dpose.R(i, j) << " ";
	//	}
	//	file << std::endl;
	//}
	//file << "t:" << endl;
	//for (int i = 0; i < 3; i++) {
	//	file << std::setw(6) << dpose.t[i] << " ";
	//}
	//file << std::endl;

	//// 关闭文件
	//file.close();

	/****************************************************************************************/


	float errMin = dfr.calcError(dpose, K, this->views[curView].contourPoints3d, alpha);
	if (errMin > errT)
	{
		const auto R0 = dpose.R;

		const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;

		const float dc = thetaT / N;

		const int subDiv = 3;
		const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
		RegionTrajectory traj(Size(subRegionSize, subRegionSize), dc / subDiv);

		Mat1b label = Mat1b::zeros(2 * N + 1, 2 * N + 1);
		struct DSeed
		{
			Point coord;
			//bool  isLocalMinima;
		};
		std::deque<DSeed>  seeds;
		seeds.push_back({ Point(N,N)/*,true*/ });
		label(N, N) = 1;


		auto checkAdd = [&seeds, &label](const DSeed& curSeed, int dx, int dy) {
			int x = curSeed.coord.x + dx, y = curSeed.coord.y + dy;
			if (uint(x) < uint(label.cols) && uint(y) < uint(label.rows))
			{
				if (label(y, x) == 0)
				{
					label(y, x) = 1;
					seeds.push_back({ Point(x, y)/*, false*/ });
				}
			}
		};

		while (!seeds.empty())
		{
			auto curSeed = seeds.front();
			seeds.pop_front();

			checkAdd(curSeed, 0, -1);
			checkAdd(curSeed, -1, 0);
			checkAdd(curSeed, 1, 0);
			checkAdd(curSeed, 0, 1);

			auto dR = theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

			auto dposex = dpose;
			dposex.R = dR * R0;

			Point2f start = dir2Theta(viewDirFromR(dposex.R * R0.t()));
			for (int itr = 0; itr < outerItrs * innerItrs; ++itr)
			{
				if (itr % innerItrs == 0)
					curView = this->_getNearestView(dposex.R, dposex.t);

				if (!dfr.update(dposex, K, this->views[curView].contourPoints3d, 1, alphaNonLocal, eps))
					break;

				Point2f end = dir2Theta(viewDirFromR(dposex.R * R0.t()));
				if (traj.addStep(start, end))
					break;
				start = end;
			}
			{
				curView = this->_getNearestView(dposex.R, dposex.t);
				float err = dfr.calcError(dposex, K, this->views[curView].contourPoints3d, alpha);
				if (err < errMin)
				{
					errMin = err;
					dpose = dposex;
				}
				if (errMin < errT)
					break;
			}
		}
	}

	for (int itr = 0; itr < outerItrs; ++itr)
	{
		curView = this->_getNearestView(dpose.R, dpose.t);
		if (!dfr.update(dpose, K, this->views[curView].contourPoints3d, innerItrs, alpha, eps))
			break;
	}
	pose = dpose;
	EdgeResponse edger(img);
	for (int itr = 0; itr < outerItrs; itr++)
	{

		//curView = this->_getNearestView(dpose.R, dpose.t);
		/*if (!edger.edgeupdate(dpose, K, this->views[curView].innerPoints3d, eps))
			break;*/
		vector<CPoint>ContourPoints3d;
		realrender(dpose, ContourPoints3d);
		if (!edger.edgeupdate(dpose, K, ContourPoints3d, innerItrs, eps))
			break;
	}

	pose = dpose;
	/*curView = this->_getNearestView(dpose.R, dpose.t);
	projimg(K, dpose, this->views[curView].contourPoints3d, img.clone(), 2);*/

	return errMin;
}

inline void Templates::build(CVRModel& model)
{
	std::vector<Vec3f>  viewDirs;
	cvrm::sampleSphere(viewDirs, 3000);

	auto center = model.getCenter();
	auto sizeBB = model.getSizeBB();
	float maxBBSize = __max(sizeBB[0], __max(sizeBB[1], sizeBB[2]));
	float eyeDist = 0.8f;
	float fscale = 2.5f;
	Size  viewSize(2000, 2000);

	this->modelCenter = center;
	this->_model = model;

	std::vector<DView> dviews;
	dviews.reserve(viewDirs.size());

	CVRender render(model);

	auto vertices = model.getVertices();

	int vi = 1;
	for (auto& viewDir : viewDirs)
	{
		printf("build templates %d/%d    \r", vi++, (int)viewDirs.size());
		{
			auto eyePos = center + viewDir * eyeDist;

			CVRMats mats;
			mats.mModel = cvrm::lookat(eyePos[0], eyePos[1], eyePos[2], center[0], center[1], center[2], 0.1f, 1.1f, 0.1f);
			mats.mProjection = cvrm::perspective(viewSize.height * fscale, viewSize, __max(0.01, eyeDist - maxBBSize), eyeDist + maxBBSize);

			auto rr = render.exec(mats, viewSize);
			Mat1b fgMask = getRenderMask(rr.depth);

			{
				auto roi = get_mask_roi(DWHS(fgMask), 127);
				int bwx = __min(roi.x, fgMask.cols - roi.x - roi.width);
				int bwy = __min(roi.y, fgMask.rows - roi.y - roi.height);
				if (__min(bwx, bwy) < 5/* || __max(roi.width,roi.height)<fgMask.cols/4*/)
				{
					imshow("mask", fgMask);
					cv::waitKey();
				}
			}

			dviews.push_back(DView());
			DView& dv = dviews.back();

			dv.viewDir = viewDir;

			Vec3f t;
			cvrm::decomposeRT(mats.mModel, dv.R, t);
			EdgeSampler::sample(dv.contourPoints3d, rr, 200);
			//InnerSampler::sample(dv.innerPoints3d, rr, 200);
		}
	}
	this->views.swap(dviews);
	this->viewIndex.build(this->views);
}

//新采样法：
struct IPoint
{
	Point3f    center;
	float      edgew;

	DEFINE_BFS_IO_2(IPoint, center, edgew)
};



class InnerSampler
{
public:
	static void _sample(Rect roiRect, const Mat1f& depth, CVRProjector& prj, std::vector<IPoint>& c3d, int nSamples, const Mat1b fgMask, const Mat1s& dx, const Mat1s& dy)
	{
		Mat1b depthMask = getRenderMask(depth);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(depthMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		//cv::findContours(depthMask, contours, RETR_LIST, CHAIN_APPROX_NONE);

		const int dstride = stepC(depth);
		auto get3D = [&depthMask, &depth, dstride, &prj, roiRect](int x, int y, Point3f& P, float& _z) {
			float z = 0;
			if (depthMask(y, x) != 0)
				z = depth(y, x);
			else
			{
				return false;
			}
			P = prj.unproject(float(x + roiRect.x), float(y + roiRect.y), z);
			_z = z;

			return true;
		};

		size_t npt = 0;
		for (auto& c : contours)
			npt += c.size();
		int nSkip = (int)npt / nSamples;

		const int smoothWSZ = 15, hwsz = smoothWSZ / 2;
		float maxDiff = 1.f;
		int nedgePoint = 0;
		for_each_1(DWHN1(fgMask), [&nedgePoint](uchar m) {
			if (m != 0)
				++nedgePoint;
			});
		int nstep = __max(1, nedgePoint / nSamples);
		nedgePoint = 0;
		for_each_1c(DWHN1(fgMask), [&nedgePoint, nstep, &c3d, prj, &dx, &dy, &get3D](uchar m, int x, int y) {
			if (m != 0)
			{
				if (++nedgePoint % nstep == 0)
				{
					IPoint P;
					float depth;
					if (get3D(x, y, P.center, depth))
					{
						P.edgew = fabs(dx(y, x)) + fabs(dy(y, x));
						if (P.edgew != 0)
							c3d.push_back(P);
					}
				}
			}});

		float wsum = 0;
		for (auto& cd : c3d) {
			wsum += cd.edgew;
		}
		wsum /= c3d.size();
		for (auto& cd : c3d) {
			cd.edgew /= wsum;
		}
	}

	static void sample(std::vector<IPoint>& c3d, CVRResult& rr, int nSamples, Rect roiRect = Rect(0, 0, 0, 0), Size imgSize = Size(0, 0))
	{
		if (imgSize.width == 0 || imgSize.height == 0)
			imgSize = rr.img.size();

		if (roiRect.width == 0 || roiRect.height == 0)
			roiRect = Rect(0, 0, imgSize.width, imgSize.height);

		CVRProjector prj(rr.mats, imgSize);

		Mat rgray = cv::convertBGRChannels(rr.img, 1);
		Mat1s dx, dy;
		cv::Sobel(rgray, dx, CV_16S, 1, 0);
		cv::Sobel(rgray, dy, CV_16S, 0, 1);

		Mat1b fgMask = rr.getMaskFromDepth();

		_sample(roiRect, rr.depth, prj, c3d, nSamples, fgMask, dx, dy);
	}
};

//全图搜索
class InnerSampler
{
public:
	static void _sample(Rect roiRect, const Mat1f& depth, CVRProjector& prj, std::vector<IPoint>& c3d, int nSamples, const Mat1b fgMask, const Mat1s& dx, const Mat1s& dy)
	{
		Mat1b depthMask = getRenderMask(depth);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(depthMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		const int dstride = stepC(depth);
		auto get3D = [&depthMask, &depth, dstride, &prj, roiRect](int x, int y, Point3f& P, float& _z) {
			float z = 0;
			if (depthMask(y, x) != 0)
				z = depth(y, x);
			else
			{
				return false;
			}
			P = prj.unproject(float(x + roiRect.x), float(y + roiRect.y), z);
			_z = z;

			return true;
		};
		size_t npt = 0;
		cv::Rect roiUnion;
		for (size_t i = 0; i < contours.size(); i++) {
			cv::Rect rect = cv::boundingRect(contours[i]);
			npt += contours[i].size();
			if (i == 0)
				roiUnion = rect;

			else
				roiUnion |= rect;
		}
		int nSkip = (int)npt / nSamples;

		int ystart = roiUnion.y, yend = roiUnion.y + roiUnion.height;
		int xstart = roiUnion.x, xend = roiUnion.x + roiUnion.width;
		std::set<std::pair<int, int>> sampledPoints;//存储
		for (int x = xstart; x < xend; x++) {
			for (int y = ystart; y < yend; y++) {
				if ((fgMask(y, x) & 255) != 255)continue;
				IPoint P; float depth;
				if (get3D(x, y, P.center, depth))
				{
					int prevy = y - 1;
					int nexty = y + 1;
					float prev = 0;
					float next = 0;
					if (prevy >= 0)prev = fabs(dy(prevy, x));
					if (nexty < depthMask.rows)next = fabs(dy(nexty, x));
					float edge = fabs(dy(y, x));
					if (edge > prev && edge > next) {
						P.edgew = edge;
						c3d.push_back(P);
						sampledPoints.insert({ x, y });
					}
				}

			}
		}
		for (int y = ystart; y < yend; y++) {
			for (int x = xstart; x < xend; x++) {
				if ((fgMask(y, x) & 255) != 255)continue;
				IPoint P; float depth;
				if (sampledPoints.count({ x, y }) > 0) {
					continue;
				}
				if (get3D(x, y, P.center, depth))
				{
					int prevx = x - 1;
					int nextx = x + 1;
					float prev = 0;
					float next = 0;
					if (prevx >= 0)prev = fabs(dx(y, prevx));
					if (nextx < depthMask.cols)next = fabs(dx(y, nextx));
					float edge = fabs(dx(y, x));
					if (edge > prev && edge > next) {
						P.edgew = edge;
						c3d.push_back(P);
					}
				}

			}
		}

		float wsum = 0;
		for (auto& cd : c3d) {
			wsum += cd.edgew;
		}
		wsum /= c3d.size();
		for (auto& cd : c3d) {
			cd.edgew /= wsum;
		}
	}

	static void sample(std::vector<IPoint>& c3d, CVRResult& rr, int nSamples, Rect roiRect = Rect(0, 0, 0, 0), Size imgSize = Size(0, 0))
	{
		if (imgSize.width == 0 || imgSize.height == 0)
			imgSize = rr.img.size();

		if (roiRect.width == 0 || roiRect.height == 0)
			roiRect = Rect(0, 0, imgSize.width, imgSize.height);

		CVRProjector prj(rr.mats, imgSize);

		Mat rgray = cv::convertBGRChannels(rr.img, 1);
		Mat1s dx, dy;
		cv::Sobel(rgray, dx, CV_16S, 1, 0);
		cv::Sobel(rgray, dy, CV_16S, 0, 1);

		/*Mat1s abs_dx = cv::abs(dx);
		Mat1s abs_dy = cv::abs(dy);
		Mat1i grad_sum;
		cv::add(abs_dx, abs_dy, grad_sum, noArray(), CV_32S);*/

		//imwrite("D:\\RBOT_dataset\\can\\dx.png", grad_sum);

		Mat1b fgMask = rr.getMaskFromDepth();

		_sample(roiRect, rr.depth, prj, c3d, nSamples, fgMask, dx, dy);
	}
};

//处理c3d中点
std::sort(c3d.begin(), c3d.end(), [](const IPoint& a, const IPoint& b) {
	return a.edgew < b.edgew;
	});

// 2. 计算需要删除的元素数量
size_t removeCount = c3d.size() * 0.25;

// 3. 删除前25%的元素
c3d.erase(c3d.begin(), c3d.begin() + removeCount);
float wsum = 0;
for (auto& cd : c3d) {
	wsum += cd.edgew;
}
wsum /= c3d.size();
for (auto& cd : c3d) {
	cd.edgew /= wsum;
}


//灰度转rgb
Mat3f rgbImage(gradf.rows, gradf.cols);

// 将灰度图的每个像素复制到RGB图像的三个通道
for (int i = 0; i < gradf.rows; ++i) {
	for (int j = 0; j < gradf.cols; ++j) {
		float grayValue = gradf(i, j); // 获取灰度图中的像素值
		// 将灰度值赋给RGB图像的对应位置的R、G、B三个通道
		rgbImage(i, j)[0] = grayValue; // B
		rgbImage(i, j)[1] = grayValue; // G
		rgbImage(i, j)[2] = grayValue; // R
	}
}

//// 此时，rgbImage就是转换后的RGB图像，可以进行显示或保存等操作
//imshow("RGB Image", rgbImage);
//waitKey(0);

// fx提前终止
struct EdgeResponse
{
	Mat originimg; int fi; float prev; float cur;
	Mat1f getImageField(Mat img, int nLayers = 1)
	{
		img = cv::convertBGRChannels(img, 1);

		auto getF = [](const Mat1b& gray, Size dsize) {
			Mat1f dx, dy;
			cv::Sobel(gray, dx, CV_32F, 1, 0, 7);
			cv::Sobel(gray, dy, CV_32F, 0, 1, 7);
			for_each_2(DWHN1(dx), DN1(dy), [](float& dx, float dy) {
				dx = fabs(dx) + fabs(dy);
				});
			if (dx.size() != dsize)
				dx = imscale(dx, dsize, INTER_LINEAR);
			return dx;
		};

		Mat1f f = getF(img, img.size());
		for (int i = 1; i < nLayers; ++i)
		{
			img = imscale(img, 0.5);
			f += getF(img, f.size());
		}
		float vmax = cv::maxElem(f);
		f *= 1.f / vmax;
		return 1.f - f;
	}
	Mat1f gradf;
public:
	EdgeResponse(Mat img)
	{
		gradf = getImageField(img);
		originimg = img.clone();
		prev = 0.f;
		cur = 0.f;
	}
	EdgeResponse(Mat img, int fi)
	{
		gradf = getImageField(img);
		originimg = img.clone();
		this->fi = fi;
		prev = 0.f;
		cur = 0.f;
	}
	float Fx(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints)
	{
		Mat1f f = gradf;

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		const Matx33f R = pose.R;
		const Point3f t = pose.t;
		int npt = cpoints.size();

		const float fx = K(0, 0), fy = K(1, 1);

		float F_x = 0.f;

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			const float w = cpoints[i].edgew;
			F_x += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);
		}
		return F_x;
	}
	int _edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, float eps)
	{
		Mat1f f = gradf;
		Mat3f rgbImage(gradf.rows, gradf.cols);
		Mat img1 = originimg.clone();

		// 将灰度图的每个像素复制到RGB图像的三个通道
		for (int i = 0; i < gradf.rows; ++i) {
			for (int j = 0; j < gradf.cols; ++j) {
				float grayValue = gradf(i, j); // 获取灰度图中的像素值
				// 将灰度值赋给RGB图像的对应位置的R、G、B三个通道
				rgbImage(i, j)[0] = grayValue; // B
				rgbImage(i, j)[1] = grayValue; // G
				rgbImage(i, j)[2] = grayValue; // R
			}
		}
		Mat3b rgbImage8bit;
		rgbImage.convertTo(rgbImage8bit, CV_8U, 255.0);
		//imwrite("D:\\RBOT_dataset\\can\\gradf.jpg", rgbImage8bit);

		Mat1f dfx, dfy;
		cv::Sobel(f, dfx, CV_32F, 1, 0);
		cv::Sobel(f, dfy, CV_32F, 0, 1);

		auto _resample = [](const Mat1f& f, float x, float y) {
			int xi = int(x), yi = int(y);
			if (uint(xi) < uint(f.cols - 1) && uint(yi) < uint(f.rows - 1))
			{
				const float* p = f.ptr<float>(yi, xi);
				int stride = stepC(f);
				float wx = x - xi, wy = y - yi;
				float a = p[0] + wx * (p[1] - p[0]);
				float b = p[stride] + wx * (p[stride + 1] - p[stride]);
				return a + wy * (b - a);
			}
#define _clip(v, vmax) (v<0? 0 : v>vmax? vmax : v)
			return f(_clip(yi, f.rows - 1), _clip(xi, f.cols - 1));
#undef _clip
		};

		const Matx33f R = pose.R;
		const Point3f t = pose.t;
		int npt = cpoints.size();

		const float fx = K(0, 0), fy = K(1, 1);

		Matx66f JJ = Matx66f::zeros();
		Vec6f J(0, 0, 0, 0, 0, 0);

		for (size_t i = 0; i < cpoints.size(); ++i)
		{
			//	continue;
			const Point3f Q = R * cpoints[i].center + t;
			const Point3f q = K * Q;

			const float X = Q.x, Y = Q.y, Z = Q.z;

			const float a = fx / Z, b = -fx * X / (Z * Z), c = fy / Z, d = -fy * Y / (Z * Z);

			Point2f pt(q.x / q.z, q.y / q.z);

			//const int x = int(q.x + 0.5), y = int(q.y + 0.5);
			if (uint(int(pt.x + 0.5)) >= uint(f.cols) || uint(int(pt.y + 0.5)) >= uint(f.rows))
				continue;

			circle(rgbImage8bit, pt, 1, Scalar(255, 255, 0), -1);
			circle(img1, pt, 1, Scalar(255, 255, 0), -1);

			//Vec2f nx(dfx(y, x), dfy(y, x));
			Vec2f nx(_resample(dfx, pt.x, pt.y), _resample(dfy, pt.x, pt.y));
			nx = normalize(nx);


			Vec3f n_dq_dQ(nx[0] * a, nx[1] * c, nx[0] * b + nx[1] * d);

			auto dt = n_dq_dQ.t() * R;
			auto dR = cpoints[i].center.cross(Vec3f(dt.val[0], dt.val[1], dt.val[2]));

			Vec6f j(dt.val[0], dt.val[1], dt.val[2], dR.x, dR.y, dR.z);


			const float w = cpoints[i].edgew;
			const float wf = (w * _resample(f, pt.x, pt.y));
			//if (fi == 4) {
			//	cout << "ptsimg" << endl;
			//	//cout << w << endl;
			//	cout << _resample(f, pt.x, pt.y) << endl;
			//	//imshow("ptsimg", rgbImage8bit);
			//	//waitKey(0);
			//}

			prev += w * _resample(f, pt.x, pt.y) * _resample(f, pt.x, pt.y);

			J += wf * j;

			JJ += w * j * j.t();
		}

		/*imwrite("D:\\RBOT_dataset\\can\\gradfpts.jpg", rgbImage8bit);
		imwrite("D:\\RBOT_dataset\\can\\imgpts.jpg", img1);*/

		const float lambda = 100000000.f * npt / 200.f;

		for (int i = 0; i < 3; ++i)
			JJ(i, i) += lambda * 1000.f;

		for (int i = 3; i < 6; ++i)
			JJ(i, i) += lambda;

		Vec6f p;// = -JJ.inv() * J;
		if (solve(JJ, -J, p))
		{
			cv::Vec3f dt(p[0], p[1], p[2]);
			cv::Vec3f rvec(p[3], p[4], p[5]);
			Matx33f dR;
			cv::Rodrigues(rvec, dR);

			pose.t = pose.R * dt + pose.t;
			pose.R = pose.R * dR;

			if (Fx(pose, K, cpoints) > prev)return 0;

			float diff = p.dot(p);
			//printf("diff=%f\n", sqrt(diff));

			return diff < eps* eps ? 0 : 1;
		}

	}
	bool edgeupdate(Optimizer::PoseData& pose, const Matx33f& K, const std::vector<CPoint>& cpoints, int maxItrs, float eps)
	{
		/*for (int itr = 0; itr < maxItrs; ++itr) {
			if (this->_edgeupdate(pose, K, cpoints, eps) <= 0)
				return false;
		}*/
		for (int itr = 0; itr < maxItrs; ++itr) {
			if (this->_edgeupdate(pose, K, cpoints, eps) <= 0)
				return false;

			prev = 0.f;
		}
		/*for (int itr = 0; itr < maxItrs; ++itr) {
			float prev = Fx(pose, K, cpoints);
			this->_edgeupdate(pose, K, cpoints, eps);
			if (Fx(pose, K, cpoints) > prev)return false;
		}*/
		return true;
	}
};