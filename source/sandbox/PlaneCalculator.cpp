#include "stdafx.h"

#include "PlaneCalculator.h"

#include <core/cvs_exception.h>
#include <core/logger.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl/common/io.h>

#include <boost/make_shared.hpp>

#include <cmath>

class PlaneCalculatorHelper {
public:
	float x, y, z;
	float weight;
};

class PlaneDistanceHelper {
public:
	cv::Point3d point;
	float distance;
};


class PointAngleHelper {
public:
	cv::Point3d point;
	float angle;
};

static cv::Mat preparePallete(cv::ColormapTypes colormap)
{
	cv::Mat pallete;
	cv::Mat_<unsigned char> tempPallete(1, 256);
	for (int i = 0; i < 256; ++i) tempPallete(0, i) = static_cast<unsigned char>(i);
	cv::applyColorMap(tempPallete, pallete, colormap);
	return pallete;
}

static cv::Vec3b pickColor(const cv::Mat_<cv::Vec3b>& pallete, float val)
{
	return pallete.at<cv::Vec3b>(0, std::min(255, std::max(0, static_cast<int>(val * 255))));
}

void setColor(pcl::PointXYZRGBA& pt, const cv::Vec3b color, uint8_t a = 255) { pt.r = color.val[2];  pt.g = color.val[1];  pt.b = color.val[0];  pt.a = a; };


static void normalizeValues(const cv::Mat_<float>& intensityImage, cv::Mat_ <uint8_t> & colorImage, double min, double max) {
	for (int row = 0; row < intensityImage.rows; ++row) {
		for (int col = 0; col < intensityImage.cols; ++col) {
			colorImage(row, col) = (intensityImage(row, col) - min) / (max - min) * 255 + 0.5;
		}
	}
}

static void filterContours( const std::vector<std::vector<cv::Point> >& contours_in, std::vector<std::vector<cv::Point> >& contours_out, int image_area) {
	for (const auto& next_contour : contours_in) {
		double next_contour_area = cv::contourArea(next_contour);

		if ( image_area < next_contour_area * 100 ) {
			continue;
		}

		if (next_contour_area < 150) { // 0.1%
			continue;
		}

		double perimeter = cv::arcLength(next_contour, true);

		// circle has a minimal perimeter square from all figures with given area;
		double circularity = pow((perimeter / (2 * M_PI)), 2) * M_PI / next_contour_area;

		if (circularity > 2) {
			continue;
		}

		contours_out.push_back(next_contour);
	}
}

static double calculatePlaneDistance(const pcl::ModelCoefficients::Ptr& plane, cv::Point3d point) 
{
	return (plane->values[0] * point.x + plane->values[1] * point.y + plane->values[2] * point.z + plane->values[3]) /
		sqrt(plane->values[0] * plane->values[0] + plane->values[1] * plane->values[1] + plane->values[2] * plane->values[2]);
}

static double calculateRange(const std::vector<double>& distances, double median, double IQR) {

	double left_bound  = median - IQR;
	double right_bound = median + IQR;

	left_bound = *std::min_element(distances.begin(), distances.end());
	right_bound = *std::max_element(distances.begin(), distances.end());

	int N = 50001;
	int bins = N - 1;

	std::vector<int> hist_vals;
	hist_vals.resize(static_cast<size_t> (bins) );
	std::fill(hist_vals.begin(), hist_vals.end(), 0);

	double step = (right_bound - left_bound) / (bins);

	int hist_idx = 0;

	for (size_t i = 0; i < distances.size(); ++i) {
		double next_val = distances[i];
		while ( ( left_bound + step * ( hist_idx + 1) < next_val ) && hist_idx < bins - 1) {
			hist_idx++;
		}

		hist_vals[hist_idx]++;
	}

	hist_vals[0] = 0;
	hist_vals[bins - 1] = 0;

	auto max_it = std::max_element(hist_vals.begin(), hist_vals.end());

	// walk to the left
	auto left_walker_it = max_it;
	bool found_left_fwhm = false;

	while (true) {
		if ( *left_walker_it < *max_it / 2 ) {
			found_left_fwhm = true;
			break;
		}
		if (left_walker_it == hist_vals.begin()) {
			break;
		}

		--left_walker_it;
	}
	double left_wfhm = std::distance(left_walker_it, max_it) * step;

	// walk to the right
	auto right_walker_it = max_it;
	bool found_right_fwhm = false;

	while (true) {
		if (*right_walker_it < *max_it / 2) {
			found_right_fwhm = true;
			break;
		}
		if (right_walker_it + 1 == hist_vals.end()) {
			break;
		}

		++right_walker_it;
	}

	double right_wfhm = std::distance(max_it, right_walker_it) * step;

	if (!found_left_fwhm && !found_right_fwhm) {
		return IQR / 2;
	}

	if (!found_left_fwhm ) {
		return right_wfhm / 2.355 * 3; // convert to 3sigma
	}

	if (!found_right_fwhm) {
		return left_wfhm / 2.355 * 3; // convert to 3sigma
	}

	return (right_wfhm + left_wfhm) / 2 / 2.355 * 3;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PlaneCalculator::calculateDistanceToPlane(const cv::Mat_<cv::Vec3f>& cloud, const pcl::ModelCoefficients::Ptr& plane, const cv::InputArray& mask, const cv::ColormapTypes& colormap, uint8_t a)
{

	if (!mask.empty() && cloud.size() != mask.size())
		throw cvs::core::CVSException("Input cloud and mask must have the same size!");

	cv::Mat pallete = preparePallete(colormap);

	cv::Mat_<bool> coloredPoints = mask.empty() ? cv::Mat_<bool>::ones(cloud.size()) : mask.getMat();

	std::vector< PlaneDistanceHelper > planeDistance;

	float max_distance = 0;

	for (int row = 0; row < cloud.rows; ++row) {
		for (int col = 0; col < cloud.cols; ++col) {
			if (coloredPoints(row, col))
			{

				const auto toPoint = [](const cv::Vec3f &pt) { cv::Point3d point; point.x = pt.val[0]; point.y = pt.val[1]; point.z = pt.val[2]; return point; };
				cv::Point3d point = toPoint(cloud(row, col));
				PlaneDistanceHelper nextHelper;
				nextHelper.point = point;
				nextHelper.distance = calculatePlaneDistance(plane, point);
				planeDistance.push_back(nextHelper);

				max_distance = std::max(nextHelper.distance, max_distance);

			}
		}
	}

	std::sort(planeDistance.begin(), planeDistance.end(), [](const PlaneDistanceHelper& a, const PlaneDistanceHelper& b) {
		return a.distance < b.distance;
	});

	double quantile_25 = planeDistance[planeDistance.size() * 0.25].distance;
	double quantile_50 = planeDistance[planeDistance.size() * 0.50].distance;
	double quantile_75 = planeDistance[planeDistance.size() * 0.75].distance;

	double prelim_IQR = 1.5 * (quantile_75 - quantile_25);

	std::vector<double> distances;
	distances.resize(planeDistance.size());
	std::transform(planeDistance.begin(), planeDistance.end(), distances.begin(), [](const auto& val) {
		return val.distance;
	});

	double real_width = calculateRange(distances, quantile_50, prelim_IQR);


	auto coloredCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
	const auto toPcl = [](const cv::Point3d &pt) { pcl::PointXYZRGBA point; point .x = pt.x; point.y = pt.y; point.z = pt.z; return point; };

	for (size_t i = 0; i < planeDistance.size(); ++i) {
		{
			pcl::PointXYZRGBA point = toPcl(planeDistance[i].point);

			double distance = planeDistance[i].distance;
			double weight = exp(-distance / real_width );
			weight = 1 / (distance / real_width / 2 + 1);

			setColor(point, pickColor(pallete, weight), a);
			coloredCloud->points.push_back(point);
		}
	}
	return coloredCloud;
}


pcl::ModelCoefficients::Ptr  PlaneCalculator::calculatePlaneHintHoles(const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat_<float>& m_intensityImage, uint8_t a) {
	cv::Mat_ <uint8_t> grayImage(m_intensityImage.rows, m_intensityImage.cols, (uint8_t)0 );

	double min, max;

	cv::minMaxLoc(m_intensityImage, &min, &max, nullptr, nullptr);
	normalizeValues(m_intensityImage, grayImage, min, max);

	cv::Mat_ <uint8_t> binImage(m_intensityImage.rows, m_intensityImage.cols, (uint8_t)0);

	cv::threshold(grayImage, binImage, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	cv::bitwise_not(binImage, binImage);

	cv::Mat result(grayImage.size(), CV_8UC3, cv::Scalar(0, 0, 0) ) ;

	std::vector<std::vector<cv::Point> >contours;
	std::vector <cv::Vec4i> hierarchy;
	cv::findContours(binImage.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point());

	std::vector<std::vector<cv::Point> >contours_filtered;

	filterContours(contours, contours_filtered, m_intensityImage.rows * m_intensityImage.cols);

	std::vector<cv::Point> centers;
	std::vector<double> areas;

	for (int i = 0; i < contours_filtered.size(); i++)
	{
		//cv::drawContours(result, contours_filtered, i, cv::Scalar(255, 0, 0), 1, 8);

		cv::Moments m = cv::moments(contours_filtered[i], true);
		cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
		centers.push_back(center);

		double area = cv::contourArea(contours_filtered[i]);
		areas.push_back(area);
	}


	int left_x = 0;
	int right_x = m_intensityImage.cols;

	int left_y = 0;
	int right_y = m_intensityImage.rows;

	if (centers.size() >= 5) {
		std::sort(centers.begin(), centers.end(), [](const cv::Point& a, const cv::Point& b) {
			return a.x < b.x;
		});

		int quantile_25 = 0.25 * centers.size();
		int quantile_50 = 0.5 * centers.size();
		int quantile_75 = 0.75 * centers.size();

		left_x = centers[quantile_25].x;
		right_x = centers[quantile_75].x;

		std::sort(centers.begin(), centers.end(), [](const cv::Point& a, const cv::Point& b) {
			return a.y < b.y;
		});

		left_y = centers[quantile_25].y;
		right_y = centers[quantile_75].y;


		std::sort(areas.begin(), areas.end(), [](double a, double b) {
			return a < b;
		});

		double median_area = areas[quantile_50];

		int offset = sqrt(median_area + 1) * 2;

		left_x = std::max(0, left_x - offset);
		left_y = std::max(0, left_y - offset);

		right_x = std::min(m_intensityImage.cols, right_x + offset);
		right_y = std::min(m_intensityImage.rows, right_y + offset);

	}

	// back bitwice. It will be a mask for cloud points
	cv::bitwise_not(binImage, binImage);

	cv::Mat_<bool> final_filter (binImage.rows, binImage.cols, false);

	for (size_t i = 0; i < final_filter.rows; ++i) {
		for (size_t j = 0; j < final_filter.cols; ++j) {
			if (binImage(i, j) > 1 && j > left_x && j < right_x && i > left_y && i < right_y) {
				final_filter(i, j) = true;
			}

		}
	}

	return calculatePlaneTheilSein(cloud, final_filter);
}

pcl::ModelCoefficients::Ptr PlaneCalculator::calculatePlaneTheilSein(const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat_<bool>&  mask ) {

	pcl::ModelCoefficients::Ptr builded_plane(new pcl::ModelCoefficients);
	builded_plane->values.resize(4);


	if (!mask.empty() && cloud.size() != mask.size())
		throw cvs::core::CVSException("Input cloud and mask must have the same size!");

	cv::Mat_<bool> coloredPoints = mask;

	const auto toPcl = [](const cv::Vec3f &pt) { pcl::PointXYZRGBA point; point.x = pt.val[0]; point.y = pt.val[1]; point.z = pt.val[2]; return point; };

	std::vector< cv::Point3d > points;

	for (int row = 0; row < cloud.rows; ++row) {
		for (int col = 0; col < cloud.cols; ++col) {
			if (coloredPoints(row, col))
			{
				pcl::PointXYZRGBA point = toPcl(cloud(row, col));

				cv::Point3d _3DPoint;
				_3DPoint.x = point.x;
				_3DPoint.y = point.y;
				_3DPoint.z = point.z;

				points.push_back(_3DPoint);
			}
		}
	}

	//std::sort(points.begin(), points.end(), [](const cv::Point3d& a, const cv::Point3d& b) {
	//	return a.z < b.z;
	//});

	std::random_shuffle(points.begin(), points.end());

	std::vector< cv::Point3d > calculated_norm_vectors;
	
	
	std::vector< PointAngleHelper  > calculated_angles;

	double norm_111 = sqrt(3);

	for (size_t i = 0; i < points.size() - 3; ++i) {
		auto p0 = points[i];
		auto p1 = points[i + 1];
		auto p2 = points[i + 2];

		auto vec_1 = p1 - p0;
		auto vec_2 = p2 - p0;

		float ex = vec_1.y * vec_2.z - vec_2.y * vec_1.z;
		float ey = - ( vec_1.x * vec_2.z - vec_2.x * vec_1.z );
		float ez = (vec_1.x * vec_2.y - vec_2.x * vec_1.y);

		if (ex < 0) {
			ex *= -1;
			ey *= -1;
			ez *= -1;

		}
		
		double norm = sqrt( ex * ex + ey * ey + ez * ez );

		double ex_norm = ex / norm;
		double ey_norm = ey / norm;
		double ez_norm = ez / norm;

		// with vector (1, 1, 1)
		double cos_angle = (1 * ex_norm + 1 * ey_norm + 1 * ez_norm) / (norm_111);

		cv::Point3d next_norm_vector;
		next_norm_vector.x = ex_norm;
		next_norm_vector.y = ey_norm;
		next_norm_vector.z = ez_norm;

		PointAngleHelper pointHelper;
		pointHelper.point = next_norm_vector;
		pointHelper.angle = cos_angle;

		calculated_angles.push_back(pointHelper);

	}

	std::nth_element(calculated_angles.begin(), calculated_angles.begin() + calculated_angles.size() / 2, calculated_angles.end(),
		[](const auto& a, const auto& b) {
		return a.angle < b.angle;
	});

	auto median_vector = calculated_angles[calculated_angles.size() / 2].point;

	// calculating constant

	std::vector<double> intercepts;
	intercepts.resize(points.size());

	for (size_t i = 0; i < points.size(); ++i) {
		auto p0 = points[i];

		double intercept = -(median_vector.x * p0.x + median_vector.y * p0.y + median_vector.z * p0.z);
		intercepts[i] = intercept;
	}

	std::nth_element(intercepts.begin(), intercepts.begin() + intercepts.size() / 2, intercepts.end(),
		[](const auto& a, const auto& b) {
		return a < b;
	});

	double median_intercept = intercepts[intercepts.size() / 2];

	builded_plane->values[0] = median_vector.x;
	builded_plane->values[1] = median_vector.y;
	builded_plane->values[2] = median_vector.z;
	builded_plane->values[3] = median_intercept;

	return builded_plane;
}


pcl::ModelCoefficients::Ptr PlaneCalculator::calculatePlaneLSE(
	const cv::Mat_<cv::Vec3f>& cloud,
	cv::InputArray mask,
	cv::InputArray weights,
	uint8_t a)
{
	pcl::ModelCoefficients::Ptr builded_plane(new pcl::ModelCoefficients);
	builded_plane->values.resize(4);

	
	if (!weights.empty() && cloud.size() != weights.size())
		throw cvs::core::CVSException("Input cloud and weights mats must have the same size!");
	if (!mask.empty() && cloud.size() != mask.size())
		throw cvs::core::CVSException("Input cloud and mask must have the same size!");

	cv::Mat_<bool> coloredPoints = mask.empty() ? cv::Mat_<bool>::ones(cloud.size()) : mask.getMat();
	cv::Mat_<float> normalizedWeights = weights.empty() ? cv::Mat_<float>::ones(cloud.size()) : weights.getMat();

	const auto toPcl = [](const cv::Vec3f &pt) { pcl::PointXYZRGBA point; point.x = pt.val[0]; point.y = pt.val[1]; point.z = pt.val[2]; return point; };

	std::vector< PlaneCalculatorHelper > weightedPoints;

	for (int row = 0; row < cloud.rows; ++row) {
		for (int col = 0; col < cloud.cols; ++col) {
			if (coloredPoints(row, col))
			{
				pcl::PointXYZRGBA point = toPcl(cloud(row, col));
				float weight = normalizedWeights(row, col);

				PlaneCalculatorHelper nextWeightedPoint;
				nextWeightedPoint.x = point.x;
				nextWeightedPoint.y = point.y;
				nextWeightedPoint.z = point.z;
				nextWeightedPoint.weight = weight;

				weightedPoints.push_back(nextWeightedPoint);
			}
		}
	}

	Eigen::MatrixXf X(weightedPoints.size(), 3);
	Eigen::MatrixXf y(weightedPoints.size(), 1);
	Eigen::MatrixXf Xw (weightedPoints.size(), 3 );

	for ( size_t i = 0; i < weightedPoints.size(); ++i ) {
		const PlaneCalculatorHelper& nextWeightedPoint = weightedPoints[i];
		y(i, 0) = nextWeightedPoint.x;

		X(i, 0) = nextWeightedPoint.y;
		X(i, 1) = nextWeightedPoint.z;
		X(i, 2) = 1;

		Xw(i, 0) = X(i, 0) * nextWeightedPoint.weight;
		Xw(i, 1) = X(i, 1) * nextWeightedPoint.weight;
		Xw(i, 2) = X(i, 2) * nextWeightedPoint.weight;

	}

	auto Xt = X.transpose();
	auto Xwt = Xw.transpose();

	//auto coeffs = ( (Xwt * X).inverse()) * Xwt * y;

	auto coeffs = ((Xt * X).inverse()) * Xt * y;


	builded_plane->values[0] = 1;
	builded_plane->values[1] = -coeffs(0, 0);
	builded_plane->values[2] = -coeffs(1, 0);
	builded_plane->values[3] = -coeffs(2, 0);

	return builded_plane;
}
