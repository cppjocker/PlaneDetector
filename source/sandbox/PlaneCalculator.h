#pragma once

#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


class PlaneCalculator {
public:
	static pcl::PointCloud<pcl::PointXYZRGBA>::Ptr calculateDistanceToPlane(const cv::Mat_<cv::Vec3f>& cloud, const pcl::ModelCoefficients::Ptr& plane, const cv::InputArray& mask = cv::noArray(), const cv::ColormapTypes& colormap = cv::ColormapTypes::COLORMAP_JET, uint8_t a = 255);

	// estimation of plane coefficients through binarization, countour selection of table holes, adaptive filters and only then applying Theil-sen regression
	static pcl::ModelCoefficients::Ptr  calculatePlaneHintHoles(const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat_<float>& m_intensityImage, uint8_t a = 255);

	// simple LSE regression for place coefficients estimation
	static pcl::ModelCoefficients::Ptr calculatePlaneLSE(const cv::Mat_<cv::Vec3f>& cloud, cv::InputArray mask, cv::InputArray weights, uint8_t a = 255);
	// Theill-sen regression for place coefficients estimation
	static pcl::ModelCoefficients::Ptr calculatePlaneTheilSein(const cv::Mat_<cv::Vec3f>& cloud, const cv::Mat_<bool>&  mask);

};