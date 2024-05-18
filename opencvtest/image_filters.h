#pragma once

#include <opencv2/core.hpp>

void applyGaussianBlur(cv::Mat& local_mat, int radius, cv::Mat& result);
void applyMedianBlur(cv::Mat& local_mat, int radius, cv::Mat& result);
void applyEdgeDetection(cv::Mat& local_mat, double low_threshold, double high_threshold, cv::Mat& result);
void applyLocalThreshold(cv::Mat& local_mat, int blockSize, cv::Mat& result);
void applyGlobalThreshold(cv::Mat& local_mat, double threshold, cv::Mat& result);
void applyColorSpaceConvertion(cv::Mat& local_mat, cv::Mat& result, int code);
void applyColorMap(cv::Mat& local_mat, cv::Mat& result, int type);
void applyrotation(cv::Mat& local_mat, cv::Mat& result,double angle);
