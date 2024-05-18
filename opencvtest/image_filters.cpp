#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"


void applyGaussianBlur(cv::Mat& local_mat, int radius, cv::Mat& result)
{
    cv::GaussianBlur(local_mat, result, cv::Size(radius, radius), 0);
}

void applyMedianBlur(cv::Mat& local_mat, int radius, cv::Mat& result)
{
    cv::medianBlur(local_mat, result, radius);
}

void applyEdgeDetection(cv::Mat& local_mat, double low_threshold, double high_threshold, cv::Mat& result)
{
    cv::Mat gray;
    cv::cvtColor(local_mat, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, result, low_threshold, high_threshold);
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

void applyLocalThreshold(cv::Mat& local_mat, int blockSize, cv::Mat& result)
{
    cv::Mat gray;
    cv::cvtColor(local_mat, gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(gray, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 2);
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

void applyGlobalThreshold(cv::Mat& local_mat, double threshold, cv::Mat& result)
{
    cv::Mat gray;
    cv::cvtColor(local_mat, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, result, threshold, 255, cv::THRESH_BINARY);
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

void applyColorMap(cv::Mat& local_mat, cv::Mat& result, int type)
{
    cv::applyColorMap(local_mat, result, type);
}

void applyrotation(cv::Mat& local_mat, cv::Mat& result, double angle)
{
    cv::Mat rotationMatrix;
    cv::Point2f center(local_mat.cols / 2.0, local_mat.rows / 2.0);
    rotationMatrix = cv::getRotationMatrix2D(center, 270.0, 1.0);
    cv::warpAffine(local_mat, result, rotationMatrix, local_mat.size());
    
}

void applyColorSpaceConvertion(cv::Mat& local_mat, cv::Mat& result, int code)
{
    cv::cvtColor(local_mat, result, code);
    if (code == 6) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
}