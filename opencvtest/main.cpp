#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include "mpi.h"
#include "image_filters.h"

using namespace std;
using namespace cv;

// Function declarations
void displayMenu(int& choice);
void readParameters(int& choice, string& input_path, string& saved_output_image, int& radius, int& blockSize, double& low_threshold, double& high_threshold, int& type, int& code, double& angle);
void applyImageProcessing(Mat& local_mat, Mat& result, int& choice, int& radius, int& blockSize, double& low_threshold, double& high_threshold, int& type, int& code, double& angle);

int main() {
    // Variables region
    int size, rank, im_row = 0, im_col = 0, radius = 0, blockSize = 0, choice = 0, type = 0, code = 0;
    double low_threshold = 0.0, high_threshold = 0.0, duration, angle = 0.0;
    double start_time, end_time;
    Mat image;
    string saved_output_image, input_path, output_path = "C:/Users/islam/Downloads/parallel_output/";

    MPI_Init(nullptr, nullptr);
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        displayMenu(choice);
        readParameters(choice, input_path, saved_output_image, radius, blockSize, low_threshold, high_threshold, type, code, angle);

        start_time = MPI_Wtime();
        image = imread(input_path);
        if (!image.empty()) {
            im_row = image.rows;
            im_col = image.cols;
        }
        else {
            cerr << "Failed to load image!" << endl;
            MPI_Finalize();
            exit(1);
        }
    }


    // Broadcast image dimensions to all processes
    MPI_Bcast(&im_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&im_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //broadcast parameters
    MPI_Bcast(&radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&high_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&low_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&code, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&angle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divide image among processes
    int rows_per_process = im_row / size;
    int remaining_rows = im_row % size;
    int local_start_row = rank * rows_per_process + min(rank, remaining_rows);
    int local_end_row = local_start_row + rows_per_process + (rank < remaining_rows ? 1 : 0);

    int local_rows = local_end_row - local_start_row;
    int local_image_size = local_rows * im_col * 3;
    vector<unsigned char> local_image(local_image_size);


    vector<unsigned char> image_buffer;
    if (rank == 0) {
        image_buffer.resize(im_row * im_col * 3); // Buffer for the entire image
        image_buffer.assign(image.data, image.data + im_row * im_col * 3); // Copy image data to buffer
    }

  

    MPI_Scatter(image_buffer.data(), local_image_size, MPI_UNSIGNED_CHAR, local_image.data(), local_image_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Convert local image buffer to cv::Mat
    Mat local_mat(local_rows, im_col, CV_8UC3, local_image.data());

    Mat result;

    applyImageProcessing(local_mat, result, choice, radius, blockSize, low_threshold, high_threshold, type, code, angle);

    // Gather processed image data to master process
    vector<unsigned char> image_result_buffer(im_row * im_col * 3);
    MPI_Gather(result.data, local_image_size, MPI_UNSIGNED_CHAR, image_result_buffer.data(), local_image_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        string output_save_path = output_path + saved_output_image;
        Mat final_image(im_row, im_col, CV_8UC3, image_result_buffer.data());
        bool success = imwrite(output_save_path, final_image);

        if (success) {
            end_time = MPI_Wtime();
            duration = (end_time - start_time) * 1000;
            cout << "operation completed successfully in " << duration << " milliseconds." << endl;
            cout << "Image saved successfully as: " << saved_output_image << endl;
            cout << "Thank you for using Parallel Image Processing with MPI.\n\n";
        }
        else {
            cerr << "Error: Failed to save image!" << endl;
            MPI_Finalize();
            exit(1);
        }
    }

    MPI_Finalize();
    return 0;
}

void displayMenu(int& choice) {
    cout << "Welcome to parallel image processing with MPI\n\n";
    cout << "Please choose an image processing operation: \n";
    cout << "1. Gaussian Blur\n2. Median Blur\n3. Edge Detection\n4. Local Threshold\n5. Global Threshold\n6. Color space conversion\n7. Color Map\n8. Rotation\n";
    cout << "Please choose an image processing operation (1-8): ";
    cin >> choice;
}

void readParameters(int& choice, string& input_path, string& saved_output_image, int& radius, int& blockSize, double& low_threshold, double& high_threshold, int& type, int& code, double& angle)
{
    switch (choice) {
        case 1: {
            cout << "You have selected Gaussian Blur.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter the file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Please enter the blur radius (+ve & odd e.g., 3 ): ";
            cin >> radius;
            cout << "Processing input image with Gaussian Blur...\n\n";
            break;
        }
        case 2: {
            cout << "You have selected Median Blur.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter the file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Please enter the blur radius (e.g., 3): ";
            cin >> radius;
            cout << "Processing input image with Median Blur...\n\n";
            break;
        }
        case 3: {
            cout << "You have selected Edge detection.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter the file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the low threshold for edge detection (e.g., 30): ";
            cin >> low_threshold;
            cout << "Enter the high threshold for edge detection (e.g., 90): ";
            cin >> high_threshold;
            cout << "Processing input image with Edge detection...\n\n";
            break;
        }
        case 4: {
            cout << "You have selected Local Threshold.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter the file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the block size for Local Threshold (e.g., 3, 5, 7, 11): ";
            cin >> blockSize;
            cout << "Processing input image with Local Threshold...\n\n";
            break;
        }
        case 5: {
            cout << "You have selected Global Threshold.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the Threshold value for Global Threshold (e.g., 150): ";
            cin >> high_threshold;
            cout << "Processing input image with Global Threshold...\n\n";
            break;
        }
        case 6: {
            cout << "You have selected Color Space Conversion.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the color code for Color Space Conversion (e.g., BGR2RGB = 4,BGR2HSV_FULL = 66,COLOR_BGR2YCrCb = 36...etc): ";
            cin >> code;
            cout << "Processing input image with Color Space Conversion...\n\n";
            break;
        }
        case 7: {
            cout << "You have selected Color Map.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the color map type for Color Space Conversion (e.g., 0- autumn, 1- bone...etc): ";
            cin >> type;
            cout << "Processing input image with Color Map...\n\n";
            break;
        }
        case 8: {
            cout << "You have selected Rotation.\n\n";
            cout << "Please enter the path of the input image (e.g., E:\\input.jpg): ";
            cin >> input_path;
            cout << "Please enter file name for the output image (e.g., output.jpg): ";
            cin >> saved_output_image;
            cout << "Enter the angle for rotation (e.g., 90): ";
            cin >> angle;
            cout << "Processing input image with Color Space Conversion...\n\n";
            break;
        }
    }
}

void applyImageProcessing(Mat& local_mat, Mat& result, int& choice, int& radius, int& blockSize, double& low_threshold, double& high_threshold, int& type, int& code, double& angle) {
    
    switch (choice) {
        case 1:
            applyGaussianBlur(local_mat, radius, result);
            break;
        case 2:
            applyMedianBlur(local_mat, radius, result);
            break;
        case 3:
            applyEdgeDetection(local_mat, low_threshold, high_threshold, result);
            break;
        case 4:
            applyLocalThreshold(local_mat, blockSize, result);
            break;
        case 5:
            applyGlobalThreshold(local_mat, high_threshold, result);
            break;
        case 6:
            applyColorSpaceConvertion(local_mat, result, code);
            break;
        case 7:
            applyColorMap(local_mat, result, type);
            break;
        case 8:
            applyrotation(local_mat, result, angle);
            break;
        default:
            cerr << "Invalid choice!" << endl;
            break;
    }
}