//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core/utils/logger.hpp>
//#include <opencv2/core.hpp>
//#include "opencv2/imgproc.hpp"
//
//#include <iostream>
//#include <vector>
//#include "mpi.h"
//
//using namespace std;
//using namespace cv;
//
//void Image_Rotation() {
//    MPI_Init(NULL, NULL);
//
//    // Get the number of processes and the rank of the current process
//    int world_size, world_rank;
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//
//    // Load the image on root process
//    cv::Mat image;
//    if (world_rank == 0) {
//        char imagePath[] = "C:/Users/islam/Downloads/1.jpg";
//        image = cv::imread(imagePath, cv::IMREAD_COLOR);
//        if (image.empty()) {
//            cerr << "Error: Failed to load image" << endl;
//            MPI_Abort(MPI_COMM_WORLD, 1);
//        }
//    }
//  
//    // Broadcast image dimensions
//    int rows, cols;
//    if (world_rank == 0) {
//        rows = image.rows;
//        cols = image.cols;
//    }
//    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//    // Calculate portion of image to process
//    int chunkSize = rows / world_size;
//    int startRow = world_rank * chunkSize;
//    int endRow = (world_rank == world_size - 1) ? rows : (world_rank + 1) * chunkSize;
//
//    // Allocate memory for local portion of image
//    Mat localImage(endRow - startRow, cols, CV_8UC3);
//
//    // Scatter the image data
//    MPI_Scatter(image.data, (chunkSize * cols * image.channels()), MPI_CHAR,
//        localImage.data, (chunkSize * cols * image.channels()), MPI_CHAR, 0, MPI_COMM_WORLD);
//
//    // Rotate local portion of the image
//    Mat rotatedImage;
//    Point2f center(cols / 2.0, (startRow + endRow) / 2.0); // Rotation center
//    double angle = 45.0; // Rotation angle in degrees (adjust as needed)
//    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0); // Get rotation matrix
//    warpAffine(localImage, rotatedImage, rotationMatrix, localImage.size()); // Apply rotation
//
//    // Gather rotated image results
//    Mat allRotatedImage;
//    if (world_rank == 0) {
//        allRotatedImage.create(rows, cols, CV_8UC3);
//    }
//    MPI_Gather(rotatedImage.data, (chunkSize * cols * image.channels()), MPI_CHAR,
//        allRotatedImage.data, (chunkSize * cols * image.channels()), MPI_CHAR, 0, MPI_COMM_WORLD);
//
//    // Output or further process the rotated image on root process
//    if (world_rank == 0) {
//        // Save the thresholded image
//      
//        const char* outputPath = "C:/Users/islam/Downloads/parallel_output/";
//        if (!cv::imwrite("C:/Users/islam/Downloads/parallel_output/", allRotatedImage)) {
//            std::cerr << "Error: Failed to save the Rotated image" << std::endl;
//        }
//        else {
//            std::cout << "Rotated image saved to: " << outputPath << std::endl;
//            imshow("Local Rotated image", allRotatedImage);
//            waitKey(0);
//        }
//
//    }
//
//    // Finalize the MPI environment
//    MPI_Finalize();
//}
//void main() {
//    Image_Rotation();
//}