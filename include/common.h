#pragma once



#include <opencv2/opencv.hpp>

#include <opencv2/core/version.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>



#include <Eigen/Core>

using namespace Eigen;

typedef Matrix<short,Dynamic,1> VectorXs;



VectorXs getLabeling( const unsigned char * im, int N, int M );

unsigned char * colorize( const VectorXs & labeling, int W, int H );





//////////////////////////////////////////////////////////////////////////

int getColor4Depth( const unsigned char * c );

void putColor4Depth( unsigned char * c, int cc );

unsigned char * colorize4Depth( const VectorXs & labeling, int W, int H );

VectorXs getLabelingDepth(const unsigned char* im, int N, int M);



cv::Mat colorize4Parsing( const VectorXs & labeling, int W, int H );
