
#include <Eigen/Core>

#include <time.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "gMF_Lib/gMF.h"
#include "NVTimer.h"


#include "permutohedral.h"
#include <cstdio>
#include <iostream>
#include <cmath>
#include "common.h"
#include "Kernel.h"



using namespace std;
using namespace cv;



template <class T>
inline T square(const T &x) { return x*x; };
//short round(float d)
//{
//	return (short) floor(d + 0.5);
//}
//////////////////////////////////////////////////////////////////////////
//void putColor( unsigned char * c, int cc ){
//	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
//}
//// Produce a color image from a bunch of labels
//unsigned char * colorize( const VectorXs & labeling, int W, int H ){
//	unsigned char * r = new unsigned char[ W*H*3 ];
//	for( int k=0; k<W*H; k++ ){
//		int c = colors[ labeling[k] ];
//		putColor( r+3*k, c );
//	}
//	return r;
//}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
// from cost to probability
MatrixXf computeUnary( const VectorXs & lbl, int M ){
	const float u_energy = -log( 1.0 / M );
	const float n_energy = -log( (1.0 - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	MatrixXf r( M, lbl.rows() );
	r.fill(u_energy);
	//printf("%d %d %d \n",im[0],im[1],im[2]);
	for( int k=0; k<lbl.rows(); k++ ){
		// Set the energy
		if (lbl[k]>=0){
			r.col(k).fill( n_energy );
			r(lbl[k],k) = p_energy;
		}
	}
	return r;
}


MatrixXf computeUnary4Depth(const VectorXs &lbl, int M){
	float LAMBDA = 0.05F;
	float DATA_K =  10000.0F;

	MatrixXf r( M, lbl.rows() );

	for (int i = 0; i<lbl.rows(); i++){
		for (int valueM = 0; valueM < M; valueM++)
		{
			float val = square((float)(lbl(i)-valueM));
			r(valueM,i) = LAMBDA*std::min(val, DATA_K);
		}
	}
	return r;
}

void expAndNormalize(MatrixXf & out, const MatrixXf & in)
{
	out.resize( in.rows(), in.cols() );
	for( int i=0; i<out.cols(); i++ ){
		VectorXf b = in.col(i);
		b.array() -= b.maxCoeff();
		b = b.array().exp();
		out.col(i) = b / b.array().sum();
	}
}


void spatialfeaturekernel(float sx, float sy, int H_, int W_, MatrixXf &feature)
{
	int N_ = H_*W_;
	//assert(feature(2, numberofpixels);
	for (int j =0 ;j < H_; j++)
		for (int i = 0; i < W_; i++)
		{
			feature(0,j*W_+i) = i/sx;
			feature(1,j*W_+i) = j/sy;
		}
}
void bilateralfeaturekernel(unsigned char *im, float sx, float sy, float sr, float sg, float sb,int H_, int W_, MatrixXf &feature)
{
	int N_ = H_*W_;
	//assert(feature( 5, N_ );
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature(0,j*W_+i) = i / sx;
			feature(1,j*W_+i) = j / sy;
			feature(2,j*W_+i) = im[(i+j*W_)*3+0] / sr;
			feature(3,j*W_+i) = im[(i+j*W_)*3+1] / sg;
			feature(4,j*W_+i) = im[(i+j*W_)*3+2] / sb;
		}
}
//////////////////////////////////////////////////////////////////////////

void createPottCompatibilityFunction(int M, MatrixXf &Pott){
	Pott=-MatrixXf::Identity(M,M);
}

//////////////////////////////////////////////////////////////////////////
void createTruncatedCompatibilityFunction(int M, MatrixXf &trunct, float threshold_){
	//trunct = VectorXf::Zero(M).asDiagonal();
	
	for (int i = 0; i < M; i++)	
		for (int j = 0; j < M; j++){
			//if (i == j)
			//	continue;
			trunct(i,j) = std::min(threshold_,(float)std::abs(i-j));
		}
}

//////////////////////////////////////////////////////////////////////////
VectorXs currentMapDepth( const MatrixXf & Q ,int M){
	MatrixXf magicVector = MatrixXf::Zero(1,M);
	for (int i = 0; i < M; i++)
	{
		magicVector(0,i) = (float)i;
	}

	MatrixXf rr = magicVector*Q;;
	VectorXs r(Q.cols());
	for (int i = 0; i < rr.cols(); i++){
		r[i] = round(rr(0,i));
		//std::cout<<r[i]<<"\t"<<std::endl;
	}


	return r;
}

//////////////////////////////////////////////////////////////////////////
VectorXs currentMap( const MatrixXf & Q ){
	VectorXs r(Q.cols());
	// Find the map
	for( int i=0; i<Q.cols(); i++ ){
		int m;
		Q.col(i).maxCoeff( &m );
		r[i] = m;
	}
	return r;
}

void float_to_cv_mat(const float* inimg, Mat& outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg.at<Vec3b>(y, x)[0] = inimg[idx];
			outimg.at<Vec3b>(y, x)[1] = inimg[idx + 1];
			outimg.at<Vec3b>(y, x)[2] = inimg[idx + 2];
		}
}

void float_to_cv_mat_with_offset(const float* inimg, Mat& outimg, int w, int h, int dim, int offeset)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg.at<Vec3b>(y, x)[0] = inimg[idx + offeset*3];
			outimg.at<Vec3b>(y, x)[1] = inimg[idx + offeset * 3 + 1];
			outimg.at<Vec3b>(y, x)[2] = inimg[idx + offeset * 3 + 2];
		}
}


void cv_mat_to_float(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg[idx] = inimg.at<Vec3b>(y, x)[0];
			outimg[idx + 1] = inimg.at<Vec3b>(y, x)[1];
			outimg[idx + 2] = inimg.at<Vec3b>(y, x)[2];
		}
}

void cv_mat_to_float_duplicate(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	int dup_times = dim / 3;

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			for (int k = 0; k < dup_times;k++)
			{
				outimg[idx + k*3] = inimg.at<Vec3b>(y, x)[0];
				outimg[idx + k * 3 + 1] = inimg.at<Vec3b>(y, x)[1];
				outimg[idx + +k * 3 + 2] = inimg.at<Vec3b>(y, x)[2];
			}
		}
}


void cv_mat_to_distribution(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			const Vec3b& pix = inimg.at<Vec3b>(y, x);
			float sum_val = pix[0] + pix[1] + pix[2] + (10e-3 * 3);
			outimg[idx] = ((float)pix[0] + 25 + 10e-3) / sum_val;
			outimg[idx + 1] = ((float)pix[1] + 20 + 10e-3) / sum_val;
			outimg[idx + 2] = ((float)pix[2] + 10e-3) / sum_val;
		}
}

void distribution_to_cv_mat(const float* inimg, Mat& outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;

			if (inimg[idx + 1] > inimg[idx])
			{
				
				if (inimg[idx + 2] > inimg[idx + 1]) outimg.at<Vec3b>(y, x) = Vec3b(255, 0, 0); // b
				else outimg.at<Vec3b>(y, x) = Vec3b(0, 255, 0); // g
			}
			else
			{
				if (inimg[idx + 2] > inimg[idx]) outimg.at<Vec3b>(y, x) = Vec3b(255, 0, 0); // b
				else outimg.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // r
			}
		}
}


MatrixXf float_to_Matrixxf(const float* in_array, int number_of_channels, int Height, int Width) {
	MatrixXf outK(number_of_channels, Width*Height);

	for(int nchannels = 0; nchannels < number_of_channels; nchannels++){
		for (int iy = 0; iy < Height; iy++){
			for (int ix = 0; ix < Width; ix++){
				outK(nchannels, iy*Width+ix) = in_array[(iy*Width+ix)*number_of_channels+nchannels];
			}
		}
	}
	return outK;
}

void Matrixxf_to_float(const MatrixXf& in_mat, float* out_array, int number_of_channels, int Height, int Width){

	for (int nchannels = 0; nchannels < number_of_channels; nchannels++){
		for (int iy = 0; iy < Height; iy++){
			for (int ix = 0; ix < Width; ix++){
				out_array[(iy*Width + ix)*number_of_channels + nchannels] = in_mat(nchannels, iy*Width + ix);
			}
		}
	}

}

int main(int argc, char** argv){
	  //if (argc<4 || argc >5){
    //	printf("Usage: %s image annotations output groundtruth[option]\n", argv[0] );
    //	return 1;
    //}
	// Number of labels
	const int M = 21;	
	// Load the color image and some crude annotations (which are used in a simple classifier)
	int W, H, nChns, GW, GH, GnChns, GGW, GGH, GGnChns;
    //std::string imagepath = argv[1];

    std::string imagepath = argv[1];//    std::string imagepath = "/home/carl/Work/Code/gmeanfild/data/im1.ppm";
    std::string responsepath = argv[2];//    std::string responsepath="/home/carl/Work/Code/gmeanfild/data/anno1.ppm";
    std::string outputpath = argv[3];//    std::string outputpath = "/home/carl/Work/Code/gmeanfild/data/im_mf1.png";

//    std::string imagepath = "/home/carl/Work/Code/gmeanfild/data/img.jpg";
//    std::string responsepath="/home/carl/Work/Code/gmeanfild/data/voc2011_anno.png";
//    std::string outputpath = "/home/carl/Work/Code/gmeanfild/data/voc2011_mf.png";

	cv::Mat imMat = cv::imread(imagepath,1);
	//cv::resize(imMat, imMat, cv::Size(512, 512));

	W = imMat.cols;
	H = imMat.rows;
	nChns = imMat.channels();
	unsigned char * im = new unsigned char[W*H*nChns];
	for (int i = 0; i < W*H*nChns; i++)
	{
		im[i] = imMat.data[i];
	}
	if (!im){
		printf("Failed to load image!\n");
		return 1;
	}
	//std::string responsepath=argv[2];


	cv::Mat annoMat = cv::imread(responsepath,1);
	//cv::resize(annoMat, annoMat, cv::Size(512, 512));

	GW = annoMat.cols;
	GH = annoMat.rows;
	GnChns = annoMat.channels();
	unsigned char * anno = new unsigned char[GW*GH*GnChns];
	for (int i = 0; i < GW*GH*nChns;i++)
	{
		anno[i] = annoMat.data[i]+1;
	}

	if (!anno){
		printf("Failed to load coarse prediction map!\n");
		return 1;
	}
	if (W!=GW || H!=GH){
        std::cout<<W<<" ";
        std::cout<<H<<std::endl;
        std::cout<<GW<<" ";
        std::cout<<GH<<std::endl;

		printf("Coarse prediction map size doesn't match image!\n");
		return 1;
	}
	int N_ = W*H;//number of pixels	
	/////////// Put your own unary classifier here! ///////////
	VectorXs coarse_labeling_results = getLabeling(anno,W*H,M);
	//////////////////////////////////////////////////////////////////////////
	//compute unary accuracy
	
	//std::string gtpath="C:\\Src\\ConvMean\\trunk\\src\\build\\src\\Debug\\gt\\bathroom_0003_46.png";	
	VectorXs gt_labeling = VectorXs::Zero(GW*GH);
	if (argc > 4){
		std::string gtpath=argv[4];
		cv::Mat gtMat = cv::imread(gtpath,1);
		GGW = gtMat.cols;
		GGH = gtMat.rows;
		GGnChns = gtMat.channels();
		unsigned char * gt = new unsigned char[GGW*GGH*GGnChns];
		for (int i = 0; i < GGW*GGH*GGnChns; i++){
			gt[i] = gtMat.data[i];
		}
		if (!gt){
			printf("Failed to load annotations!\n");
			return 1;
		}
		gt_labeling = getLabelingDepth(gt, GGW*GGH,M);
		

		float *sameind = new float[W*H];
		float meanaccuracy = 0.0F;
		for (int i = 0; i < W*H; i++)
		{
			if (coarse_labeling_results[i]==gt_labeling[i]){
				sameind[i] = 1;
			}else{
				sameind[i] = 0;
			}
			meanaccuracy += sameind[i];
		}
		meanaccuracy = meanaccuracy/((float)(W*H));
		std::cout<<meanaccuracy<<std::endl;

		delete sameind;
	}
	MatrixXf unary = computeUnary(coarse_labeling_results,M);
	//////////////////////////////////////////////////////////////////////////
	// Initialize Q_i <- 1/Z_i (exp(-\phi_u (x_i) for all i
	MatrixXf Q(M,N_);
	MatrixXf tmp0, tmp1, tmp2;
	MatrixXf unary_=unary;

	expAndNormalize(Q, -unary);
	//////////////////////////////////////////////////////////////////////////
	//prepare filters
	float sx_1 = 3.0;
	float sy_1 = 3.0;
	MatrixXf skernel(2, H*W);
	spatialfeaturekernel(sx_1, sy_1, H,  W,skernel);
	DenseKernel skernelfilter(skernel);
	// weight parameters for gaussian kernel
	float weight_gaussian = 3.0;
    float sx = 40.0;
    float sy = 40.0;
    float sr =  5.0;
    float sg = 5.0;
    float sb = 5.0;
	MatrixXf bkernel(5,H*W);
	bilateralfeaturekernel(im,  sx, sy, sr, sg, sb, H,  W,bkernel);
	DenseKernel bkernelfilter(bkernel);
	// weight parameters for bilateral filter
    float weight_bilateralfilter = 10.0;

	MatrixXf compatibilityFunction(M,M);
	createPottCompatibilityFunction(M,compatibilityFunction);
	int n_iterations = 5;
	//////////////////////////////////////////////////////////////////////////
	//
	float *in_array = new float[W * H * M];
	float *out_array= new float[W * H * M];
	cv::Mat frame, frame_in, frame_out;
	frame_in.create(H, W, CV_8UC3);
	frame_out.create(H, W, CV_8UC3);
	cv::resize(imMat, frame, cv::Size(W,H));


    gMF::inference_engine *my_CRF = new gMF::inference_engine(W,H,M);

	gMF::BF_info *my_BF_info = new gMF::BF_info(sx, sr);
    gMF::BF_info *my_GF_info = new gMF::BF_info(3, 10000);
    //gMF::GF_info *my_GF_info = new gMF::GF_info(sx_1);

    my_CRF->load_reference_image(frame.data, W, H);

	//////////////////////////////////////////////////////////////////////////
	StopWatchInterface *my_timer;
	sdkCreateTimer(&my_timer);
	//sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);

    MatrixXf tmpUnary = -unary;
    float *unary_array = new float[W*H*M];
    float *cmpt_array = new float[M*M];


    Matrixxf_to_float(compatibilityFunction,cmpt_array,M,M,1);
    my_CRF->load_compatibility_function(cmpt_array);

    Matrixxf_to_float(tmpUnary,unary_array, M, H, W);
    my_CRF->load_unary_potential(unary_array);
    my_CRF->exp_and_normalize();

    n_iterations = 5;

    for (int iter = 0; iter < n_iterations; iter++) {
        sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);

        my_CRF->filter_bilateral(weight_bilateralfilter, M, W, H, my_BF_info, false);
        my_CRF->filter_bilateral(weight_gaussian, M, W, H, my_GF_info, true);
        my_CRF->apply_compatibility_transform();
        my_CRF->substract_update_from_unary_potential();
        my_CRF->exp_and_normalize();

        cudaThreadSynchronize();
        sdkStopTimer(&my_timer); printf("iteration in:[%.2f]ms\n", sdkGetTimerValue(&my_timer));
    }

    my_CRF->get_Q_distribution(out_array);
    Q = float_to_Matrixxf(out_array,M,H,W);

   // expAndNormalize(Q,tmpUnary);

	//sdkStopTimer(&my_timer); printf("total inference time:[%.2f]ms\n", sdkGetTimerValue(&my_timer));

	VectorXs resultmap = currentMap(Q);
	if (argc > 4){
		float *sameid_res = new float[W*H];
		float meanaccuracy_res = 0.0F;
		for (int i = 0; i < W*H; i++)
		{
			if (resultmap[i]==gt_labeling[i]){
				sameid_res[i] = 1;
			}else {
				sameid_res[i] = 0;
			}
			meanaccuracy_res += sameid_res[i];
		}
		meanaccuracy_res = meanaccuracy_res/((float)(W*H));
		std::cout<<meanaccuracy_res<<std::endl;

		delete sameid_res;
	}
	cv::Mat resMat = colorize4Parsing(resultmap,W,H);
	//For visualization
	//std::string outputpath=argv[3];




	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

    cv::imwrite(outputpath,resMat);

    delete my_CRF;

	delete in_array;
	delete out_array;

	delete im;
	return 0;
}
