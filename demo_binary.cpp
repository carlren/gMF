
#include <stdio.h>
#include <iostream>
#include<fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "gMF_Lib/gMF.h"
#include "NVTimer.h"

#include "image_helper.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    if(argc<4){
        cout<<"Usage: ./gMF_binary <input image> <input unary> <output image>"<<endl;
        return -1;
    }
    
    //---------------   there are the parameters that you can play with --------------------------------------------------
    const int M = 2;                                                                       // number of lables
    const float sigma_BF_xy = 30;                                             // std of spatial kernel in bilateral filter
    const float sigma_BF_rgb = 10;                                             // std of range kernel in bilateral filter
    const float sigma_GF_xy = 3;                                               // std of Gaussian filter
	const float weight_gaussian = 3.0;                                    // weight of gaussian filter
    const float weight_bilateralfilter = 10.0;                        // weight of bilateral filter
    const int no_iterations = 5;                                                  // number of interations
    //---------------------------------------------------------------------------------------------------------------------------------------------
    
    
    cout << "--------- running gMF with following configuration: ---------"<<endl;
    cout <<"M="<<M<<endl;
    cout <<"sigma_BF_xy ="<<sigma_BF_xy<<endl;
    cout <<"sigma_BF_rgb = "<<sigma_BF_rgb<<endl;
    cout <<"sigma_GF_xy = "<<sigma_GF_xy<<endl;
    cout <<"weight_gaussian = "<<weight_gaussian<<endl;
    cout <<"weight_bilateralfilter = "<<weight_bilateralfilter<<endl;
    cout <<"no_iterations = "<<no_iterations<<endl;
    cout << "-----------------------------------------------------------------------------------"<<endl;
    
    int W, H, tmpW, tmpH;
    bool need_resize=false;
    std::string image_path = argv[1];
    std::string anno_path = argv[2];
    std::string output_path = argv[3];
   
    cv::Mat in_img = cv::imread(image_path,1);
    cv::Mat in_anno = cv::imread(anno_path,cv::IMREAD_GRAYSCALE);
    
    W = in_img.cols;
    H = in_img.rows;
    
    if (W<256&&H<256)
    {
        need_resize = true;
        cv::Mat tmp_img; in_img.copyTo(tmp_img);
        cv::Mat tmp_anno; in_anno.copyTo(tmp_anno);
        
        tmpW = W; tmpH = H;
         W = 256; H =256;
        cv::resize(tmp_img,in_img,Size(W,H));
        cv::resize(tmp_anno,in_anno,Size(W,H));
    }
    
    
    float *unary_data = new float[W*H*M];
    float *Q_dist_data = new float[W*H*M];
    float *pott_model_data = new float[M*M];
    
    
    grayscale_to_binary_unary(unary_data,in_anno,W,H);
    create_pott_compatibility_func(pott_model_data,M);
    StopWatchInterface *my_timer;

    gMF::inference_engine *my_CRF = new gMF::inference_engine(W,H,M);
	gMF::BF_info *my_BF_info = new gMF::BF_info(sigma_BF_xy, sigma_BF_rgb);
    gMF::GF_info *my_GF_info = new gMF::GF_info(sigma_GF_xy);

    my_CRF->load_reference_image(in_img.data, W, H);
    my_CRF->load_compatibility_function(pott_model_data);
    my_CRF->load_unary_potential(unary_data);
    my_CRF->exp_and_normalize();

    sdkCreateTimer(&my_timer);
    for (int iter = 0; iter < no_iterations; iter++) {
        sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);

        my_CRF->filter_bilateral(weight_bilateralfilter, M, W, H, my_BF_info, false);
        my_CRF->filter_gaussian(weight_gaussian,M,W,H,my_GF_info,true);
        my_CRF->apply_compatibility_transform();
        my_CRF->substract_update_from_unary_potential();
        my_CRF->exp_and_normalize();

        cudaThreadSynchronize();
        sdkStopTimer(&my_timer); printf("iteration in:[%.2f]ms\n", sdkGetTimerValue(&my_timer));
    } 
    my_CRF->get_Q_distribution(Q_dist_data); 
    
    cv::Mat out_img; out_img.create(H,W,CV_8UC3); 
   binary_Q_to_rgb(out_img,Q_dist_data,W,H);
    
    if(need_resize)
    {
        cv::Mat tmp_img; out_img.copyTo(tmp_img);
        cv::resize(tmp_img, out_img,Size(tmpW,tmpH),0,0,INTER_NEAREST);
    }
    
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
    cv::imwrite(output_path,out_img);

    delete my_CRF;
    delete unary_data;
    delete Q_dist_data;
    delete pott_model_data;
    delete my_timer;
    
	return 0;
 
    }
