
#include <stdio.h>
#include <iostream>
#include<fstream>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "gMF_Lib/gMF.h"
#include "NVTimer.h"
#include "image_helper.h"

using namespace std;
using namespace cv;


bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int main(int argc, char** argv){
    
    if(argc!=4){
        cout<<"Usage: "<< argv[0] <<" <Image folder> <Mask folder> <Output folder>"<<endl;
        return -1;
    }
    
    std::string image_dir = argv[1];
    std::string prob_dir = argv[2];
    std::string output_dir = argv[3];
    
    if (!boost::filesystem::exists(image_dir) || !boost::filesystem::exists(prob_dir))
    {
        std::cerr << "image or prob dir does not exist!";
        return -1;
    }
    
    if (!boost::filesystem::exists(output_dir))
        boost::filesystem::create_directories(output_dir);
    
    std::vector<std::string> image_name_list;
    std::vector<std::string> prob_name_list;
    
    for (boost::filesystem::directory_iterator entry(prob_dir); entry != boost::filesystem::directory_iterator(); ++entry )
     {
         std::string prob_file_name = entry->path().filename().string();
         
         if (hasEnding(prob_file_name,"png"))
         {
             string jpg_name = prob_file_name;
             jpg_name[jpg_name.length()-3] = 'j';
             jpg_name[jpg_name.length()-2] = 'p';
             jpg_name[jpg_name.length()-1] = 'g';
            
             prob_name_list.push_back(prob_file_name);
             image_name_list.push_back(jpg_name);
             
//             std::cout << prob_file_name + "----" + jpg_name <<std::endl;
         }
         
     }
    
    if (image_name_list.size()==0 || prob_name_list.size()==0)
    {
        std::cerr << "fail to load images" << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> probs;
    
    for (size_t i=0;i<image_name_list.size(); i++)
    {
        cv::Mat frame = cv::imread(image_dir+ "/" + image_name_list[i],cv::IMREAD_COLOR);
        cv::Mat prob = cv::imread(prob_dir+ "/" + prob_name_list[i],cv::IMREAD_GRAYSCALE);
        
        cv::resize(frame,frame,cv::Size(frame.cols/2, frame.rows/2));
        cv::resize(prob,prob,frame.size());
        
        frames.push_back(frame);
        probs.push_back(prob);
    }
    
    
    //---------------   there are the parameters that you can play with --------------------------------------------------
    const int M = 2;                                                                       // number of lables
    const float sigma_BF_xy = 30;                                             // std of spatial kernel in bilateral filter
    const float sigma_BF_rgb = 13  ;                                             // std of range kernel in bilateral filter
    const float sigma_GF_xy = 3;                                               // std of Gaussian filter
	  const float weight_gaussian = 5.0;                                    // weight of gaussian filter
    const float weight_bilateralfilter = 5.0;                        // weight of bilateral filter
    const int no_iterations = 5;                                                  // number of interations
    
    
    cout << "--------- running gMF with following configuration: ---------"<<endl;
    cout <<"M="<<M<<endl;
    cout <<"sigma_BF_xy ="<<sigma_BF_xy<<endl;
    cout <<"sigma_BF_rgb = "<<sigma_BF_rgb<<endl;
    cout <<"sigma_GF_xy = "<<sigma_GF_xy<<endl;
    cout <<"weight_gaussian = "<<weight_gaussian<<endl;
    cout <<"weight_bilateralfilter = "<<weight_bilateralfilter<<endl;
    cout <<"no_iterations = "<<no_iterations<<endl;
    cout << "-----------------------------------------------------------------------------------"<<endl;
    //---------------------------------------------------------------------------------------------------------------------------------------------
    
    
    cv::Mat & sample_frame = frames[0];
    int W = sample_frame.cols;
    int H = sample_frame.rows;
    
    gMF::inference_engine *my_CRF = new gMF::inference_engine(W,H,M);
    gMF::BF_info *my_BF_info = new gMF::BF_info(sigma_BF_xy, sigma_BF_rgb);
    gMF::GF_info *my_GF_info = new gMF::GF_info(sigma_GF_xy);

    float *unary_data = new float[W*H*M];
    float *Q_dist_data = new float[W*H*M];
        
    float *pott_model_data = new float[M*M];
    create_pott_compatibility_func(pott_model_data,M);
    my_CRF->load_compatibility_function(pott_model_data);
    
    cv::Mat out_img(H,W,CV_8UC1); 
    cv::Mat tmp_img(H,W,CV_8UC1); 
    
    Mat erode_kernel = getStructuringElement(cv::MORPH_OPEN,cv::Size(3, 3),cv::Point(1, 1) );
    
    StopWatchInterface *my_timer;
    sdkCreateTimer(&my_timer);
    
    for (size_t i=0;i<frames.size();i++)
    {
        sdkResetTimer(&my_timer);
        sdkStartTimer(&my_timer);
        
        grayscale_to_binary_unary(unary_data,probs[i],W,H);    
        
        my_CRF->load_reference_image(frames[i].data, W, H);    
        my_CRF->load_unary_potential(unary_data);
        my_CRF->exp_and_normalize();
    
        for (int iter = 0; iter < no_iterations; iter++) {
    
            my_CRF->filter_bilateral(weight_bilateralfilter, M, W, H, my_BF_info, false);
            my_CRF->filter_gaussian(weight_gaussian,M,W,H,my_GF_info,true);
            my_CRF->apply_compatibility_transform();
            my_CRF->substract_update_from_unary_potential();
            my_CRF->exp_and_normalize();
    
            cudaThreadSynchronize();
           
        } 
        my_CRF->get_Q_distribution(Q_dist_data); 
        
        binary_Q_to_gray(out_img,Q_dist_data,W,H);
              
        cv::erode(out_img,tmp_img,erode_kernel,cv::Point(-1,-1),2);
        cv::floodFill(tmp_img,cv::Point(0,0),100);

        for (int y=0;y<out_img.rows;y++)
            for (int x=0;x<out_img.cols;x++)
            {
                if (tmp_img.at<uchar>(y,x)==100) out_img.at<uchar>(y,x) = 0;
                else out_img.at<uchar>(y,x) = 255;
            }
      
      cv::GaussianBlur(out_img, out_img,cv::Size(5,5),0);
      cv::GaussianBlur(out_img, out_img,cv::Size(5,5),0);
      cv::imwrite(output_dir + "/" + image_name_list[i] , out_img);  
      sdkStopTimer(&my_timer); 
      
      
      cv::imshow("result", out_img);
      cv::waitKey();
      printf("total time per image :[%.2f]ms\n", sdkGetTimerValue(&my_timer));
      std::cout << std::flush;
    }
    

    delete my_CRF;
    delete[] unary_data;
    delete[] Q_dist_data;
    delete[] pott_model_data;
    delete my_timer;
    
	return 0;
 
    }
