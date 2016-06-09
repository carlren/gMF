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

std::vector <cv::Mat3b> load_video(string input_filename)
{
    std::cerr << "Loading video '" << input_filename << "'\n";
    cv::VideoCapture video_capture (input_filename);
    std::vector <cv::Mat3b> frames;

    //size_t i = 0;
    while (video_capture.grab ())
    {
        cv::Mat3b frame;
        video_capture.retrieve (frame);
        frames.push_back (frame.clone ());        
    }
    if (frames.empty ())
    {
        throw std::runtime_error ("The input frames are empty");
    }
    std::cerr << "Loaded " << frames.size () << " frames." << std::endl;

    return (frames);
}

int main(int argc, char** argv){

    if(argc<3){
        cout<<"Usage: ./bSeg <input video> <mask video>"<<endl;
        return -1;
    }

    //---------------   there are the parameters that you can play with --------------------------------------------------
    const int M = 2; // number of lable
    const float sigma_BF_xy = 30; // std of spatial kernel in bilateral filter
    const float sigma_BF_rgb = 5; // std of range kernel in bilateral filter
    const float sigma_GF_xy = 3; // std of Gaussian filter
	  const float weight_gaussian = 5.0; // weight of gaussian filter
    const float weight_bilateralfilter = 10.0; // weight of bilateral filter
    const int no_iterations = 5;// number of interations
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


    std::string video_path = argv[1];
    std::string anno_path = argv[2];
    std::vector <cv::Mat3b> all_frames = load_video(video_path);
    std::vector <cv::Mat3b> all_masks = load_video(anno_path);
    
    int W, H;
    W = all_frames[0].cols;
    H = all_frames[0].rows;
    cv::Size tsize; tsize.width = W; tsize.height = H;

    int *labeling_data = new int[W*H];
    float *unary_data = new float[W*H*M];
    float *Q_dist_data = new float[W*H*M];
    float *pott_model_data = new float[M*M];

    bool show_frames = true;
    bool need_refresh = true;

    cv::Mat ori_frame;
    cv::Mat seg_frame; seg_frame.create(H,W,CV_8UC3);
    cv::Mat in_anno, tmp_anno;

    cv::cvtColor(all_masks[0],tmp_anno,cv::COLOR_BGR2GRAY);
    cv::threshold(tmp_anno,tmp_anno,100,255,cv::THRESH_BINARY);
    cv::cvtColor(tmp_anno,in_anno,CV_GRAY2BGR);
    
    read_labling_from_image(labeling_data, in_anno,W,H,M);
    labeling_to_unary(unary_data,labeling_data,W,H,M);
    create_pott_compatibility_func(pott_model_data,M);
    StopWatchInterface *my_timer;  sdkCreateTimer(&my_timer);

    gMF::inference_engine *my_CRF = new gMF::inference_engine(W,H,M);
	  gMF::BF_info *my_BF_info = new gMF::BF_info(sigma_BF_xy, sigma_BF_rgb);
    gMF::GF_info *my_GF_info = new gMF::GF_info(sigma_GF_xy);
    my_CRF->load_compatibility_function(pott_model_data);

    cv::Size vid_size;vid_size.width = 2*W; vid_size.height = H;
    cv::Mat vid_frame; vid_frame.create(vid_size,CV_8UC3);

    int frame_id=0;
    my_CRF->load_unary_potential(unary_data);
    
    while(show_frames)
    {
        if(need_refresh)
        {
            cv::Mat tmp_frame;
            cv::resize(all_frames[frame_id],tmp_frame,tsize);
            tmp_frame.copyTo(ori_frame);
            need_refresh = false;

            sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
            
            cv::cvtColor(all_masks[frame_id],tmp_anno,cv::COLOR_BGR2GRAY);
            cv::threshold(tmp_anno,tmp_anno,100, 255,cv::THRESH_BINARY);
            cv::cvtColor(tmp_anno,in_anno,CV_GRAY2BGR);
            
            cv::imshow("mask",in_anno);
            read_labling_from_image(labeling_data, in_anno,W,H,M);
            labeling_to_unary(unary_data,labeling_data,W,H,M);
            my_CRF->load_unary_potential(unary_data);
            
            my_CRF->exp_and_normalize();
            my_CRF->load_reference_image(ori_frame.data, W, H);
            for (int i=0;i<5;i++){
                my_CRF->filter_bilateral(weight_bilateralfilter, M, W, H, my_BF_info, false);
                my_CRF->filter_gaussian(weight_gaussian,M,W,H,my_GF_info,true);
                my_CRF->apply_compatibility_transform();
                my_CRF->substract_update_from_unary_potential();
                my_CRF->exp_and_normalize();
            }
            cudaThreadSynchronize();
            sdkStopTimer(&my_timer); printf("processed in:[%.2f]ms\n", sdkGetTimerValue(&my_timer)); cout<<flush;

            my_CRF->get_Q_distribution(Q_dist_data);
            Q_dist_to_labeling(labeling_data,Q_dist_data,W,H,M);
            draw_image_from_labeling(seg_frame,labeling_data,W,H);

            ori_frame.copyTo(vid_frame(Range::all(),Range(0,W)));
            seg_frame.copyTo(vid_frame(Range::all(),Range(W,2*W)));

            //vw<<vid_frame;
        }

        cv::imshow("segmentation",vid_frame);


        char key = cv::waitKey(10);

        if (key == 'x')
        {
            ++frame_id;
            if (frame_id >= all_frames.size ()) frame_id = all_frames.size ()-1;
            need_refresh = true;
        }
        if (key == 'z')
        {
            --frame_id;
            if (frame_id < 0) frame_id = 0;
            need_refresh = true;
        }
        if (key == 'q' )
            show_frames = false;
    }



	return 0;

    }
