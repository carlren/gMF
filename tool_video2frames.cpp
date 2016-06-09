#include <string>
#include "opencv2/opencv.hpp"

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
    
    if(argc<4){
        cout<<"Usage: ./Tool_video2frames <input video> <output dir>"<<endl;
        return -1;
    }
    
    std::string video_path = argv[1];
    std::string output_path = argv[2];
    int begin_idx = std::atoi(argv[3]);
    
    std::vector <cv::Mat3b> all_frames = load_video(video_path);
    
    for (size_t i=begin_idx;i<all_frames.size();i++){
        
        char out_name[200];
        sprintf(out_name,"%s/%04i.png",output_path.c_str(),i-begin_idx);
        cv::imwrite(out_name, all_frames[i]);
        
    }
    
    
    
}
