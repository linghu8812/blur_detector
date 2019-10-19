#ifndef BLUR_DETECTOR_BLUR_DETECTOR_H
#define BLUR_DETECTOR_BLUR_DETECTOR_H

#include <opencv2/opencv.hpp>

class BlurDetector
{
public:
    bool check_image_size(cv::Mat &image, int block_size=8);
    float get_blurness(cv::Mat image, int block_size=8);
private:
    float dct_threshold = 8.0;
    float max_hist = 0.1;
    float hist_weight[8][8] =
            {
                    {8, 7, 6, 5, 4, 3, 2, 1},
                    {7, 8, 7, 6, 5, 4, 3, 2},
                    {6, 7, 8, 7, 6, 5, 4, 3},
                    {5, 6, 7, 8, 7, 6, 5, 4},
                    {4, 5, 6, 7, 8, 7, 6, 5},
                    {3, 4, 5, 6, 7, 8, 7, 6},
                    {2, 3, 4, 5, 6, 7, 8, 7},
                    {1, 2, 3, 4, 5, 6, 7, 8}
            };
    float weight_total = 344.0;
};

#endif //BLUR_DETECTOR_BLUR_DETECTOR_H
