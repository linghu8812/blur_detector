#include "blur_detector.h"

bool BlurDetector::check_image_size(cv::Mat &image, int block_size)
{
    bool bFlag = true;
    int height = image.rows, width = image.cols;
    int _y = height % block_size;
    int _x = width % block_size;
    int pad_x = 0, pad_y = 0;
    if (_y != 0)
    {
        pad_y = block_size - _y;
        bFlag = false;
    }
    if (_x != 0)
    {
        pad_x = block_size - _x;
        bFlag = false;
    }
    cv::copyMakeBorder(image, image, 0, pad_y, 0, pad_x, cv::BORDER_REPLICATE);
    return bFlag;
}

float BlurDetector::get_blurness(cv::Mat image, int block_size)
{
    cv::Mat hist = cv::Mat::zeros(cv::Size(block_size, block_size), CV_32FC1);
    int channels = image.channels();
    if (channels==3)
        cv::cvtColor(image, image, CV_BGR2GRAY);
    int height = image.rows, width = image.cols;
    int round_v = width / block_size;
    int round_h = height / block_size;
    for (int v = 0; v < round_v; v++)
        for (int h = 0; h < round_h; h++)
        {
            int v_start = v * block_size;
            int h_start = h * block_size;
            cv::Mat image_patch;
            image(cv::Rect(v_start, h_start, 8, 8)).convertTo(image_patch, CV_32FC1);
            cv::Mat patch_spectrum;
            cv::dct(image_patch, patch_spectrum);
            cv::Mat patch_none_zero;
            cv::threshold(cv::abs(patch_spectrum), patch_none_zero, dct_threshold, 1, cv::THRESH_BINARY);
			hist += patch_none_zero;
        }
    float blr_thresh = max_hist * hist.at<float>(0, 0);
    cv::Mat blur;
    cv::threshold(hist, blur, blr_thresh, 1, cv::THRESH_BINARY_INV);
    cv::Mat blur_mul(8, 8, CV_32FC1, hist_weight);
    blur = blur.mul(blur_mul);
    cv::Scalar _blur_sum = cv::sum(blur);
    return _blur_sum[0] / weight_total;
}