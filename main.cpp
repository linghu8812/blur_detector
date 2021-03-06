#include "blur_detector.h"

int main()
{
    cv::Mat src_img = cv::imread("test.jpg");
	if (!src_img.data)
	{
		printf("Image file is not exist!\n");
		return -1;
	}
    cv::Mat gray_img;
    cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);
    BlurDetector bd;
    bool result = bd.check_image_size(gray_img);
    float blur = bd.get_blurness(gray_img);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat blur33;
    cv::blur(gray_img, blur33, cv::Size(3, 3));
    blur = bd.get_blurness(blur33);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat blur55;
    cv::blur(gray_img, blur55, cv::Size(5, 5));
    blur = bd.get_blurness(blur55);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat blur77;
    cv::blur(gray_img, blur77, cv::Size(7, 7));
    blur = bd.get_blurness(blur77);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat gaussian33;
    cv::GaussianBlur(gray_img, gaussian33, cv::Size(3, 3), 0);
    blur = bd.get_blurness(gaussian33);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat gaussian55;
    cv::GaussianBlur(gray_img, gaussian55, cv::Size(5, 5), 0);
    blur = bd.get_blurness(gaussian55);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat gaussian77;
    cv::GaussianBlur(gray_img, gaussian77, cv::Size(7, 7), 0);
    blur = bd.get_blurness(gaussian77);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat median33;
    cv::medianBlur(gray_img, median33, 3);
    blur = bd.get_blurness(median33);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat median55;
    cv::medianBlur(gray_img, median55, 5);
    blur = bd.get_blurness(median55);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat median77;
    cv::medianBlur(gray_img, median77, 7);
    blur = bd.get_blurness(median77);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat bilateral33;
    cv::bilateralFilter(gray_img, bilateral33, 5, 21, 21);
    blur = bd.get_blurness(bilateral33);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat bilateral55;
    cv::bilateralFilter(gray_img, bilateral55, 7, 31, 31);
    blur = bd.get_blurness(bilateral55);
    std::cout << "Blurness: " << blur << std::endl;
    cv::Mat bilateral77;
    cv::bilateralFilter(gray_img, bilateral77, 9, 41, 41);
    blur = bd.get_blurness(bilateral77);
    std::cout << "Blurness: " << blur << std::endl;
    return 0;
}