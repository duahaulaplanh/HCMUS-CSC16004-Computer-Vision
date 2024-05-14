#ifndef FUNC_HPP
#define FUNC_HPP

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#endif

class Image
{
public:
    Image();
    Image(const cv::Mat& img);

    static Image ReadImg(std::string imgPath);
    void Write(std::string imgPath);
    void Show(std::string windowName);

    Image ConvertGrayScale();
    Image AdjustBrightness(int value);
    Image AdjustContrast(float value);
    Image AverageFilter(int k);
    Image MedianFilter(int k);
    Image GaussianFilter(int k);
    Image SobelDetect();
    Image LaplaceDetect();
    Image HarrisCornerDetect(double k);

#ifdef HAVE_OPENCV_XFEATURES2D
    static Image MatchingImgSIFT(const Image& object, const Image& scene);

    // Matching descriptor vectors with a FLANN based matcher
    static std::vector<cv::DMatch> Matching(const cv::Mat& descObj, const cv::Mat& descScene, float threshold);

    // Draw line between matching point (homography)
    static void DrawHomography(cv::Mat& imgMatches, const std::vector<cv::Point2f>& objPoint, const std::vector<cv::Point2f>& scenePoint, const cv::Mat& object);
#endif

private:
    cv::Mat data;
    int rows;
    int cols;
    int channels;

    uchar Clamp(int value);

    template <typename T>
    T Convolution1C(const cv::Mat& img, const cv::Mat& kernel, int x, int y, int offset)
    {
        assert(kernel.rows == kernel.cols);

        T sum;
        for (int i=0; i < kernel.rows; i++)
        {
            for (int j=0; j < kernel.cols; j++)
            {
                int idx_x = x + i - offset;
                int idx_y = y + j - offset;

                sum += img.at<uchar>(idx_x, idx_y) * kernel.at<double>(i, j);
            }
        }

        return sum;
    }

    template <typename T>
    cv::Vec<T, 3> Convolution3C(const cv::Mat& img, const cv::Mat& kernel, int x, int y, int offset)
    {
        assert(kernel.rows == kernel.cols);

        cv::Vec<T, 3> sum;
        for (int i=0; i < kernel.rows; i++)
        {
            for (int j=0; j < kernel.cols; j++)
            {
                int idx_x = x + i - offset;
                int idx_y = y + j - offset;

                cv::Vec3b pixel = img.at<cv::Vec3b>(idx_x, idx_y);
                sum[0] += pixel[0] * kernel.at<double>(i, j);
                sum[1] += pixel[1] * kernel.at<double>(i, j);
                sum[2] += pixel[2] * kernel.at<double>(i, j);
            }
        }

        return sum;
    }

    std::pair<cv::Mat, cv::Mat> CalculateGradientSobel();
};

#endif // FUNC_HPP