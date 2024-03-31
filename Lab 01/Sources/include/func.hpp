#ifndef FUNC_HPP
#define FUNC_HPP

#include <opencv2/opencv.hpp>

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
};

#endif // FUNC_HPP