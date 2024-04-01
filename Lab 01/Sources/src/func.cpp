#include "func.hpp"
#include <cassert>

Image::Image()
: data()
, rows(0)
, cols(0)
, channels(0)
{}

Image::Image(const cv::Mat& img)
: data(img)
, rows(img.rows)
, cols(img.cols)
, channels(img.channels())
{}

Image Image::ReadImg(std::string imgPath)
{
    std::string path = std::string(SOURCE_PATH) + imgPath;

    cv::Mat data = cv::imread(path);

    // check if reading image properly
    assert(!data.empty());

    // if yes
    return Image(data);
}

void Image::Write(std::string imgPath)
{
    std::string path = std::string(SOURCE_PATH) + imgPath;
    cv::imwrite(path, data);    
}

void Image::Show(std::string windowName)
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);    
    cv::imshow(windowName, data);
}

Image Image::ConvertGrayScale()
{
    if (channels == 1)
        return Image(data);

    cv::Mat newImg(rows, cols, CV_8UC1);
    for (int x=0; x < rows; x++)
    {
        for (int y=0; y < cols; y++)
        {
            cv::Vec3b pixel = data.at<cv::Vec3b>(x, y);
            newImg.at<uchar>(x, y) = 0.11*pixel[0] + 0.59*pixel[1] + 0.3*pixel[2];
        }
    }

    return Image(newImg);
}

Image Image::AdjustBrightness(int value)
{
    cv::Mat newImg;
    if (channels == 1) { newImg = cv::Mat(rows, cols, CV_8UC1); }
    else               { newImg = cv::Mat(rows, cols, CV_8UC3); }

    for (int x=0; x < rows; x++)
    {
        for (int y=0; y < cols; y++)
        {
            if (channels == 1) // grayscale image
            {
                newImg.at<uchar>(x, y) = Clamp(static_cast<int>(data.at<uchar>(x, y)) * value);
            }
            else // color image
            {
                cv::Vec3b pixel = data.at<cv::Vec3b>(x, y);

                // g(x) = f(x) + beta
                pixel[0] = Clamp(static_cast<int>(pixel[0]) + value);
                pixel[1] = Clamp(static_cast<int>(pixel[1]) + value);
                pixel[2] = Clamp(static_cast<int>(pixel[2]) + value);

                // assign pixel to new img
                newImg.at<cv::Vec3b>(x, y) = pixel;
            }
        }
    }
    return Image(newImg); 
}

Image Image::AdjustContrast(float value)
{
    assert(value >= 0.0f);

    cv::Mat newImg;
    if (channels == 1) { newImg = cv::Mat(rows, cols, CV_8UC1); }
    else               { newImg = cv::Mat(rows, cols, CV_8UC3); }

    for (int x=0; x < rows; x++)
    {
        for (int y=0; y < cols; y++)
        {
            if (channels == 1) // grayscale image
            {
                newImg.at<uchar>(x, y) = Clamp(static_cast<int>(static_cast<float>(data.at<uchar>(x, y)) * value));
            }
            else // color image
            {
                cv::Vec3b pixel = data.at<cv::Vec3b>(x, y);
                
                // g(x) = alpha x f(x)
                pixel[0] = Clamp(static_cast<int>(static_cast<float>(pixel[0]) * value));
                pixel[1] = Clamp(static_cast<int>(static_cast<float>(pixel[1]) * value));
                pixel[2] = Clamp(static_cast<int>(static_cast<float>(pixel[2]) * value));

                // assign pixel to new img
                newImg.at<cv::Vec3b>(x, y) = pixel;
            }
        }
    }
    return Image(newImg); 
}

Image Image::AverageFilter(int k)
{
    assert(k >= 1); // kernel size bigger than 1
    assert(k % 2 != 0); // kernel size is odd

    // create kernel (average)
    cv::Mat kernel = cv::Mat(k, k, CV_64F, 1.0);
    kernel = kernel / static_cast<double>(k * k); 

    // get offset value
    int offset = k / 2;

    // create new image
    cv::Mat newImg;
    if (channels == 1) { newImg = cv::Mat(rows, cols, CV_8UC1); }
    else               { newImg = cv::Mat(rows, cols, CV_8UC3); }

    for (int x=offset; x < rows-offset; x++)
    {
        for (int y=offset; y < cols-offset; y++)
        {
            if (channels == 1)
                newImg.at<uchar>(x, y) = Convolution1C<uchar>(data, kernel, x, y, offset);
            else
                newImg.at<cv::Vec3b>(x, y) = Convolution3C<uchar>(data, kernel, x, y, offset);
        }
    }

    return Image(newImg);
}

Image Image::MedianFilter(int k)
{
    assert(k >= 1); // kernel size bigger than 1
    assert(k % 2 != 0); // kernel size is odd

    // create new image
    cv::Mat newImg;
    if (channels == 1) { newImg = cv::Mat(rows, cols, CV_8UC1); }
    else               { newImg = cv::Mat(rows, cols, CV_8UC3); }

    int offset = k/2;

    for (int x=offset; x < rows-offset; x++)
    {
        for (int y=offset; y < cols-offset; y++)
        {
            if (channels == 1)
            {
                std::vector<uchar> vec;
                for (int i=0; i < k; i++)
                {
                    for (int j=0; j < k; j++)
                    {
                        int idx_x = x + i - offset;
                        int idx_y = y + j - offset;
                        vec.push_back(data.at<uchar>(idx_x, idx_y));
                    }
                }
                std::sort(vec.begin(), vec.end());
                newImg.at<uchar>(x, y) = vec[vec.size() / 2 + 1];
            }
            else
            {
                std::vector<uchar> vecR;
                std::vector<uchar> vecB;
                std::vector<uchar> vecG;

                for (int i=0; i < k; i++)
                {
                    for (int j=0; j < k; j++)
                    {
                        int idx_x = x + i - offset;
                        int idx_y = y + j - offset;

                        cv::Vec3b pixel = data.at<cv::Vec3b>(idx_x, idx_y); 
                        vecR.push_back(pixel[0]);
                        vecB.push_back(pixel[1]);
                        vecG.push_back(pixel[2]);
                    }
                }

                std::sort(vecR.begin(), vecR.end());
                std::sort(vecB.begin(), vecB.end());
                std::sort(vecG.begin(), vecG.end());

                cv::Vec3b newPixel;
                newPixel[0] = vecR[vecR.size() / 2 + 1];
                newPixel[1] = vecB[vecB.size() / 2 + 1];
                newPixel[2] = vecG[vecG.size() / 2 + 1];

                newImg.at<cv::Vec3b>(x, y) = newPixel;
            }
        }
    }

    return Image(newImg);
}

Image Image::GaussianFilter(int k)
{
    assert(k >= 1); // kernel size bigger than 1
    assert(k % 2 != 0); // kernel size is odd

    // create kernel (gaussian)
    double sum_kernel = 0.0;
    double sigma = 1.0; // change this to make something change
    cv::Mat kernel = cv::Mat(k, k, CV_64FC1);

    // get offset value
    int offset = k / 2;

    for (int x=0; x < k; x++)
    {
        for (int y=0; y < k; y++)
        {
            double i = x - offset;
            double j = y - offset;
            double value = (1.0 / (2 * M_PI * pow(sigma, 2))) * exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)));
            kernel.at<double>(x, y) = value;
            sum_kernel += value;
        }
    }

    kernel = kernel / sum_kernel;

    // create new image
    cv::Mat newImg;
    if (channels == 1) { newImg = cv::Mat(rows, cols, CV_8UC1); }
    else               { newImg = cv::Mat(rows, cols, CV_8UC3); }

    for (int x=offset; x < rows-offset; x++)
    {
        for (int y=offset; y < cols-offset; y++)
        {
            if (channels == 1)
                newImg.at<uchar>(x, y) = Convolution1C<uchar>(data, kernel, x, y, offset);
            else
                newImg.at<cv::Vec3b>(x, y) = Convolution3C<uchar>(data, kernel, x, y, offset);
        }
    }

    return Image(newImg);
}

Image Image::SobelDetect()
{
    // convert image to grayscale
    Image grayScale = this->ConvertGrayScale(); 

    // sobel kernel size
    int kerSize = 3;

    // Create Sobel kernel
    double sobelX[kerSize][kerSize] = {{-1.0, 0.0, 1.0},
                                       {-2.0, 0.0, 2.0},
                                       {-1.0, 0.0, 1.0}};
    cv::Mat kernelX = cv::Mat(kerSize, kerSize, CV_64F, &sobelX);

    double sobelY[kerSize][kerSize] = {{-1.0, -2.0, -1.0},
                                       {0.0, 0.0, 0.0},
                                       {1.0, 2.0, 1.0}};
    cv::Mat kernelY = cv::Mat(kerSize, kerSize, CV_64F, &sobelY);

    // Result after convolution with sobel kernel
    cv::Mat gradX = cv::Mat(rows, cols, CV_64F);
    cv::Mat gradY = cv::Mat(rows, cols, CV_64F);

    int offset = kerSize / 2;

    // Convolution image with each kernel
    for (int x=offset; x < rows-offset; x++)
    {
        for (int y=offset; y < cols-offset; y++)
        {
            gradX.at<double>(x, y) = Convolution1C<double>(grayScale.data, kernelX, x, y, offset);
            gradY.at<double>(x, y) = Convolution1C<double>(grayScale.data, kernelY, x, y, offset);
        }
    }

    // Turn image to binary image with edge is black color
    double threshold = 70.0;
    cv::Mat res = cv::Mat(rows, cols, CV_8U);
    for (int x=0; x < rows; x++)
    {
        for (int y=0; y < cols; y++)
        {
            double prod = sqrt(pow(gradX.at<double>(x, y), 2) + pow(gradY.at<double>(x, y), 2));
            res.at<uchar>(x, y) = (prod > threshold) ? 0 : 255;
        }
    }

    return res;
}

Image Image::LaplaceDetect()
{
    // convert image to grayscale
    Image grayScale = this->ConvertGrayScale(); 

    // laplace kernel size
    int kerSize = 3;

    // create laplace kernel
    double laplace[kerSize][kerSize] = {{0, 1, 0},
                                        {1, -4, 1},
                                        {0, 1, 0}};
    cv::Mat laplaceKer = cv::Mat(kerSize, kerSize, CV_64F, &laplace);

    int offset = kerSize / 2;

    // Turn image to binary image with edge is black color
    double threshold = 50.0;
    cv::Mat res = cv::Mat(rows, cols, CV_8U);
    for (int x=offset; x < rows-offset; x++)
    {
        for (int y=offset; y < cols-offset; y++)
        {
            double conv = Convolution1C<double>(grayScale.data, laplaceKer, x, y, offset);
            res.at<uchar>(x, y) = (conv > threshold) ? 0 : 255;
        }
    }

    return Image(res);
}

uchar Image::Clamp(int value)
{
    if (value > 255)
        return 255;
    else if (value < 0)
        return 0;
    else
        return static_cast<uchar>(value);
}