#include "func.hpp"

int main(int argc, char const *argv[])
{
    std::string cmd = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];

    Image original = Image::ReadImg(input);
    Image newImg;

    if (cmd == "-rgb2gray")
    {
        newImg = original.ConvertGrayScale();
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Grayscale");
    }
    else if (cmd == "-brightness")
    {
        int value = atoi(argv[4]);
        newImg = original.AdjustBrightness(value);
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Brightness");
    }
    else if (cmd == "-constrast")
    {
        float value = atof(argv[4]);
        newImg = original.AdjustContrast(value);
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Contrast");
    }
    else if (cmd == "-avg")
    {
        int k = atoi(argv[4]);
        newImg = original.AverageFilter(k);
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Average Filter");
    }
    else if (cmd == "-med")
    {
        int k = atoi(argv[4]);
        newImg = original.MedianFilter(k);
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Median Filter");
    }
    else if (cmd == "-gau")
    {
        int k = atoi(argv[4]);
        newImg = original.GaussianFilter(k);
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Gaussian Filter");
    }
    else if (cmd == "-sobel")
    {
        newImg = original.SobelDetect();
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Sobel Detection");
    }
    else if (cmd == "-laplace")
    {
        newImg = original.LaplaceDetect();
        newImg.Write(output);
        original.Show("Original");
        newImg.Show("Laplace Detect");
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}