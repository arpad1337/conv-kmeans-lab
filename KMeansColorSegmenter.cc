#ifndef __arbc__KMeansColorSegmenter__
#define __arbc__KMeansColorSegmenter__

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "Config.cc"

class PixelLAB
{
public:
    uchar a, b, l; // 8 bit, 0-255

    PixelLAB()
    {
        this->a = (uchar)0;
        this->b = (uchar)0;
        this->l = (uchar)0;
    }

    PixelLAB(Vec3b lab_pixel)
    {
        this->a = (uchar)lab_pixel[1];
        this->b = (uchar)lab_pixel[2];
        this->l = (uchar)lab_pixel[0];
    }

    PixelLAB(uchar a, uchar b, uchar l)
    {
        this->a = a;
        this->b = b;
        this->l = l;
    }

    PixelLAB(double a, double b, double l)
    {
        this->a = (uchar)(floor(a));
        this->b = (uchar)(floor(b));
        this->l = (uchar)(floor(l));
    }

    bool operator==(const PixelLAB &pixel) const
    {
        return euclideanDistance3(
            (double)a, 
            (double)b, 
            (double)l, 
            (double)pixel.a, 
            (double)pixel.b, 
            (double)pixel.l
        ) == 0.0;
    }

    bool operator<(const PixelLAB &pixel) const
    {
        return (int)l < (int)pixel.l;
    }

    double operator-(const PixelLAB &pixel) const
    {
        return euclideanDistance3(
            (double)a, 
            (double)b, 
            (double)l, 
            (double)pixel.a, 
            (double)pixel.b, 
            (double)pixel.l
        );
    }

    PixelLAB operator/(const PixelLAB &pixel) const
    {
        PixelLAB copy(a, b, l);
        copy.a = (uchar)(floor((double)((int)copy.a + (int)pixel.a) / 2));
        copy.b = (uchar)(floor((double)((int)copy.b + (int)pixel.b) / 2));
        copy.l = (uchar)(floor((double)((int)copy.l + (int)pixel.l) / 2));
        return copy;
    }

    ~PixelLAB()
    {
    }

    static double euclideanDistance3(double x1, double y1, double c1, double x2, double y2, double c2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(c1 - c2, 2));
    }

    static double euclideanDistance2(double x1, double y1, double x2, double y2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    }
};

class KMeansColorSegmenter
{
private:
    std::vector<PixelLAB> clusterCentres;
    cv::Mat image;
    cv::Mat output;
    cv::Mat labels;
    uint K;
    bool isDebug;
    uint padding;

public:
    static KMeansColorSegmenter create(cv::Mat image, uint K, uint padding = 0)
    {
        KMeansColorSegmenter instance(image, K, padding, true);
        return instance;
    }

    KMeansColorSegmenter()
    {
        
    }

    KMeansColorSegmenter(cv::Mat image, uint K, uint padding, bool isLab = false)
    {
        this->isDebug = Config::isDebug();
        if (!isLab)
        {
            this->image = this->convertToLAB(image);
            this->output = this->image.clone();
        }
        else
        {
            this->image = image;
            this->output = this->image.clone();
        }
        this->padding = padding;
        this->labels = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
        this->setK(K);
    }

    void setK(uint K)
    {
        int root = floor(sqrt((float)(K)));
        this->K = root * root;
        if (clusterCentres.size() == 0)
        {
            initClusterCentres();
        }
        else
        {
            // reduce exiting clusters to new K by grouping per distance
            double a;
            uint x, y;
            uint size = clusterCentres.size();

            if (size == this->K)
            {
                return;
            }

            sort(clusterCentres.begin(), clusterCentres.end());

            while (size > this->K)
            {
                double min = clusterCentres[0] - clusterCentres[1];

                for (int i = 0; i < size; i++)
                {
                    for (int j = i + 1; j < size; j++)
                    {
                        a = clusterCentres[i] - clusterCentres[j];

                        if (a <= min)
                        {
                            min = a;
                            x = i;
                            y = j;
                        }
                    }
                }

                PixelLAB newPixel = clusterCentres[x] / clusterCentres[y];
                clusterCentres[x] = newPixel;
                clusterCentres.erase(clusterCentres.begin() + y);

                size = size - 1;
            }
        }

        assignNewClusterCentres();
    }

    void setPreviousOutputAsInput()
    {
        this->image = this->output.clone();
        this->output = this->image.clone();
    }

    void setInput(Mat bgrImage)
    {
        this->image = this->convertToLAB(bgrImage);
        this->output = this->convertToLAB(bgrImage);
        initClusterCentres();
    }

    void setInputLAB(Mat labImage)
    {
        this->image = labImage.clone();
        this->output = labImage.clone();
        initClusterCentres();
    }

    Mat getOutput()
    {
        Mat bgr_output;
        cvtColor(this->output, bgr_output, CV_Lab2BGR);
        return bgr_output;
    }

    Mat getLABOutput()
    {
        return this->output;
    }

    void train(int iterations)
    {
        if (isDebug)
        {
            std::cout << "Training with K " << this->K << ", epochs: " << iterations << std::endl;
        }
        for (int i = 0; i < iterations; i++)
        {
            computeCentroids();
            assignNewClusterCentres();
        }
    }

    void convert()
    {
        for (int r = 0; r < image.rows; r++)
        {
            for (int c = 0; c < image.cols; c++)
            {
                PixelLAB p = clusterCentres.at(labels.at<uchar>(r, c));
                cv::Vec3b lab_pixel(p.l, p.a, p.b);
                output.at<cv::Vec3b>(r, c) = lab_pixel;
            }
        }
    }

    void distributeClusterCentresEqually() {
        uint i = 0;
        while (i < K)
        {
            PixelLAB p = clusterCentres.at(i);
            clusterCentres.at(i) = PixelLAB((uchar)floor(i * (255.0 / (clusterCentres.size() - 1))), p.a, p.b);
            i++;
        }
    }

    vector<PixelLAB> getClusterCentres()
    {
        return clusterCentres;
    }

private:
    void initClusterCentres()
    {
        clusterCentres.resize(0);
        uint i = 0;
        int root = floor(sqrt((float)this->K));
        uint rStep = floor((float)(this->image.rows - (this->padding * 2)) / (root - 1));
        uint cStep = floor((float)(this->image.cols - (this->padding * 2)) / (root - 1));
        uint r = this->padding - 1;
        uint c = this->padding - 1;
        while (i < this->K)
        {
            cv::Vec3b lab_pixel = this->image.at<cv::Vec3b>(r, c);
            uchar a = lab_pixel[1];
            uchar b = lab_pixel[2];
            uchar l = lab_pixel[0];
            clusterCentres.push_back(PixelLAB(a, b, l));
            if (r + rStep > this->image.rows - this->padding - 1)
            {
                r = this->padding - 1;
                c = c + cStep;
            }
            else
            {
                r = r + rStep;
            }
            i++;
        }

        sort(clusterCentres.begin(), clusterCentres.end());
    }

    Mat convertToLAB(Mat src)
    {
        Mat lab_image;
        cv::cvtColor(src, lab_image, CV_BGR2Lab);
        return lab_image;
    }

    void assignNewClusterCentres()
    {
        uint size = clusterCentres.size();

        for (int r = 0; r < image.rows; r++)
        {
            for (int c = 0; c < image.cols; c++)
            {
                int centroidLabel = 0;
                uchar a, b, l;
                cv::Vec3b lab_pixel = image.at<cv::Vec3b>(r, c);

                a = lab_pixel[1];
                b = lab_pixel[2];
                l = lab_pixel[0];

                PixelLAB pixel(a, b, l);

                double distance, min_dist;

                min_dist = clusterCentres[0] - pixel;

                for (int i = 1; i < size; i++)
                {
                    distance = clusterCentres[i] - pixel;

                    if (distance < min_dist)
                    {
                        min_dist = distance;
                        centroidLabel = i;
                    }
                }

                labels.at<uchar>(r, c) = (uchar)centroidLabel;
            }
        }
    }

    void computeCentroids()
    {
        uint size = clusterCentres.size();
        for (int i = 0; i < size; i++)
        {
            double mean_a = 0.0, mean_b = 0.0, mean_l = 0.0;
            int n = 0;
            for (int r = 0; r < image.rows; r++)
            {
                for (int c = 0; c < image.cols; c++)
                {
                    if (labels.at<uchar>(r, c) == i)
                    {
                        cv::Vec3b lab_pixel = image.at<cv::Vec3b>(r, c);
                        mean_a += lab_pixel[1];
                        mean_b += lab_pixel[2];
                        mean_l += lab_pixel[0];
                        n++;
                    }
                }
            }
            mean_a /= n;
            mean_b /= n;
            mean_l /= n;
            clusterCentres.at(i) = PixelLAB(mean_a, mean_b, mean_l);
        }
    }
};

#endif 
