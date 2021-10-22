#ifndef FAF_FNS_H
#define FAF_FNS_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <ftw.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

void dftProc(const cv::Mat,cv::Mat&);
void flatten(const cv::Mat, cv::Mat*, double*, double*, double, int);
void gaussian(cv::Mat&, double*, double);
void butterworth(cv::Mat&, double*, double, int);
double findMedian(std::vector<int>& );
std::vector<std::vector<int>> ellipsePoints(double*, double*);

void linearContrast(cv::Mat, cv::Mat, double, double);
void gammaContrast(cv::Mat, cv::Mat, double);
double transmittance(cv::Mat, double*, double*, double, double*, int);
void findCenter(cv::Mat, double*);
#endif
