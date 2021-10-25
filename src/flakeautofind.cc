#include "faf_fns.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <ftw.h>
#include <math.h>

#define nl "\n"

using std::sort;
using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::vector;
using namespace cv;

int main(int argc, char** argv){ //Syntax is ./FlakeAutoFind {image/path} {n degree}
	cout<<"OpenCV version: "<< CV_VERSION << endl;
	if (argc < 2){
		cout<< "No image specified" <<endl;
		return -1;
	}
	string image_path = argv[1]; 

	//open image
	Mat img = imread( image_path, IMREAD_COLOR );

	//dimensions of image
	double dim[] = {(double) img.cols, (double) img.rows};
	cout<<dim[0] << "x" << dim[1] <<endl;
	
	//if image container is empty, return error
	if( img.empty()){
		cout<< "Could not read the image: " << image_path << endl;
		return -1;
	}
	cout<<"Image Loaded!"<<endl;
	
	//level of approximation. 
	int N = 3;
	if (argc > 2){
		N = std::stoi(argv[2]);
	}
	
	//blur to reduce some noise; kernel size 3x3 and gaussian blur
	//Mat imgblur;
	//GaussianBlur(img, imgblur, Size(3,3), 0, 0, BORDER_DEFAULT);

	
	//split the image into the blue-0, green-1, red-2 channels
	Mat bgr[3];	
	split(img, bgr); //replace img with imgblur?

	
	//placeholder for flattening
	Mat flat[3][2]; //should be [3][2] for bgr/ fg,bg

	//aspect ratio
	double ar = double(dim[0])/dim[1];
	//placeholder for coefficients
	double a[3][N];
	//flatten all three channels individually
	for( int i = 0; i<3; i++){
		//{1380,800} for 100x
		//{1400,900} for 20x
		double c[] = {1400,900};
		//findCenter(bgr[i], c);
		flatten(bgr[i], flat[i], a[i], c, ar, N);
	}
	
	/*
	//Morphological transforms to reduce noise
	Mat open;
	Mat const open_kernel = getStructuringElement(MORPH_RECT, Size(7,7));
	morphologyEx(flat[2][0], open, MORPH_OPEN, open_kernel);
	
	Mat erode;
	Mat const erode_kernel = getStructuringElement(MORPH_RECT, Size(5,5));
	morphologyEx(open, erode, MORPH_ERODE, erode_kernel);

	Mat close;
	Mat const close_kernel = getStructuringElement(MORPH_RECT, Size(20,20));
	morphologyEx(erode,close, MORPH_CLOSE, close_kernel);
	
	//Increase contrast
	double alpha = 10;
	double beta  = 15;
	double gamma = 0.5;
	Mat contrast = Mat::zeros(close.size(), close.type());
	//linear contrast increase
	//linearContrast(flat[2][0], blur, 10, 15); doesn't work for some reason, despite being exactly the same? i probably screwed up a pointer/reference or something
	for (int y = 0; y < close.rows; y++) {
		for (int x = 0; x < close.cols; x++) {
			contrast.at<uchar>(y,x) = alpha*(flat[2][0].at<uchar>(y,x) - beta);
		}
	}

	//blur again but for edge detection methods
	Mat blur;
	GaussianBlur(contrast, blur, Size(3,3), 0, 0, BORDER_DEFAULT);
	
	//Sobel Derivatives
	//Mat delx, dely;
	//Sobel(blur, delx, CV_16S, 1, 0, 1, 1, 0, BORDER_DEFAULT);
	//Sobel(blur, dely, CV_16S, 0, 1, 1, 1, 0, BORDER_DEFAULT);
	//Mat absdelx, absdely;
	//convertScaleAbs(delx, absdelx);
	//convertScaleAbs(dely, absdely);
	//Mat grad;
	//addWeighted(absdelx, 0.5, absdely, 0.5, 0, grad);
	

	//Laplacian edge detection/
	Mat lap, abslap;
	Laplacian( blur, lap, CV_16S, 1,1,0, BORDER_DEFAULT);
	convertScaleAbs(lap, abslap);
	*/

	Mat butter;
	//for 100x:
	//butterworth(butter, dim, 1000, 5);
	//for 20x:
	butterworth(butter, dim, 1000, 7);
	Mat cbutter;
	normalize(butter, cbutter, 0, 255, NORM_MINMAX);

	Mat vig;
        multiply(butter, flat[2][0], vig, 1, CV_8UC1);
	
	Mat dft;
	dftProc(vig,dft);

	//display window to show image manipulation
	double scale = 0.6;

	namedWindow("Debug View", WINDOW_NORMAL);
	resizeWindow("Debug View", scale*dim[1], scale*dim[0]);
	cout<<"Opened Debug Window!"<<endl;
	
	Mat debug(Size(dim[0]*2, dim[1]*2), CV_8UC1);

	bgr[2].copyTo(debug(Rect(0,0,dim[0],dim[1])));
	cout<<"Displayed Red!"<<nl;
	vig.copyTo(debug(Rect(dim[0],0,dim[0],dim[1])));
	cout<<"Displayed FT!"<<nl;
	flat[2][1].copyTo(debug(Rect(0,dim[1],dim[0],dim[1])));
	cout<<"Displayed BG!"<<nl;
	flat[2][0].copyTo(debug(Rect(dim[0],dim[1],dim[0],dim[1])));
	cout<<"Displayed Flattened!"<<endl;
	
	imshow("Debug View", debug);
	cout<<"Displayed Debug!"<<endl;

	int k = waitKey(0);
	if (k == 's'){
	imwrite("output.png", debug);
	}
	
	return 0;
}
