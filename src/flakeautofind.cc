#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <ftw.h>

#define nl "\n"

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::vector;
using namespace cv;

Mat flatten(Mat);
vector<int*> circlePoints(int*, int*);

void printPoint(int*);

int main(int argc, char** argv){
	cout<<"OpenCV version: "<< CV_VERSION << nl;

	string image_path = argv[1];

	Mat img = imread( image_path, IMREAD_COLOR );
	int dim[] = {img.cols, img.rows};
	cout<<dim[0] << "x" << dim[1] <<nl;

	if( img.empty()){
		cout<< "Could not read the image: " << image_path << endl;
		return -1;
	}
	cout<<"Image Loaded!"<<nl;
	
	Mat contrast = Mat::zeros(img.size(), img.type() );
	float a = 10; //5;
	int b = 130; //80;
	for( int y = 0; y< dim[1]; y++){
		for( int x = 0; x < dim[0]; x++){
			for( int c = 0; c < img.channels(); c++){
				contrast.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(a * (img.at<Vec3b>(y,x)[c]-b));
			}
		}
	}

	Mat bgr[3]; 
	split(contrast, bgr);

	Mat blue = bgr[1];

	float scale = 0.6;

	Mat flat= flatten(blue);
	cout<<"Flattened!"<<nl;

	namedWindow("Debug View", WINDOW_NORMAL);
	resizeWindow("Debug View", scale*dim[1], scale*dim[0]);
	cout<<"Opened Debug Window!"<<nl;

	/*
	Mat debug(Size(dim[0]*2, dim[1]*2), CV_8UC1);
	bgr[0].copyTo(debug(Rect(0,0,dim[0]-1,dim[1]-1)));
	bgr[1].copyTo(debug(Rect(dim[0],0,2*dim[0],dim[1])));
	bgr[2].copyTo(debug(Rect(0,dim[1],dim[0],2*dim[1])));
	flat.copyTo(debug(Rect(dim[0],dim[1],2*dim[0], 2*dim[1])));
	*/

	imshow("Debug View", blue);
	cout<<"Displayed Debug!"<<endl;
	int k = waitKey(0);
	
	return 0;
}

Mat flatten(Mat img){
	int dim[] = {img.cols, img.rows};
	int centre[] = {dim[0]/2, dim[1]/2};
	vector<int*> points = circlePoints(100, centre);

	Mat flat = Mat::zeros(img.size(), img.type());
	
	int* point;
	for (int i = 0; i < points.size(); i++){
		point = points.at(i);
		cout<<"("<<point[0]<<','<<point[1]<<"), ";
		flat.at<uchar>(point[0],point[1]) = 255;
	}
	cout<<endl;

	return flat;
}

void printPoint(int* p){
	cout<<"("<<p[0]<<','<<p[1]<<"), ";
}

vector<int *> circlePoints(int* r, int* c){
	vector<int*> points;
	int x = r, y = 0;
	
	int P = 1-r;

	int point[2];

	while(x>y){
		y++;
			if (P<=0){
				P=P+2*y+1;
			}
			else{
				x--;
				P=P+2*y-2*x+1;
			}
		if (x<y)
			break;

		cout<<c[0]<<','<<x<<','<<c[1]<<','<<y<<nl;
		cout<<"oct1"<<nl;
		point[0]=c[0]+x;
		point[1]=c[1]+y;
		points.push_back(point);
		cout<<"oct8"<<nl;
		point[0]=c[0]+x;
		point[1]=c[1]-y;
		points.push_back(point);
		cout<<"oct4"<<nl;
		point[0]=c[0]-x;
		point[1]=c[1]+y;
		points.push_back(point);
		cout<<"oct5"<<nl;
		point[0]=c[0]-x;
		point[1]=c[1]-y;
		points.push_back(point);

		if (x != y){
			cout<<"oct2"<<nl;
			point[0]=c[0]+y;
			point[1]=c[1]+x;
			points.push_back(point);
			cout<<"oct7"<<nl;
			point[0]=c[0]+y;
			point[1]=c[1]-x;
			points.push_back(point);
			cout<<"oct3"<<nl;
			point[0]=c[0]-y;
			point[1]=c[1]+x;
			points.push_back(point);
			cout<<"oct6"<<nl;
			point[0]=c[0]-y;
			point[1]=c[1]-x;
			points.push_back(point);
		}
	}
	for(int i = 0; 
	return points;
}
