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
vector<vector<int>> ellipsePoints(int*, int*);

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

	Mat test = Mat::zeros( , CV_8UC1);
	vector<vector<int>> ellipse = ellipsePoints


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

vector<vector<int>> ellipsePoints(int* r, int* c){
	vector<vector<int>> ellipse;
	float x = r[0], y = 0;
	float dx dy d1 d2;

	d1 = (r[1]*r[1])-(r[0]*r[0]*r[1])+(0.25*r[0]*r[x]);
	dx = 2 * r[1] * r[1] * x
	dy = 2 * r[0] * r[0] * y

	while( dx < dy ){
		vector<int> v1, v2, v3, v4;
		
		v1.push_back( (int) x+c[0]);
		v1.push_back( (int) y+c[1]);	
		
		v2.push_back( (int) -x+c[0]);
		v2.push_back( (int) y+c[1]);

		v3.push_back( (int) x+c[0]);
		v3.push_back( (int) -y+c[1]);

		v4.push_back( (int) -x+c[0]);
		v4.push_back( (int) -y+c[1]);

		ellipse.push_back(v1);
		ellipse.push_back(v2);
		ellipse.push_back(v3);
		ellipse.push_back(v4);

		if (d1<0){
			x++;
			dx+= (2*r[1]*r[1]);
			d1+= (dx+ (r[1]*r[1]);
		}
		else{
			x++;
			y--;
			dx+= (2*r[1]*r[1]);
			dy-= (2*r[0]*r[0]);
			d1 = dl + dx - dy +(r[1]*r[1]);
		}
	}
	
	d2 = ( (r[1]*r[1])*((x+0.5)*(x+0.5)) ) +
		( (r[0]*r[0]) * ((y-1)*(y-1)) )-
		( r[0] * r[0] * r[1] * r[1] );

	while( y>=0 ){
		vector<int> v1, v2, v3, v4;
		
		v1.push_back( (int) x+c[0]);
		v1.push_back( (int) y+c[1]);	
		
		v2.push_back( (int) -x+c[0]);
		v2.push_back( (int) y+c[1]);

		v3.push_back( (int) x+c[0]);
		v3.push_back( (int) -y+c[1]);

		v4.push_back( (int) -x+c[0]);
		v4.push_back( (int) -y+c[1]);

		ellipse.push_back(v1);
		ellipse.push_back(v2);
		ellipse.push_back(v3);
		ellipse.push_back(v4);

		if( d2>0 ){
			y--;
			dy-= (2*r[0]*r[0]));
			d2 = d2 + (r[0]*r[0])-dy;
		}
		else{
			y--;
			x++;
			dx+= (2*r[1]*r[1]);
			dy-= (2*r[0]*r[0]);
			d2 = d2+dx-dy +(r[0]*r[0]);
	}

	return ellipse;
}
