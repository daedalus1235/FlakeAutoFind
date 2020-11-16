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

void flatten(Mat, Mat*, float*, float, int);
Mat bumpContrast(Mat);
float* findDims(Mat);
float findMedian(vector<int>);
vector<vector<int>> ellipsePoints(float*, float*);

int main(int argc, char** argv){
	cout<<"OpenCV version: "<< CV_VERSION << nl;

	string image_path = argv[1];

	Mat img = imread( image_path, IMREAD_COLOR );
	float dim[] = {img.cols, img.rows};
	cout<<dim[0] << "x" << dim[1] <<nl;

	if( img.empty()){
		cout<< "Could not read the image: " << image_path << endl;
		return -1;
	}
	cout<<"Image Loaded!"<<nl;
	

	Mat contrast = bumpContrast(img);

	Mat blur;
	//GaussianBlur(img, blur, Size(5,5), 1);

	Mat bgr[3];	
	split(img, bgr);

	Mat blue = bgr[1];

	int a = bgr[2].at<uchar>(dim[0]/2,dim[1]/2);

	Mat flat[3][2];

	float c[2]= {dim[0]/2, dim[1]/2};
	float ar = dim[0]/dim[1];
	for( int i = 0; i<3; i++){
		flatten(bgr[i], flat[i], c, ar, 4);
	}

	float scale = 0.6;

	namedWindow("Debug View", WINDOW_NORMAL);
	resizeWindow("Debug View", scale*dim[1], scale*dim[0]);
	cout<<"Opened Debug Window!"<<nl;
	
	Mat open;
	Mat const open_kernel = getStructuringElement(MORPH_RECT, Size(7,7));
	morphologyEx(flat[2][0], open, MORPH_OPEN, open_kernel);
	
	Mat erode;
	Mat const erode_kernel = getStructuringElement(MORPH_RECT, Size(5,5));
	morphologyEx(open, erode, MORPH_ERODE, erode_kernel);

	Mat close;
	Mat const close_kernel = getStructuringElement(MORPH_RECT, Size(20,20));
	morphologyEx(erode,close, MORPH_CLOSE, close_kernel);

	Mat debug(Size(dim[0]*2, dim[1]*2), CV_8UC1);
	/*
	bgr[0].copyTo(debug(Rect(0,0,dim[0],dim[1])));
	bgr[1].copyTo(debug(Rect(dim[0],0,dim[0],dim[1])));
	bgr[2].copyTo(debug(Rect(0,dim[1],dim[0],dim[1])));
	flat[2].copyTo(debug(Rect(dim[0],dim[1],dim[0], dim[1])));
	*/
	bgr[2].copyTo(debug(Rect(0,0,dim[0],dim[1])));
	cout<<"Displayed Red!"<<nl;
	flat[2][0].copyTo(debug(Rect(dim[0],0,dim[0],dim[1])));
	cout<<"Displayed BG!"<<nl;
	flat[2][1].copyTo(debug(Rect(0,dim[1],dim[0],dim[1])));
	cout<<"Displayed Flattened!"<<nl;
	close.copyTo(debug(Rect(dim[0],dim[1],dim[0],dim[1])));
	cout<<"Displayed Morphology Transformed!"<<nl;
	
	imshow("Debug View", debug);
	cout<<"Displayed Debug!"<<endl;
	int k = waitKey(0);
	
	return 0;
}

float* findDims(Mat img){
	return 0;
}

void flatten(Mat img, Mat* flat, float* c, float ar, int n){
	cout<<"Samples to be taken: "<<n<<nl;
	float dim[2]={img.cols, img.rows};
	float a[n]; //coefficient array
	float samp2[n]; //sample locations squared
	float b[n]; //sample values

	flat[0] = Mat::zeros(img.rows, img.cols, CV_8UC1);//
	flat[1] = Mat::zeros(img.rows, img.cols, CV_8UC1);//
	
	samp2[0]=pow(dim[0],2)/n;
	for( int i = 1; i<n; i++){
		samp2[i] = samp2[i-1]+samp2[0];
	}//populate s^2 values
	
	for( int i = 0; i < n; i++ ){
		cout<<"entered loop: "<<i<<nl;
		float r[2] = {sqrt(samp2[i]), ar*sqrt(samp2[i])};
	       	vector<vector<int>> ellpts = ellipsePoints(c, r);
		cout<<"  generated "<<ellpts.size()<<" points"<<nl;

		vector<int> ellvals;
		for( int j = 0; j<ellpts.size(); j++){
			int val = img.at<uchar>(ellpts[j][1],ellpts[j][0]);
			cout<<"  measured point "<<j<<nl;
			ellvals.push_back(val);
			cout<<"  recorded measurement "<<j<<nl;
		}
		cout<<"took measurements"<<nl;
		b[i]=findMedian(ellvals);
	}//determine b at each s^2
	cout<<"Samples Taken!"<<nl;

	//regression
	int Sarr[n][n];
	for( int i = 0; i<n; i++){
		for( int j=0; j<n; i++){
			if( i == 0)
				Sarr[i][j]=1;
			else
				Sarr[i][j]=j*Sarr[i-1][j];
		}
	}
	Mat S(n,n, CV_32F, Sarr);
	cout<<"S="<<nl<<S<<nl;
	
	int Carr[n][n];
	for( int i = 0; i<n; i++ ){
		for( int j=0; j<n; j++){
			if(j>i){
				Carr[i][j]=0;
			}
			else if (i == 0){
				if( j==0 )
					Carr[i][j]=1;
				else
					Carr[i][j]=-Carr[i-1][j];
			}
			else{
				Carr[i][j]=Carr[i][j-1]*(i-j)/(j+1);
			}
		}
	}
	Mat C(n,n, CV_32F, Carr);
	cout<<"C="<<nl<<C<<nl;
		
	Mat T=C*S;
	cout<<"T="<<nl<<T<<nl;
	
	/*
	for( int x = 0; x<dim[0]; x++){
		for( int y = 0; y<dim[1]; y++){
			float s2 = (x-c[0])*(x-c[0]) + ar*ar*(y-c[1])*(y-c[1]);
			float bg = a[n];
			for( i = n-1; i>=0; i--){
				bg*=s2;
				bg+=a[i];
			}
			flat[1].at<uchar>(y,x)=bg;
			flat[0].at<uchar>(y,x)=img.at<uchar>(y,x) - bg;
		}
	}
	*/
	flat[1]=Mat::zeros(img.rows, img.cols, CV_8UC1);
	flat[0]=img;
	/*
	float a[2];
	float s_1 = dim[0]/3;
	float s_2 = dim[0]/9;

	float r_1[2] = {s_1, ar*s_1};
	vector<vector<int>> es1 = ellipsePoints(c, r_1);
	
	float r_2[2] = {s_2, ar*s_2};
	vector<vector<int>> es2 = ellipsePoints(c, r_2);

	vector<int> vals1, vals2;
	for( int i = 0; i<es1.size(); i++){
		int val = img.at<uchar>(es1[i][1],es1[i][0]);
		vals1.push_back(val);
	}
	for( int i = 0; i<es2.size(); i++){
		int val = img.at<uchar>(es2[i][1],es2[i][0]);
		vals2.push_back(val);
	}

	float B_1 = findMedian(vals1);
	float B_2 = findMedian(vals2);
	
	a[1]= (B_2-B_1)/(s_2*s_2-s_1*s_1);
	a[0]= B_1-a[1]*s_1*s_1;
	
	cout<<"Coefficients found: a0="<<a[0]<<", a1="<<a[1]<<nl;

	for( int x = 0; x<dim[0]; x++){
		for( int y = 0; y<dim[1]; y++){
			float s2 = (x-c[0])*(x-c[0]) + ar * ar * (y-c[1]) * (y-c[1]);
			float bg = a[0]+a[1]*s2;
			flat[1].at<uchar>(y,x)=bg;
			flat[0].at<uchar>(y,x)=img.at<uchar>(y,x) -bg;
		}
	}
	*/
}

Mat bumpContrast(Mat img){
	int dim[2]={img.cols, img.rows};
	float scale = 0.4;
	Mat grey;
        cvtColor(img, grey, COLOR_BGR2GRAY);
	
	int avg = mean(grey)[0]+5;	

	Mat contrast = Mat::zeros(img.size(), img.type() );
	float a = 100; 
	for( int y = 0; y< dim[1]; y++){
		for( int x = 0; x < dim[0]; x++){
			for( int c = 0; c < img.channels(); c++){
				contrast.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(a * (img.at<Vec3b>(y,x)[c]-avg));
			}
		}
	}
	return contrast;
}

float findMedian(vector<int> values){
	vector<int> sorted = values;
	sort(sorted.begin(), sorted.end());
	int length = sorted.size();
	int middle = length/2;
	if( length%2==0 ){
		return ( ((float)sorted[middle-1] +(float)sorted[middle])/2);
	}
	return (float)sorted[middle];

}

vector<vector<int>> ellipsePoints(float* c, float* r){
	vector<vector<int>> ellipse;
	float x = 0, y = r[1];
	float dx=0, dy=0, d1=0, d2=0;

	d1 = (r[1]*r[1])-(r[0]*r[0]*r[1])+(0.25*r[0]*r[0]);
	dx = 2 * r[1] * r[1] * x;
	dy = 2 * r[0] * r[0] * y;

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
			d1+= (dx+ (r[1]*r[1]));
		}
		else{
			x++;
			y--;
			dx+= (2*r[1]*r[1]);
			dy-= (2*r[0]*r[0]);
			d1 = d1 + dx - dy +(r[1]*r[1]);
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
			dy-= (2*r[0]*r[0]);
			d2 = d2 + (r[0]*r[0])-dy;
		}
		else{
			y--;
			x++;
			dx+= (2*r[1]*r[1]);
			dy-= (2*r[0]*r[0]);
			d2 = d2+dx-dy +(r[0]*r[0]);
		}
	}

	return ellipse;
}
