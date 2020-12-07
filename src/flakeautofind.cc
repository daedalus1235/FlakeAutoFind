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

void flatten(Mat, Mat*, float*, float*, float, int);
float transmittance(Mat, float*, float*, float, float*, int);
float findMedian(vector<int>);
vector<vector<int>> ellipsePoints(float*, float*);

int main(int argc, char** argv){
	cout<<"OpenCV version: "<< CV_VERSION << endl;

	string image_path = argv[1];

	Mat img = imread( image_path, IMREAD_COLOR );
	float dim[] = {img.cols, img.rows};
	cout<<dim[0] << "x" << dim[1] <<endl;

	if( img.empty()){
		cout<< "Could not read the image: " << image_path << endl;
		return -1;
	}
	cout<<"Image Loaded!"<<endl;

	int N = 3;
	if (argc >1){
		N = std::stoi(argv[2]);
	}
	
	Mat bgr[3];	
	split(img, bgr);

	Mat blue = bgr[1];

	Mat flat[3][2];

	float c[2]= {dim[0]/2, dim[1]/2};
	float ar = dim[0]/dim[1];
	float a[3][N];
	for( int i = 0; i<3; i++){
		flatten(bgr[i], flat[i], a[i], c, ar, N);
	}
	
	float t = transmittance(bgr[2], c, c, ar, a[2], N);

	float scale = 0.6;

	namedWindow("Debug View", WINDOW_NORMAL);
	resizeWindow("Debug View", scale*dim[1], scale*dim[0]);
	cout<<"Opened Debug Window!"<<endl;
	
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

	bgr[2].copyTo(debug(Rect(0,0,dim[0],dim[1])));
	cout<<"Displayed Red!"<<nl;
	flat[2][0].copyTo(debug(Rect(dim[0],0,dim[0],dim[1])));
	cout<<"Displayed BG!"<<nl;
	flat[2][1].copyTo(debug(Rect(0,dim[1],dim[0],dim[1])));
	cout<<"Displayed Flattened!"<<nl;
	close.copyTo(debug(Rect(dim[0],dim[1],dim[0],dim[1])));
	cout<<"Displayed Morphology Transformed!"<<endl;
	
	imshow("Debug View", debug);
	cout<<"Displayed Debug!"<<endl;
	int k = waitKey(0);
	
	return 0;
}

void flatten(Mat img, Mat* flat, float* a, float* c, float ar, int n){
	cout<<"Samples to be taken: "<<n<<endl;
	float dim[2]={img.cols, img.rows};
	float samp2[n]; //sample locations squared
	float b[1][n]; //sample values

	flat[0] = Mat::zeros(img.rows, img.cols, CV_8UC1);//
	flat[1] = Mat::zeros(img.rows, img.cols, CV_8UC1);//
	
	samp2[0]=(dim[0]-c[0])*(dim[0]-c[0])+ar*ar*(dim[1]-c[1])*(dim[1]-c[1]);
	samp2[0]-=2;
	samp2[0]/=n;
	cout<<"Sample Separation: "<<samp2[0]<<endl;

	for( int i = 1; i<n; i++){
		samp2[i] = samp2[i-1]+samp2[0];
	}//populate s^2 values
	
	for( int i = 0; i < n; i++ ){
		float r[2] = {sqrt(samp2[i]), sqrt(samp2[i])/ar};
	       	vector<vector<int>> ellpts = ellipsePoints(c, r);

		vector<int> ellvals;
		for( int j = 0; j<ellpts.size(); j++){
			int y = ellpts[j][1];
			int x = ellpts[j][0];
			if ( x>=0 && y>=0 && x<dim[0] && y<dim[1] ){
				int val = img.at<uchar>(ellpts[j][1],ellpts[j][0]);
				ellvals.push_back(val);
			}
		}
		b[0][i]=findMedian(ellvals);
		cout<<"B"<<i<<": "<<b[0][i]<<endl;
	}//determine b at each s^2
	cout<<"Samples Taken!"<<endl;

	//regression
	float Sarr[n][n];
	for( int i = 0; i < n; i++){
		for( int j=0; j < n; j++){
			if( j == 0 )
				Sarr[i][j]=1;
			else
				Sarr[i][j]=(i+1)*Sarr[i][j-1];
		}
	}
	Mat S(n,n, CV_32F, Sarr);
	cout<<"S="<<nl<<S<<endl;
	
	float Carr[n][n];
	for( int i = 0; i < n; i++ ){
		for( int j = 0; j < n; j++){
			if( j > i ){
				Carr[i][j]=0;
			}
			else if ( j == 0 ){
				if( i == 0 )
					Carr[i][j]=1;
				else
					Carr[i][j]=-1*Carr[i-1][j];
			}
			else{
				Carr[i][j]=-Carr[i][j-1]*(i-j+1)/(j);
			}
		}
	}
	Mat C(n,n, CV_32F, Carr);
	cout<<"C="<<nl<<C<<endl;
		
	Mat T=C*S;
	cout<<"T="<<nl<<T<<endl;

	Mat B(n,1,CV_32F,b);
	cout<<"B="<<nl<<B<<endl;
	
	Mat CB = C*B;
	cout<<"CB="<<nl<<CB<<endl;

	for( int i = n-1; i >= 0; i--){
		a[i]=CB.at<float>(i,0);
		for( int j = i+1; j < n; j++){
			a[i]-=(T.at<float>(j,i)*a[j]*pow(samp2[j],j));
		}
		a[i]/=(T.at<float>(i,i)*pow(samp2[i],i));
	}
	cout<<"Coefficients: [ ";
	for( int i = 0; i<n; i++){
		cout<<a[i]<<" ";
	}
	cout<<"]"<<endl;

	for( int x = 0; x<dim[0]; x++){
		for( int y = 0; y<dim[1]; y++){
			float s2 = (x-c[0])*(x-c[0]) + ar*ar*(y-c[1])*(y-c[1]);
			float bg = 0;
			for( int m = n; m > 0; m--){
				bg*=s2;
				bg+=a[m-1];
			}
			flat[1].at<uchar>(y,x)=bg;
			flat[0].at<uchar>(y,x)=img.at<uchar>(y,x) - bg;
		}
	}
}

float transmittance(Mat img, float* r, float* c, float ar, float* a, int n){
	float s2 = (r[0]-c[0])*(r[0]-c[0]) + ar*ar*(r[1]-c[1])*(r[1]-c[1]);
	float base = 0;

	for( int i = n; i>0; i--){
		base*=s2;
		base+=a[i-1];
	}
	float value = img.at<uchar>(r[1],r[0]);

	cout<<"Background Value: "<<base<<endl;
	cout<<"Transmitted: "<<value<<endl;
	return value/base;
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
