#include "faf_fns.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
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

void dftProc(const Mat img, Mat& out){
	//DFT
	Mat padded;
	int ny = getOptimalDFTSize( img.rows );
	int nx = getOptimalDFTSize( img.cols );
	copyMakeBorder(img, padded, 0, ny-img.rows, 0, nx-img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded),Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);
	split(complexImg, planes);
	magnitude(planes[0],planes[1],planes[0]);
	Mat mag = planes[0];

	mag+= Scalar::all(1);
	log(mag, mag);

	mag= mag(Rect(0,0,mag.cols & -2, mag.rows & -2));
	
	int dftdim[] = {mag.cols,mag.rows};
	cout<<"DFT: "<<dftdim[0] << "x" << dftdim[1] <<endl;
	int cx = dftdim[0]/2;
	int cy = dftdim[1]/2;

	Mat q0(mag, Rect(0 ,0 ,cx,cy));
	Mat q1(mag, Rect(cx,0 ,cx,cy));
	Mat q2(mag, Rect(0 ,cy,cx,cy));
	Mat q3(mag, Rect(cx,cy,cx,cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0,255, NORM_MINMAX);
	out = mag;
}

void gaussian(Mat& out, double* dim, double var){
	double c[] = {dim[0]/2,dim[1]/2};
	double ar = dim[0]/dim[1];
	out = Mat::zeros(dim[1], dim[0], CV_32F);
	for( int x = 0; x<dim[0]; x++ ){
		for( int y = 0; y<dim[1]; y++){
			double s2 = (x-c[0])*(x-c[0]) + ar*ar*(y-c[1])*(y-c[1]);
			out.at<double>(y,x) = exp(-s2/var);
		}
	}
}

void butterworth(Mat& out, double* dim, double w0, int n){
	double c[] = {dim[0]/2,dim[1]/2};
	double ar = dim[0]/dim[1];
	out = Mat::zeros(dim[1], dim[0], CV_32F);
	for( int x = 0; x<dim[0]; x++ ){
		for( int y = 0; y<dim[1]; y++){
			double s2 = (x-c[0])*(x-c[0]) + ar*ar*(y-c[1])*(y-c[1]);
			out.at<double>(y,x) = 1/(1+pow(s2/pow(w0,2),n));
		}
	}
}

/* Flatten the background using elliptical symmetry and polynomial approximation
 * arguments:	Mat img the image to be flattened
 * 		Mat* flat an array to be contain the flattened image and the calculated background
 * 		double* a an array to hold the calculated coefficients of the background approximation
 * 		double* c the assumed centre of the background
 * 		double ar the assumed aspect ratio of the background
 * 		int n the number of samples to be taken. Should be less than 10.
 */
void flatten(/*const*/ Mat img, Mat* flat, double* a, double* c, double ar, int n){
	cout<<"Samples to be taken: "<<n<<endl;
	double dim[2]={(double) img.cols,(double) img.rows};
	double samp2[n]; //sample locations squared
	double b[1][n]; //sample values

	flat[0] = Mat::zeros(img.rows, img.cols, CV_8UC1);//init ``empty''
	flat[1] = Mat::zeros(img.rows, img.cols, CV_8UC1);//init ``empty''
	
	//determine locations s_i^2 to obtain samples
	int ra = max(c[0],abs(dim[0]-c[0]));
	int rb = max(c[1],abs(dim[1]-c[1]));
	samp2[0]=ra*ra+ar*ar*rb*rb;
	samp2[0]*=0.99;
	samp2[0]/=n;
	cout<<"Sample Separation: "<<samp2[0]<<endl;

	for( int i = 1; i<n; i++){
		samp2[i] = samp2[i-1]+samp2[0];
	}//populate s^2 values
	
	
	for( int i = 0; i < n; i++ ){
		double r[2] = {sqrt(samp2[i]), sqrt(samp2[i])/ar};
	       	vector<vector<int>> ellpts = ellipsePoints(c, r);

		vector<int> ellvals;
		for( auto & point:ellpts){
			if ( point.at(0)>=0 && point.at(1)>=0 && point.at(0)<dim[0] && point.at(1)<dim[1] ){
				int val = img.at<uchar>(point.at(1), point.at(0));
				ellvals.push_back(val);
				//img.at<uchar>(point.at(1), point.at(0)) = 255;
			}
		}
		cout<<endl;
		b[0][i]=findMedian(ellvals);
		cout<<"B"<<i<<": "<<b[0][i]<<endl;
	}//determine median b at each s^2
	cout<<"Samples Taken!"<<endl;
	
	//regression
	float Sarr[n][n];
	for( int i = 0; i < n; i++){
		for( int j=0; j < n; j++){
			if( j == 0 )
				Sarr[i][j]=1.0;
			else
				Sarr[i][j]=(i+1)*Sarr[i][j-1];
		}
	}//build S matrix
	Mat S(n,n, CV_32F, Sarr);
	cout<<"S="<<nl<<S<<endl;
	
	float Carr[n][n];
	for( int i = 0; i < n; i++ ){
		for( int j = 0; j < n; j++){
			if( j > i ){
				Carr[i][j]=0.0;
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
	}//build C matrix
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
	}//back substitute for coefficient values
	cout<<"Coefficients: [ ";
	for( int i = 0; i<n; i++){
		cout<<a[i]<<" ";
	}
	cout<<"]"<<endl;

	for( int x = 0; x<dim[0]; x++){
		for( int y = 0; y<dim[1]; y++){
			double s2 = (x-c[0])*(x-c[0]) + ar*ar*(y-c[1])*(y-c[1]);
			double bg = 0;
			for( int m = n; m > 0; m--){
				bg*=s2;
				bg+=a[m-1];
			}
			flat[1].at<uchar>(y,x)=bg;
			flat[0].at<uchar>(y,x)=img.at<uchar>(y,x) - bg;
		}
	}//return flat[1] which is the pure background, and flat[0] which is the background corrected image
}

/* From the background coefficients a[n], determines the background value and calculates the transmission of a point
 * arguments:	Mat img is the image to be analyzed
 * 		double* r is the location of the point to be analyzed
 * 		double* c is the centre of the background
 * 		double ar is the aspect ratio of the background
 * 		double* a is the array of background cefficients
 * 		int n is the size of the array a[n]
 */
double transmittance(Mat img, double* r, double* c, double ar, double* a, int n){
	double s2 = (r[0]-c[0])*(r[0]-c[0]) + ar*ar*(r[1]-c[1])*(r[1]-c[1]);
	double base = 0;

	for( int i = n; i>0; i--){
		base*=s2;
		base+=a[i-1];
	}
	double value = img.at<uchar>(r[1],r[0]);

	cout<<"Background Value: "<<base<<endl;
	cout<<"Transmitted: "<<value<<endl;
	return value/base;
}

/* Find the median value of a vector of values */
/* Taken from Matthew Fioravante's SO post*/
double findMedian(vector<int> &values){
	if(values.empty()){
		return 0.0;
	}
	auto n = values.size() / 2;
	std::nth_element(values.begin(), values.begin()+n, values.end());
	auto med = values[n];
	if(!(values.size() & 1)){
		auto max_it = max_element(values.begin(), values.begin()+n);
		med = (*max_it + med) / 2.0;
	}
	return med;
}

/* Use the midpoint algorithm to rasterize an ellipse and generate a list of points.
 * arguments: 	double array c[2] which denotes the centre of the ellipse
 * 		double array r[2] which denotes the length of the axes.
 */
vector<vector<int>> ellipsePoints(double* c, double* r){
	vector<vector<int>> ellipse;
	double x = 0, y = r[1];
	double dx=0, dy=0, d1=0, d2=0;

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

/* Attempt to find centre and aspect ratio by increasing contrast to create clear boundaries between intensities.
 * *DOES NOT WORK*
 * changing the contrast changes the centre, so likely not a good way to determine the centre
 */
void findCenter(Mat img, double* c){
	Mat center=Mat::zeros( img.size(), img.type() );//not sure if this line is used but too scared to delete it
	Mat shift=Mat::zeros( img.size(), img.type() );
	double avg = mean(img)[0];//average pixel value
	
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			shift.at<uchar>(y,x) = img.at<uchar>(y,x)-avg-4; //set shift to be the entire image shifted down more than the average.
		}
	}
	
	Mat thresh;
	threshold( shift, thresh, 200, 255, 0); //thresh_binary to make sharp edges to detect
	
	//morphology to make edge easier to detect
	Mat dilate;
	Mat const dilate_kernel = getStructuringElement(MORPH_RECT, Size(8,8));
	morphologyEx(thresh, dilate, MORPH_DILATE, dilate_kernel);

	Mat close;
	Mat const close_kernel = getStructuringElement(MORPH_RECT, Size(10,10));
	morphologyEx(thresh, close, MORPH_CLOSE, close_kernel);

	Mat erode;
	Mat const erode_kernel = getStructuringElement(MORPH_RECT, Size(10,10));
	morphologyEx(close, erode, MORPH_ERODE, erode_kernel);

	Mat open;
	Mat const open_kernel = getStructuringElement(MORPH_RECT, Size(7,7));
	morphologyEx(erode, open, MORPH_OPEN, open_kernel);
	
	
	Mat lap, abslap;
	Laplacian( open, lap, CV_16S, 1,1,0, BORDER_DEFAULT);
	convertScaleAbs(lap, abslap);
	
	//obtain contours of same intensity
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( abslap, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
	Mat drawing = Mat::zeros( abslap.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ ){
	        Scalar color = Scalar(0,255,25*i);
        	drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    	}

	namedWindow("Preview", WINDOW_NORMAL);
	imshow("Preview", drawing);
	int k = waitKey(0);
	if( k == 's' ){
	}
}

/* Increase contrast by using the linear transform y = a(x-b)
 * *DOES NOT WORK*
 * arguments:	Mat img the image to be modified
 * 		Mat contrast the location to store the new image
 * 		double a the scale factor
 * 		double b the offset factor
 */
void linearContrast(Mat img, Mat contrast, double a, double b){
	contrast=Mat::zeros( img.size(), img.type() );
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			contrast.at<uchar>(y,x) = a*(img.at<uchar>(y,x) - b);
		}
	}
}

/* Increase contrast by altering gamma value
 * *DOES NOT WORK*
 * arguments:	Mat img the image to imcrease contrast
 * 		Mat gamma the location to store the modified image
 * 		double g the gamma value
 */
void gammaContrast(Mat img, Mat gamma, double g){
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = saturate_cast<uchar>(pow(i/255, g)*255);
	}
	gamma = img.clone();
	LUT(img, lookUpTable, gamma);
}
