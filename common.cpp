#include "common.h"
#include <Eigen/Core>

using namespace Eigen;
typedef Matrix<short,Dynamic,1> VectorXs;

#include <iostream>
#include <cstdio>
#include <cstring>


// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];
int getColor( const unsigned char * c ){
	return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, int cc ){
	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const VectorXs & labeling, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c = colors[ labeling[k] ];
		putColor( r+3*k, c );
	}
	return r;
}
// Read the labeling from a file
VectorXs getLabeling( const unsigned char * im, int N, int M ){
	VectorXs res(N);
	for( int k=0; k<N; k++ ){
		// Map the color to a label
		int c = getColor( im + 3*k );
		int i;
		for( i=0;i<nColors && c!=colors[i]; i++ );
		if (c && i==nColors){
			if (i<M)
				colors[nColors++] = c;
			else
				c=0;
		}
		res[k] = c?i:-1;
	}
	return res;
}
//////////////////////////////////////////////////////////////////////////
int getColor4Depth( const unsigned char * c ){
	return c[0];
}
void putColor4Depth( unsigned char * c, int cc ){
	c[0] = cc; c[1] = cc; c[2] = cc;
}
VectorXs getLabelingDepth(const unsigned char* im, int N, int M){
	VectorXs res(N);

	for (int k = 0; k < N; k++){
		int c = getColor4Depth(im + 3*k);		
		res[k] = c;
	}
	return res;
}
unsigned char * colorize4Depth( const VectorXs & labeling, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c =  labeling[k];
		 putColor4Depth( r+3*k, c );
	}
	return r;
}
cv::Mat colorize4Parsing( const VectorXs & labeling, int W, int H ){
	cv::Mat resMat(H,W,CV_8UC3);

	for (int y = 0; y< H; y++)
		for (int x = 0; x < W; x++) {
			int k = resMat.cols * y + x;
			int c = colors[labeling[k]];
			resMat.data[resMat.channels()*(resMat.cols * y + x) + 0] = c&0xff;//R
			resMat.data[resMat.channels()*(resMat.cols * y + x) + 1] = (c>>8)&0xff;//G
			resMat.data[resMat.channels()*(resMat.cols * y + x) + 2] = (c>>16)&0xff;//B
		}
	return resMat;
}
