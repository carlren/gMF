#pragma once

#include "../ORUtils/PlatformIndependence.h"
#include "../ORUtils/Vector.h"
#include "../ORUtils/Matrix.h"
#include "../ORUtils/Image.h"
#include "../ORUtils/MathUtils.h"
#include "../ORUtils/MemoryBlock.h"

//------------------------------------------------------
// 
// Compile time GPU Settings, don't touch it!
//
//------------------------------------------------------

#define BLOCK_DIM				16
#define HALF_BLOCK_DIM			(BLOCK_DIM / 2)
#define BLOCK_TOTAL_SIZE		(BLOCK_DIM * BLOCK_DIM)

#define PER_THREAD_RADIUS		8
#define PER_THREAD_SEARCH_SIZE	(PER_THREAD_RADIUS * 2)
#define PER_THREAD_SEARCH_AREA	(PER_THREAD_SEARCH_SIZE * PER_THREAD_SEARCH_AREA)

#define MAX_SAMPLE_1D			16
#define MAX_SAMPLE_TOTAL		((MAX_SAMPLE_1D + 1)*(MAX_SAMPLE_1D +1))

#define MIN_SAMPLE_1D			8
#define MIN_SAMPLE_TOTAL		((MIN_SAMPLE_1D + 1)*(MIN_SAMPLE_1D +1))

#define MAX_RGB_SEARCH_RANGE	256
#define MAX_XY_SEARCH_RANGE		256

#define MAX_FILTER_DIM			21
#define MAX_HIERARCHY_LEVEL		3
#define HIERARCHY_FACTOR		2
#define SUBSAMPLE_FACTOR		2

//------------------------------------------------------
// 
// Compile time GPU Settings for MF Bilateral grid
//
//------------------------------------------------------

#define MIN_STD_DEV_XY			32
#define MIN_STD_DEV_RGB			3


namespace gMF
{
	//------------------------------------------------------
	// 
	// math defines
	//
	//------------------------------------------------------

	typedef unsigned char uchar;
	typedef unsigned short ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

	typedef class ORUtils::Matrix3<float> Matrix3f;
	typedef class ORUtils::Matrix4<float> Matrix4f;

	typedef class ORUtils::Vector2<short> Vector2s;
	typedef class ORUtils::Vector2<int> Vector2i;
	typedef class ORUtils::Vector2<float> Vector2f;
	typedef class ORUtils::Vector2<double> Vector2d;

	typedef class ORUtils::Vector3<short> Vector3s;
	typedef class ORUtils::Vector3<double> Vector3d;
	typedef class ORUtils::Vector3<int> Vector3i;
	typedef class ORUtils::Vector3<uint> Vector3ui;
	typedef class ORUtils::Vector3<uchar> Vector3u;
	typedef class ORUtils::Vector3<float> Vector3f;

	typedef class ORUtils::Vector4<float> Vector4f;
	typedef class ORUtils::Vector4<int> Vector4i;
	typedef class ORUtils::Vector4<short> Vector4s;
	typedef class ORUtils::Vector4<uchar> Vector4u;

	//------------------------------------------------------
	// 
	// image defines
	//
	//------------------------------------------------------

	typedef  ORUtils::Image<Vector3f> Float3Image;
	typedef	 ORUtils::Image<Vector2f> Float2Image;
	typedef  ORUtils::Image<Vector3u> UChar3Image;
	typedef  ORUtils::Image<int> IntImage;
	

	//------------------------------------------------------
	// 
	// Other defines
	//
	//------------------------------------------------------

	typedef ORUtils::MemoryBlock<float> FloatArray;
	typedef ORUtils::MemoryBlock<Vector2i> Point2iArray;

}


#ifndef DEBUGBREAK
#define DEBUGBREAK \
	{ \
	int ryifrklaeybfcklarybckyar=0; \
	ryifrklaeybfcklarybckyar++; \
	}
#endif