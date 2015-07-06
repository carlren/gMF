#include <Eigen/Core>
using namespace Eigen;
typedef Matrix<short,Dynamic,1> VectorXs;

#include "Kernel.h"


void DenseKernel::initLattice( const MatrixXf & f ) {
	const int N = f.cols();
	lattice_.init( f );

	norm_ = lattice_.compute( VectorXf::Ones( N ).transpose() ).transpose();

	for ( int i=0; i<N; i++ )
		norm_[i] = 1.0 / (norm_[i]+1e-20);
}

void DenseKernel::filter( MatrixXf & out, const MatrixXf & in ) const {
		// Read in the values
		out = in;

		// Filter
		lattice_.compute( out, out );
		// 			lattice_.compute( out.data(), out.data(), out.rows() );

		// Normalize again
		out = out*norm_.asDiagonal();
}

DenseKernel::DenseKernel(const MatrixXf & f):f_(f){
	initLattice( f );
}

void DenseKernel::apply( MatrixXf & out, const MatrixXf & Q ) const {
	filter( out, Q );
}
