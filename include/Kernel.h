#pragma once

#include "permutohedral.h"

#include <Eigen/Core>

using namespace Eigen;

typedef Matrix<short,Dynamic,1> VectorXs;



class DenseKernel {

protected:

		MatrixXf f_;

		Permutohedral lattice_;

		VectorXf norm_;

public:



	~DenseKernel(){};

	void apply( MatrixXf & out, const MatrixXf & Q ) const;

	void initLattice( const MatrixXf & f );

	void filter( MatrixXf & out, const MatrixXf & in) const;

	

	DenseKernel(const MatrixXf & f);

};
