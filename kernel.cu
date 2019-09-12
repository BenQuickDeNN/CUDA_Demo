#include "cuda_getGPUInfo.h"
#include "cuda_vector.h"
#include "cuda_matrix.h"

int main()
{
	getGPUInfo();
	vector_add();
	matrix_mul();

    return 0;
}