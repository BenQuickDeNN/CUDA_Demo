#include "cuda_matrix.h"

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
* @brief �Զ���Host�˾���
* ��host��ʹ��
* �������ش���Ϊ-1073741818��δ֪������ʱ���ó������
*/
class Matrix
{
protected:
	Matrix() {}
	int width;
	int height;
	float *elements;
public:
	/**
	* @brief ��ȡ������
	* @return ����Ŀ��
	*/
	int getWidth()
	{
		return width;
	}
	/**
	* @brief ��ȡ����߶�
	* @return ����ĸ߶�
	*/
	int getHeight()
	{
		return height;
	}
	/**
	* @brief ��ȡ����Ԫ��
	* @return ����Ԫ��
	*/
	float* getElements()
	{
		return elements;
	}
	/**
	* @brief �������������
	* @param i ������
	* @param j ������
	* @return ��Ӧλ�õľ���Ԫ��
	*/
	float& operator()(int i, int j)
	{
		return elements[i * width + j];
	}
	/**
	* @brief ���캯��
	* @param h ����߶�
	* @param w ������
	* ����Host���ڴ�ռ�
	*/
	Matrix(int h, int w)
	{
		height = h;
		width = w;
		elements = (float*)malloc(h * w * sizeof(float));
	}
	/**
	* @brief ��������
	* �ͷ�Host���ڴ�ռ�
	*/
	~Matrix()
	{
		free(elements);
	}
};

class CUDAMatrix:Matrix
{
public:
	/**
	* @brief ��ȡ����߶�
	* @return ����ĸ߶�
	*/
	__device__ int getHeight()
	{
		return height;
	}
	/**
	* @brief ��ȡ������
	* @return ����Ŀ��
	*/
	__device__ int getWidth()
	{
		return width;
	}
	/**
	* @brief ��ȡ����Ԫ��
	* @return ����Ԫ��
	*/
	float* getElements()
	{
		return elements;
	}
	/**
	* @brief �������������
	* @param i ������
	* @param j ������
	* @return ��Ӧλ�õľ���Ԫ��
	*/
	__device__ float& operator()(int i, int j)
	{
		return elements[i * width + j];
	}
	/**
	* @brief ��Host�������ݵ�Device
	* @param mat Host�˾���
	*/
	void memCpyFrom(Matrix mat)
	{
		cudaMemcpy((void*)elements, (void*)mat.getElements(), mat.getHeight() * mat.getWidth(), cudaMemcpyHostToDevice);
	}
	/**
	* @brief ���캯��
	* @param h ����߶�
	* @param w ������
	* ����Device���ڴ�ռ�
	*/
	CUDAMatrix(int h, int w)
	{
		height = h;
		width = w;
		cudaMalloc((void**)&elements, h * w * sizeof(float));
	}
	/**
	* @brief ��������
	* �ͷ�Device���ڴ�ռ�
	*/
	~CUDAMatrix()
	{

		cudaFree(elements);
	}
};

/**
* @brief �˺���-�����
* @param A �������
* @param B �������
* @param C �������
* @param Aw ����A�Ŀ��
* @param Bw ����B�Ŀ��
* @param Cw ����C�Ŀ��
* ע�����A�Ŀ��Ҫ�����B�ĸ߶����
*/
__global__ void cuda_matrix_mul(float *A, float *B, float *C, int Aw, int Bw, int Cw)
{
	// ����Ԫ�صľ���λ��
	int row = threadIdx.y + blockIdx.y * blockDim.y; // ��
	int col = threadIdx.x + blockIdx.x * blockDim.x; // ��
	int indexC = row * Cw + col;
	int indexA = row * Aw;
	/* ÿ���̼߳��㵥��Ԫ�� */
	//C(row, col) = 0.0;
	for (int i = 0; i < Aw; i++)
	{
		C[indexC] += A[indexA + i] * B[i * Bw + col];
	}
}

/**
* @brief Host�˺���-�����
*/
void matrix_mul()
{
	printf("----------------��������----------------\r\n");
	int Ah = 1 << 10; // A����߶�
	int Aw = 1 << 10; // A������
	int Bh = Aw; // B����߶�
	int Bw = 1 << 10; // B������
	int Ch = Ah; // C����߶�
	int Cw = Bw; // C������
	/* Host������ռ� */
	// Matrix A(Ah, Aw), B(Bh, Bw), C(Ch, Cw);
	float *A, *B, *C;
	int lenA = Ah * Aw * sizeof(float);
	int lenB = Bh * Bw * sizeof(float);
	int lenC = Ch * Cw * sizeof(float);
	A = (float*)malloc(lenA);
	B = (float*)malloc(lenB);
	C = (float*)malloc(lenC);
	/* ��A��B����ֵ */
	for (int i = 0; i < Ah; i++)
		for (int j = 0; j < Aw; j++)
			A[i * Aw + j] = 2.0;
	for (int i = 0; i < Bh; i++)
		for (int j = 0; j < Bw; j++)
			B[i * Bw + j] = 3.0;
	/* Device������ռ� */
	//CUDAMatrix cA(Ah, Aw), cB(Bh, Bw), cC(Ch, Cw);
	float *cA, *cB, *cC;
	cudaMalloc((void**)&cA, lenA);
	cudaMalloc((void**)&cB, lenB);
	cudaMalloc((void**)&cC, lenC);
	/* �����ݴ�Host����Device */
	//cA.memCpyFrom(A);
	//cB.memCpyFrom(B);
	cudaMemcpy((void*)cA, (void*)A, lenA, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)cB, (void*)B, lenB, cudaMemcpyHostToDevice);
	/* ����kernel */
	dim3 blockSize(32, 32);
	dim3 gridSize((Ch + blockSize.x - 1) / blockSize.x, (Cw + blockSize.y - 1) / blockSize.y);
	printf("blockSize.x = %d, blockSize.y = %d\r\n", blockSize.x, blockSize.y);
	printf("gridSize.x = %d, gridSize.y = %d\r\n", gridSize.x, gridSize.y);
	/* ִ�м��� */
	printf("��ʼ�ں˼���...\r\n");
	cuda_matrix_mul << <gridSize, blockSize >> > (cA, cB, cC, Aw, Bw, Cw);
	printf("����ں˼��㣡\r\n");
	/* �����ݴ�Device������Host */
	cudaMemcpy((void*)C, (void*)cC, lenC, cudaMemcpyDeviceToHost);
	// �����
	printf("�������ƽ����...\r\n");
	float maxError = 0.0;
	for (int i = 0; i < Ch; i++)
		for (int j = 0; j < Cw; j++)
			maxError += (C[i * Cw + j] - 2.0 * 3.0 * Aw) * (C[i * Cw + j] - 2.0 * 3.0 * Aw);
	printf("���ƽ����: %f\r\n", maxError);
}