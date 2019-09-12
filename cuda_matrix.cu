#include "cuda_matrix.h"

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
* @brief 自定义Host端矩阵
* 在host端使用
* 遇到返回代码为-1073741818的未知错误，暂时不用抽象的类
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
	* @brief 获取矩阵宽度
	* @return 矩阵的宽度
	*/
	int getWidth()
	{
		return width;
	}
	/**
	* @brief 获取矩阵高度
	* @return 矩阵的高度
	*/
	int getHeight()
	{
		return height;
	}
	/**
	* @brief 获取矩阵元素
	* @return 矩阵元素
	*/
	float* getElements()
	{
		return elements;
	}
	/**
	* @brief 括号运算符重载
	* @param i 行索引
	* @param j 列索引
	* @return 对应位置的矩阵元素
	*/
	float& operator()(int i, int j)
	{
		return elements[i * width + j];
	}
	/**
	* @brief 构造函数
	* @param h 矩阵高度
	* @param w 矩阵宽度
	* 申请Host端内存空间
	*/
	Matrix(int h, int w)
	{
		height = h;
		width = w;
		elements = (float*)malloc(h * w * sizeof(float));
	}
	/**
	* @brief 析构函数
	* 释放Host端内存空间
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
	* @brief 获取矩阵高度
	* @return 矩阵的高度
	*/
	__device__ int getHeight()
	{
		return height;
	}
	/**
	* @brief 获取矩阵宽度
	* @return 矩阵的宽度
	*/
	__device__ int getWidth()
	{
		return width;
	}
	/**
	* @brief 获取矩阵元素
	* @return 矩阵元素
	*/
	float* getElements()
	{
		return elements;
	}
	/**
	* @brief 括号运算符重载
	* @param i 行索引
	* @param j 列索引
	* @return 对应位置的矩阵元素
	*/
	__device__ float& operator()(int i, int j)
	{
		return elements[i * width + j];
	}
	/**
	* @brief 从Host拷贝数据到Device
	* @param mat Host端矩阵
	*/
	void memCpyFrom(Matrix mat)
	{
		cudaMemcpy((void*)elements, (void*)mat.getElements(), mat.getHeight() * mat.getWidth(), cudaMemcpyHostToDevice);
	}
	/**
	* @brief 构造函数
	* @param h 矩阵高度
	* @param w 矩阵宽度
	* 申请Device端内存空间
	*/
	CUDAMatrix(int h, int w)
	{
		height = h;
		width = w;
		cudaMalloc((void**)&elements, h * w * sizeof(float));
	}
	/**
	* @brief 析构函数
	* 释放Device端内存空间
	*/
	~CUDAMatrix()
	{

		cudaFree(elements);
	}
};

/**
* @brief 核函数-矩阵乘
* @param A 输入矩阵
* @param B 输入矩阵
* @param C 输出矩阵
* @param Aw 矩阵A的宽度
* @param Bw 矩阵B的宽度
* @param Cw 矩阵C的宽度
* 注意矩阵A的宽度要与矩阵B的高度相等
*/
__global__ void cuda_matrix_mul(float *A, float *B, float *C, int Aw, int Bw, int Cw)
{
	// 计算元素的绝对位置
	int row = threadIdx.y + blockIdx.y * blockDim.y; // 行
	int col = threadIdx.x + blockIdx.x * blockDim.x; // 列
	int indexC = row * Cw + col;
	int indexA = row * Aw;
	/* 每个线程计算单个元素 */
	//C(row, col) = 0.0;
	for (int i = 0; i < Aw; i++)
	{
		C[indexC] += A[indexA + i] * B[i * Bw + col];
	}
}

/**
* @brief Host端函数-矩阵乘
*/
void matrix_mul()
{
	printf("----------------计算矩阵乘----------------\r\n");
	int Ah = 1 << 10; // A矩阵高度
	int Aw = 1 << 10; // A矩阵宽度
	int Bh = Aw; // B矩阵高度
	int Bw = 1 << 10; // B矩阵宽度
	int Ch = Ah; // C矩阵高度
	int Cw = Bw; // C矩阵宽度
	/* Host端申请空间 */
	// Matrix A(Ah, Aw), B(Bh, Bw), C(Ch, Cw);
	float *A, *B, *C;
	int lenA = Ah * Aw * sizeof(float);
	int lenB = Bh * Bw * sizeof(float);
	int lenC = Ch * Cw * sizeof(float);
	A = (float*)malloc(lenA);
	B = (float*)malloc(lenB);
	C = (float*)malloc(lenC);
	/* 给A、B赋初值 */
	for (int i = 0; i < Ah; i++)
		for (int j = 0; j < Aw; j++)
			A[i * Aw + j] = 2.0;
	for (int i = 0; i < Bh; i++)
		for (int j = 0; j < Bw; j++)
			B[i * Bw + j] = 3.0;
	/* Device端申请空间 */
	//CUDAMatrix cA(Ah, Aw), cB(Bh, Bw), cC(Ch, Cw);
	float *cA, *cB, *cC;
	cudaMalloc((void**)&cA, lenA);
	cudaMalloc((void**)&cB, lenB);
	cudaMalloc((void**)&cC, lenC);
	/* 将数据从Host拷入Device */
	//cA.memCpyFrom(A);
	//cB.memCpyFrom(B);
	cudaMemcpy((void*)cA, (void*)A, lenA, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)cB, (void*)B, lenB, cudaMemcpyHostToDevice);
	/* 定义kernel */
	dim3 blockSize(32, 32);
	dim3 gridSize((Ch + blockSize.x - 1) / blockSize.x, (Cw + blockSize.y - 1) / blockSize.y);
	printf("blockSize.x = %d, blockSize.y = %d\r\n", blockSize.x, blockSize.y);
	printf("gridSize.x = %d, gridSize.y = %d\r\n", gridSize.x, gridSize.y);
	/* 执行计算 */
	printf("开始内核计算...\r\n");
	cuda_matrix_mul << <gridSize, blockSize >> > (cA, cB, cC, Aw, Bw, Cw);
	printf("完成内核计算！\r\n");
	/* 将数据从Device拷贝回Host */
	cudaMemcpy((void*)C, (void*)cC, lenC, cudaMemcpyDeviceToHost);
	// 检查结果
	printf("计算误差平方和...\r\n");
	float maxError = 0.0;
	for (int i = 0; i < Ch; i++)
		for (int j = 0; j < Cw; j++)
			maxError += (C[i * Cw + j] - 2.0 * 3.0 * Aw) * (C[i * Cw + j] - 2.0 * 3.0 * Aw);
	printf("误差平方和: %f\r\n", maxError);
}