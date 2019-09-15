#include "cuda_vector.h"

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void vector_add();

/**
* @brief �˺���-������
* @param x ��������
* @param y ��������
* @param z �������
* @param n ��������
*/
__global__ void cuda_vector_add(float *x, float *y, float *z, int n)
{
	// ��ȡȫ�����������������������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ��ȡ������һ��������ĳ��ȣ�
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) z[i] = x[i] + y[i];
}

/**
* @brief Host�˺���-������
*/
void vector_add()
{
	printf("----------------����������----------------\r\n");
	// ��������
	int n = 1 << 24;
	int nBytes = n * sizeof(float);
	// ����host�ڴ�
	float *x, *y, *z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);
	// ���ݳ�ʼ��
	for (int i = 0; i < n; i++)
	{
		x[i] = 10.0;
		y[i] = 20.0;
	}
	// ����device�ڴ�
	float *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, nBytes);
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);
	// �����ݴ�host������device
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
	// ����kernel
	dim3 blockSize(256); // GTX 710 block���̸߳������ܳ���1024
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x); // gridSize��block��������������
	printf("blockSize = %d\r\n", blockSize.x);
	printf("gridSize = %d\r\n", gridSize.x);
	// ִ�м���
	cuda_vector_add << < gridSize, blockSize >> > (d_x, d_y, d_z, n);
	// �����ݴ�device������host
	cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
	// �����
	float maxError = 0.0;
	for (int i = 0; i < n; i++) maxError += (z[i] - 30.0) * (z[i] - 30.0);
	printf("���ƽ����: %f\r\n", maxError);
	// �ͷ�device�ڴ�
	cudaFree(d_z);
	cudaFree(d_y);
	cudaFree(d_x);
	// �ͷ�host�ڴ�
	free(z);
	free(y);
	free(x);
}