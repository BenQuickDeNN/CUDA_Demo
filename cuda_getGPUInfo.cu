#include "cuda_getGPUInfo.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void getGPUInfo();

/* 统计GPU信息 */
void getGPUInfo()
{
	printf("----------------统计GPU信息----------------\r\n");
	int dev = 0;				// 设备
	cudaDeviceProp devProp;		// 设备属性
	cudaGetDeviceProperties(&devProp, dev);
	printf("使用GPU device %d: %s\r\n", dev, devProp.name);
	printf("SM的数量: %d\r\n", devProp.multiProcessorCount);
	printf("每个线程块的共享内存大小: %f KB\r\n", devProp.sharedMemPerBlock / 1024.0);
	printf("每个线程块的最大线程数: %d\r\n", devProp.maxThreadsPerBlock);
	printf("每个SM的最大线程数: %d\r\n", devProp.maxThreadsPerMultiProcessor);
	printf("线程束大小: %d\r\n", devProp.warpSize);
	printf("每个SM的最大线程束数: %d\r\n", devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
	//printf("最大网格尺寸: %ld\r\n", devProp.maxGridSize);
}