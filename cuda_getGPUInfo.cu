#include "cuda_getGPUInfo.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void getGPUInfo();

/* ͳ��GPU��Ϣ */
void getGPUInfo()
{
	printf("----------------ͳ��GPU��Ϣ----------------\r\n");
	int dev = 0;				// �豸
	cudaDeviceProp devProp;		// �豸����
	cudaGetDeviceProperties(&devProp, dev);
	printf("ʹ��GPU device %d: %s\r\n", dev, devProp.name);
	printf("SM������: %d\r\n", devProp.multiProcessorCount);
	printf("ÿ���߳̿�Ĺ����ڴ��С: %f KB\r\n", devProp.sharedMemPerBlock / 1024.0);
	printf("ÿ���߳̿������߳���: %d\r\n", devProp.maxThreadsPerBlock);
	printf("ÿ��SM������߳���: %d\r\n", devProp.maxThreadsPerMultiProcessor);
	printf("�߳�����С: %d\r\n", devProp.warpSize);
	printf("ÿ��SM������߳�����: %d\r\n", devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
	//printf("�������ߴ�: %ld\r\n", devProp.maxGridSize);
}