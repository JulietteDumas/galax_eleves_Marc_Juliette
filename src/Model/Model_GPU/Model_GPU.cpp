#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>
#include <cstring>

#include "Model_GPU.hpp"
#include "kernel.cuh"


inline bool cuda_malloc(void ** devPtr, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to allocate buffer" << std::endl;
		return false;
	}
	return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst, src, count, kind);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to copy buffer" << std::endl;
		return false;
	}
	return true;
}

void update_position_gpu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU,  int n_particles)
{
	update_position_cu(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}


Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  positionsf4    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;

	for (int i = 0; i < n_particles; i++)
	{
		positionsf4[i].x     = initstate.positionsx [i];
		positionsf4[i].y     = initstate.positionsy [i];
		positionsf4[i].z     = initstate.positionsz [i];
		positionsf4[i].w     = initstate.masses	[i];
		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
		accelerationsf3[i].x = 0;
		accelerationsf3[i].y = 0;
		accelerationsf3[i].z = 0;
	}

	cuda_malloc((void**)&positionsGPU,     n_particles * sizeof(float4));
	cuda_malloc((void**)&velocitiesGPU,     n_particles * sizeof(float3));
	cuda_malloc((void**)&accelerationsGPU,     n_particles * sizeof(float3));



	cuda_memcpy(positionsGPU,  positionsf4.data()     , n_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cuda_memcpy(velocitiesGPU,  velocitiesf3.data()     , n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	cuda_memcpy(accelerationsGPU,  accelerationsf3.data()     , n_particles * sizeof(float3), cudaMemcpyHostToDevice);

}

Model_GPU
::~Model_GPU()
{
	cudaFree((void**)&positionsGPU);
	cudaFree((void**)&velocitiesGPU);
	cudaFree((void**)&accelerationsGPU);
}

void Model_GPU
::step()
{	
	update_position_gpu(positionsGPU,velocitiesGPU,accelerationsGPU,n_particles);
	cuda_memcpy(positionsf4.data(), positionsGPU, n_particles * sizeof(float4), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = positionsf4[i].x;
		particles.y[i] = positionsf4[i].y;
		particles.z[i] = positionsf4[i].z;
	}
}

#endif // GALAX_MODEL_GPU
