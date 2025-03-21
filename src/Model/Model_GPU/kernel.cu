#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define p 128
#define BLOCK_SIZE 128

// __global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
// {
//         unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i >= n_particles ){
//                 return;
//         }
//         float3 acc_local ;
//         acc_local.x = 0.0;
//         acc_local.y = 0.0;
//         acc_local.z = 0.0;
//         for (int j = 0; j < n_particles; j++)
//         {

//                         const float diffx = positionsGPU[j].x - positionsGPU[i].x;
//                         const float diffy = positionsGPU[j].y - positionsGPU[i].y;
//                         const float diffz = positionsGPU[j].z - positionsGPU[i].z;
//                         float dij = diffx * diffx + diffy * diffy + diffz * diffz ;
//                         dij = max(1.0,dij);

//                         dij = rsqrtf(dij);
//                         dij = 10.0 * (dij * dij * dij);
                        

//                         acc_local.x += diffx * dij * positionsGPU[j].w;
//                         acc_local.y += diffy * dij * positionsGPU[j].w;
//                         acc_local.z += diffz * dij * positionsGPU[j].w;
                
//         }
//         accelerationsGPU[i].x = acc_local.x;
//         accelerationsGPU[i].y = acc_local.y;
//         accelerationsGPU[i].z = acc_local.z;
// }

__device__ float3
bodyBodyInteraction(float4 particules1, float4 particules2, float3 accel){
        float3 diff;
        diff.x = particules1.x - particules2.x;
        diff.y = particules1.y - particules2.y;
        diff.z = particules1.z - particules2.z;
        float dij = diff.x * diff.x  + diff.y  * diff.y  + diff.z  * diff.z  ;
        dij = max(1.0,dij);

        dij = rsqrtf(dij);
        dij = 10.0 * (dij * dij * dij);
        float s = dij * particules2.w;
        

        accel.x += diff.x  * s;
        accel.y += diff.y  * s;
        accel.z += diff.z  * s;
        return accel;

}
__device__ float3
tile_calculation(float4 myPosition, float3 accel){
        int i;
        extern __shared__ float4 shPosition[];
        for (i = 0; i < blockDim.x; i++){
                accel = bodyBodyInteraction(myPosition,shPosition[i],accel);
        }
        return accel;
}


__global__ void
calculate_forces(float4 * globalPosition, float4 * globalAcceleration, int n_particles){
        extern __shared__ float4 shPosition[];
        float4 myPosition;
        int i, tile;
        float3 acc = {0.0f, 0.0f, 0.0f};
        int gtid = blockIdx.x * blockDim.x + threadIdx.x;
        myPosition = globalPosition[gtid];
        for (i = 0, tile = 0 ; i < n_particles; i += p,tile++){
                int idx = tile * blockDim.x + threadIdx.x;
                shPosition[threadIdx.x] = globalPosition[idx];
                __syncthreads();
                acc = tile_calculation(myPosition, acc);
                __syncthreads();
        }
        
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        globalAcceleration[gtid] = acc4;
}

__global__ void maj_pos(float4 * positionsGPU, float4 * velocitiesGPU, float4 * accelerationsGPU,int n_particles)
{
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_particles ){
                return;
        }

        velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
        velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
        velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
        positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
        positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
        positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
        accelerationsGPU[i].x = 0.0f;
        accelerationsGPU[i].y = 0.0f;
        accelerationsGPU[i].z = 0.0f;

}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, int n_particles)
{
        int nthreads = BLOCK_SIZE;
        int nblocks =  (n_particles + (nthreads -1)) / nthreads;

        calculate_forces<<<nblocks, nthreads, p>>>(positionsGPU, accelerationsGPU,n_particles);
        maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU,n_particles);
}

#endif //GALAX_MODEL_GPU

