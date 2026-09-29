/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
#include <iostream>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements. This exmple shows Vector addition using Unified memory.
 */


__global__ void vecAdd(float4* A, float4* B, float4* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength) {
        C[workIndex].x = A[workIndex].x + B[workIndex].x;
        C[workIndex].y = A[workIndex].y + B[workIndex].y;
        C[workIndex].z = A[workIndex].z + B[workIndex].z;
        C[workIndex].w = A[workIndex].w + B[workIndex].w;
    }
}

void initArray(float* A, int length)
{
    std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    int vectorLengthReal = 1024;
    if(argc >=2)
    {
        vectorLengthReal = std::atoi(argv[1]);
    }
    int vectorLengthSupple = (vectorLengthReal + 3) / 4 * 4;


    //unified-memory-example-begin

    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLengthSupple*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLengthSupple*sizeof(float));
    cudaMallocManaged(&B, vectorLengthSupple*sizeof(float));
    cudaMallocManaged(&C, vectorLengthSupple*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLengthSupple);
    initArray(B, vectorLengthSupple);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLengthSupple / 4, threads);
    float averageTime = 0;
    int countExperements = 100;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float4* ACust = reinterpret_cast<float4*>(A);
    float4* BCust = reinterpret_cast<float4*>(B);
    float4* CCust = reinterpret_cast<float4*>(C);
    for (int i = 0; i < countExperements; ++i)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        cudaEventRecord(start, stream);
        vecAdd<<<blocks, threads, 0, stream>>>(ACust, BCust, CCust, vectorLengthSupple / 4);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        // Wait for the kernel to complete execution

        float timeWork;
        cudaEventElapsedTime(&timeWork, start, stop);
        averageTime += timeWork / countExperements;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaStreamDestroy(stream);
    double bytes = 3.0 * vectorLengthReal * sizeof(float);
    double gbps  = bytes / (averageTime * 1e6);
    std::cout << "time: " << averageTime << " ms" << std::endl;
    std::cout << "bandwidth: " << gbps << " GB/s" << std::endl;
    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLengthReal);

    // Confirm that CPU and GPU got the same answer
    C = reinterpret_cast<float*>(CCust);
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLengthReal))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

    //unified-memory-example-end

    return 0;
}
