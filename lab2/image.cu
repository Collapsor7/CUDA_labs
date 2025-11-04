#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#define CSC(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ float gray_of(uchar4 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__ void roberts(cudaTextureObject_t tex, uchar4* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x1 = (x < w - 1) ? (x + 1) : (w - 1);
    int y1 = (y < h - 1) ? (y + 1) : (h - 1);

    float fx  = x  + 0.5f, fy  = y  + 0.5f;
    float fx1 = x1 + 0.5f, fy1 = y1 + 0.5f;

    uchar4 p11 = tex2D<uchar4>(tex, fx,  fy ); 
    uchar4 p21 = tex2D<uchar4>(tex, fx1, fy ); 
    uchar4 p12 = tex2D<uchar4>(tex, fx,  fy1); 
    uchar4 p22 = tex2D<uchar4>(tex, fx1, fy1); 

    float w11 = gray_of(p11);
    float w21 = gray_of(p21);
    float w12 = gray_of(p12);
    float w22 = gray_of(p22);

    float gx = w22 - w11;
    float gy = w21 - w12;

    float G = sqrtf(gx*gx + gy*gy);
    if (G > 255.0f) G = 255.0f;

    out[(size_t)y * (size_t)w + (size_t)x] =
        make_uchar4((unsigned char)G, (unsigned char)G, (unsigned char)G, 0);
}

int main(int argc, char** argv) {
    std::string in_path, out_path;
    std::cin >> in_path >> out_path;
    
    int n_block = std::atoi(argv[2]);
    int n_grid = std::atoi(argv[1]);

    FILE* fp = fopen(in_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", in_path.c_str());
        return 1;
    }
    int w = 0, h = 0;
    if (fread(&w, sizeof(int), 1, fp) != 1 ||
        fread(&h, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "bad header\n");
        fclose(fp);
        return 1;
    }
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "invalid size\n");
        fclose(fp);
        return 1;
    }
    size_t N = (size_t)w * (size_t)h;

    std::vector<uchar4> host(N);
    if (fread(host.data(), sizeof(uchar4), N, fp) != N) {
        fprintf(stderr, "bad pixel data\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    cudaArray_t arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, host.data(),
                            w * sizeof(uchar4),
                            w * sizeof(uchar4), h,
                            cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModePoint;
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; 

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    uchar4* dev_out = nullptr;
    CSC(cudaMalloc(&dev_out, N * sizeof(uchar4)));


    dim3 block(n_block,n_block);
    dim3 grid(n_grid,n_grid);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    roberts<<<grid, block>>>(tex, dev_out, w, h);
    
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    CSC(cudaMemcpy(host.data(), dev_out, N * sizeof(uchar4),
                   cudaMemcpyDeviceToHost));

    fp = fopen(out_path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", out_path.c_str());
        return 1;
    }
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(host.data(), sizeof(uchar4), N, fp);
    fclose(fp);

    fprintf(stderr,"Kernel exec:      %.6f s\n", kernel_ms);

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    return 0;
}