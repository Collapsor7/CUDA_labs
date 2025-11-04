// lab3.cu — 常量内存 + 一维线程网格 的图像分类 (Mahalanobis)
// 编译：nvcc -O2 -std=c++17 lab3.cu -o lab3
// 运行：echo -e "in.data\nout.data\n2\n4 1 2 1 0 2 2 2 1\n4 0 0 0 1 1 1 2 0" | ./lab3

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

// ---- 常量内存参数表（pc<=32） ----
__constant__ float c_avg[32*3]; // [pc][3]
__constant__ float c_inv[32*9]; // [pc][3x3]
__constant__ int   c_pc;        // 类别数

// ---- CUDA 错误检查 ----
#define CSC(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

// ---- 设备端：马氏距离 ----
__device__ inline float mahalanobis3(const float x[3], int j){
    const float* mu = &c_avg[j*3];
    float dx0 = x[0]-mu[0];
    float dx1 = x[1]-mu[1];
    float dx2 = x[2]-mu[2];
    const float* M = &c_inv[j*9]; // 行主序
    float y0 = M[0]*dx0 + M[1]*dx1 + M[2]*dx2;
    float y1 = M[3]*dx0 + M[4]*dx1 + M[5]*dx2;
    float y2 = M[6]*dx0 + M[7]*dx1 + M[8]*dx2;
    return dx0*y0 + dx1*y1 + dx2*y2; // d^2
}

// ---- 一维网格分类：把类号写入 alpha ----
__global__ void classify_kernel(uint8_t* rgba, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int p = idx * 4;
    float rgb[3] = { (float)rgba[p+0], (float)rgba[p+1], (float)rgba[p+2] };

    int best = 0; 
    float bestD = mahalanobis3(rgb, 0);
    for (int j = 1; j < c_pc; ++j){
        float d = mahalanobis3(rgb, j);
        if (d < bestD || (d == bestD && j < best)) { best = j; bestD = d; }
    }
    rgba[p+3] = (uint8_t)best;
}

// ======== 3×3 工具（裸数组，不用 struct） ========
inline void add_epsilonI(float* A, float eps){
    A[0]+=eps; A[4]+=eps; A[8]+=eps;
}
inline float det3(const float* A){
    return A[0]*(A[4]*A[8]-A[5]*A[7])
         - A[1]*(A[3]*A[8]-A[5]*A[6])
         + A[2]*(A[3]*A[7]-A[4]*A[6]);
}
inline void invert3(const float* A, float* Out){
    float d = det3(A);
    if (fabsf(d) < 1e-12f){
        Out[0]=1; Out[1]=0; Out[2]=0;
        Out[3]=0; Out[4]=1; Out[5]=0;
        Out[6]=0; Out[7]=0; Out[8]=1;
        return;
    }
    float id = 1.0f/d;
    Out[0] =  (A[4]*A[8]-A[5]*A[7]) * id;
    Out[1] = -(A[1]*A[8]-A[2]*A[7]) * id;
    Out[2] =  (A[1]*A[5]-A[2]*A[4]) * id;
    Out[3] = -(A[3]*A[8]-A[5]*A[6]) * id;
    Out[4] =  (A[0]*A[8]-A[2]*A[6]) * id;
    Out[5] = -(A[0]*A[5]-A[2]*A[3]) * id;
    Out[6] =  (A[3]*A[7]-A[4]*A[6]) * id;
    Out[7] = -(A[0]*A[7]-A[1]*A[6]) * id;
    Out[8] =  (A[0]*A[4]-A[1]*A[3]) * id;
}

// ======== 由样本坐标计算 avg 与 inv(cov)（CPU 侧） ========
static void compute_stats_from_coords(
    const std::vector<uchar4>& host, int W, int H,
    const std::vector<std::vector<int2>>& coords,
    std::vector<float>& h_avg,     // 输出: [pc*3]
    std::vector<float>& h_invCov   // 输出: [pc*9]
){
    const int pc = (int)coords.size();
    h_avg.assign(pc*3, 0.0f);
    h_invCov.assign(pc*9, 0.0f);

    for (int j = 0; j < pc; ++j) {
        const auto& C = coords[j];
        const int np = (int)C.size();

        // ---- 均值 ----
        long double sR=0, sG=0, sB=0;
        for (auto xy : C) {
            size_t idx = (size_t)xy.y * (size_t)W + (size_t)xy.x;
            const uchar4 p = host[idx];
            sR += p.x; sG += p.y; sB += p.z;
        }
        const long double n = std::max(1, np);
        const float muR = (float)(sR / n);
        const float muG = (float)(sG / n);
        const float muB = (float)(sB / n);
        h_avg[j*3+0] = muR;
        h_avg[j*3+1] = muG;
        h_avg[j*3+2] = muB;

        // ---- 协方差（样本协方差，分母 np-1）----
        long double C00=0, C01=0, C02=0, C11=0, C12=0, C22=0;
        for (auto xy : C) {
            size_t idx = (size_t)xy.y * (size_t)W + (size_t)xy.x;
            const uchar4 p = host[idx];
            const long double d0 = (long double)p.x - muR; // R
            const long double d1 = (long double)p.y - muG; // G
            const long double d2 = (long double)p.z - muB; // B
            C00 += d0*d0; C01 += d0*d1; C02 += d0*d2;
            C11 += d1*d1; C12 += d1*d2;
            C22 += d2*d2;
        }
        const float denom = (np > 1) ? (float)(np - 1) : 1.0f;

        float cov[9];
        cov[0] = (float)(C00/denom);
        cov[1] = (float)(C01/denom);
        cov[2] = (float)(C02/denom);
        cov[3] = cov[1];
        cov[4] = (float)(C11/denom);
        cov[5] = (float)(C12/denom);
        cov[6] = cov[2];
        cov[7] = cov[5];
        cov[8] = (float)(C22/denom);

        // 正则化 + 求逆 → 写入 h_invCov[j]
        add_epsilonI(cov, 1e-3f);
        float inv[9];
        invert3(cov, inv);
        for (int k = 0; k < 9; ++k)
            h_invCov[j*9 + k] = inv[k];
    }
}

// ======== 上传至常量内存 ========
static void upload_constants_from_host(
    int pc, const std::vector<float>& h_avg, const std::vector<float>& h_inv)
{
    CSC(cudaMemcpyToSymbol(c_pc,  &pc,             sizeof(int)));
    CSC(cudaMemcpyToSymbol(c_avg, h_avg.data(), pc*3*sizeof(float)));
    CSC(cudaMemcpyToSymbol(c_inv, h_inv.data(), pc*9*sizeof(float)));
}

int main() {
    // ---------- 读 in/out 路径 ----------
    std::string in_path, out_path;
    if (!(std::cin >> in_path >> out_path)) {
        std::fprintf(stderr, "usage: <in.data> <out.data> then pc/coords on stdin\n");
        return 1;
    }

    // ---------- 读 in.data ----------
    FILE* fp = std::fopen(in_path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }

    int32_t W=0, H=0;
    if (std::fread(&W, sizeof(int32_t), 1, fp) != 1 ||
        std::fread(&H, sizeof(int32_t), 1, fp) != 1) {
        std::fprintf(stderr, "bad header\n"); std::fclose(fp); return 1;
    }
    if (W <= 0 || H <= 0) {
        std::fprintf(stderr, "invalid size\n"); std::fclose(fp); return 1;
    }
    size_t N = (size_t)W * (size_t)H;
    std::vector<uchar4> host(N);
    if (std::fread(host.data(), sizeof(uchar4), N, fp) != N) {
        std::fprintf(stderr, "bad pixel data\n"); std::fclose(fp); return 1;
    }
    std::fclose(fp);

    // ---------- 读 pc 与样本坐标 ----------
    int pc = 0; if (!(std::cin >> pc) || pc <= 0 || pc > 32) { std::fprintf(stderr, "bad pc\n"); return 1; }
    std::vector<std::vector<int2>> coords(pc);
    for (int j = 0; j < pc; ++j) {
        int np = 0; if (!(std::cin >> np) || np <= 0) { std::fprintf(stderr, "bad np at class %d\n", j); return 1; }
        coords[j].reserve(np);
        for (int i = 0; i < np; ++i) {
            int x, y; if (!(std::cin >> x >> y)) { std::fprintf(stderr, "bad coord #%d class %d\n", i, j); return 1; }
            if (x<0 || x>=W || y<0 || y>=H) { std::fprintf(stderr, "coord OOB (%d,%d) class %d\n", x, y, j); return 1; }
            coords[j].push_back(make_int2(x,y));
        }
    }

    // ---------- 统计 & 上传常量 ----------
    std::vector<float> h_avg, h_inv;
    compute_stats_from_coords(host, W, H, coords, h_avg, h_inv);
    upload_constants_from_host(pc, h_avg, h_inv);

    // ---------- 拷图像到 GPU ----------
    uint8_t* d_rgba = nullptr;
    CSC(cudaMalloc(&d_rgba, N * 4));
    CSC(cudaMemcpy(d_rgba, host.data(), N * 4, cudaMemcpyHostToDevice));

    // ---------- 一维网格分类 ----------
    int block = THREADS_PER_BLOCK;
    int grid  = (int)((N + block - 1) / block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    classify_kernel<<<grid, block>>>(d_rgba, (int)N);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
        fprintf(stderr,"Kernel exec:      %.6f s\n", kernel_ms);


    // ---------- 拷回并写 out.data ----------
    CSC(cudaMemcpy(host.data(), d_rgba, N * 4, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_rgba));

    FILE* fo = std::fopen(out_path.c_str(), "wb");
    if (!fo) { std::fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    std::fwrite(&W, sizeof(int32_t), 1, fo);
    std::fwrite(&H, sizeof(int32_t), 1, fo);
    std::fwrite(host.data(), sizeof(uchar4), N, fo);
    std::fclose(fo);

    std::fprintf(stderr, "Done. W=%d H=%d pc=%d (alpha=class id)\n", W, H, pc);
    return 0;
}
