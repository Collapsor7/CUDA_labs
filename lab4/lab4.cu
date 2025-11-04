#include<cstdio>
#include<cmath>
#include<iostream>
#include<thrust/device_ptr.h>
#include<thrust/extrema.h>
using namespace std;

// |A[i,k]| for i>=k -> buf[i-k]
__global__ void extractAbsColFromK(const double* A, int n, int k, double* buf){
  int i = k + blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) buf[i-k] = fabs(A[(size_t)i*n + k]);
}

__global__ void swapRowsFromK(double* A, int n, int r1, int r2, int k){
  int j = k + blockDim.x * blockIdx.x + threadIdx.x;
  if (j < n){
    double t = A[(size_t)r1*n + j];
    A[(size_t)r1*n + j] = A[(size_t)r2*n + j];
    A[(size_t)r2*n + j] = t;
  }
}

// m[i] = A[i,k]/A[k,k], i>k; and set A[i,k]=0
__global__ void buildMultipliers(double* A, int n, int k, double* m){
  int i = k + 1 + blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n){
    double piv = A[(size_t)k*n + k];
    double mi  = A[(size_t)i*n + k] / piv;
    m[i] = mi;
    A[(size_t)i*n + k] = 0.0;
  }
}
// A[i,j] -= m[i] * A[k,j], i>k, j>k  (2D grid)
__global__ void eliminate(double* A, int n, int k, const double* __restrict__ m){
  int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && j < n){
    A[(size_t)i*n + j] -= m[i] * A[(size_t)k*n + j];
  }
}

int main(){
  int n;
  scanf("%d", &n);
  size_t N = (size_t)n * (size_t)n;
  double *hA;
  hA = (double*)malloc(sizeof(double) * N);

  for (int i = 0; i < n;i++)
  for (int j = 0; j < n;j++){
    scanf("%lf", &hA[(size_t)i * n + j]);
  }

  double *dA, *dAbsCol, *dMul;
  cudaMalloc(&dA, sizeof(double) * N);
  cudaMalloc(&dAbsCol, sizeof(double) * n);
  cudaMalloc(&dMul, sizeof(double) * n);

  cudaMemcpy(dA, hA, sizeof(double) * N, cudaMemcpyHostToDevice);


  double EPS = 1e-7;
  int swaps = 0;

  for (int k = 0; k < n;k++){
    int rest = n - k;
    int tpb = 256, bl = (rest + tpb - 1) / tpb;
    extractAbsColFromK<<<bl, tpb>>>(dA, n, k, dAbsCol);
    cudaGetLastError();
    cudaDeviceSynchronize();

    thrust::device_ptr<double> beg(dAbsCol);
    thrust::device_ptr<double> end(dAbsCol + rest);
    auto it = thrust::max_element(beg, end);
    int pivotOff = (int)(it - beg);
    int piv = k + pivotOff;

    double pivAbs;
    cudaMemcpy(&pivAbs, dAbsCol + pivotOff, sizeof(double), cudaMemcpyDeviceToHost);
    if (pivAbs < EPS){
      printf("%.10e\n", 0.0);
      cudaFree(dA); 
      cudaFree(dAbsCol); 
      cudaFree(dMul); 
      free(hA);
      return 0;
    }

    if (piv != k){
      int bls = (n - k + tpb - 1)/tpb;
      swapRowsFromK<<<bls, tpb>>>(dA, n, k, piv, k);
      cudaGetLastError();
      cudaDeviceSynchronize();
      ++swaps;
    }

    int rows = n - (k+1);
    if (rows > 0){
      int bl1 = (rows + tpb - 1)/tpb;
      buildMultipliers<<<bl1, tpb>>>(dA, n, k, dMul);
      cudaGetLastError();
      cudaDeviceSynchronize();

      // eliminate on the trailing submatrix
      dim3 block2d(32, 8);
      dim3 grid2d((n - (k+1) + block2d.x - 1)/block2d.x,
                  (n - (k+1) + block2d.y - 1)/block2d.y);
      eliminate<<<grid2d, block2d>>>(dA, n, k, dMul);
      cudaGetLastError();
      cudaDeviceSynchronize();
    }
  }

  long double det = (swaps % 2 == 0 ? 1.0L : -1.0L);
  for (int i=0;i<n;i++){
    double diag_i;
    cudaMemcpy(&diag_i, dA + (size_t)i * n + i, sizeof(double), cudaMemcpyDeviceToHost);
    det *= (long double)diag_i;
  }
  printf("%.10Le\n", det);

  cudaFree(dA); 
  cudaFree(dAbsCol); 
  cudaFree(dMul); 
  free(hA);
  return 0;
}