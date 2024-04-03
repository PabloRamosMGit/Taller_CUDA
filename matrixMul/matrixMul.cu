#include <cstdlib>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h> 


using namespace std;
//Kernel de multiplicación de dos matrices de 4x4
__global__ void matrixMul(int *a,int *b,int *c,int N){
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        int tmpSum = 0;

        for (int i = 0; i < N; i++) {
            tmpSum += a[ROW * N + i] * b[i * N + COL];
        }
        c[ROW * N + COL] = tmpSum;
    }
}

void verify_resul(int *a, int *b, int *c, int N) {
    int tmp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            assert(tmp == c[i * N + j]);
        }
    }
}

void init_matrix(int *m,int N){
    srand(clock());
    for(int i=0; i<N*N; i++){
        m[i] = rand()%100;
    }

}

void print_matrix(int *m, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m[i * N + j] << "\t";
        }
        cout << endl;
    }
}
int main(){
    //Tamaño de la matriz
    int N =4;

    int bytes = N*N*sizeof(int);

    //Los inputs son las matrices A,B y el output es C
    int *a,*b,*c;

    //Segun la documentacion de NIVIDA: Allocates memory that will be automatically managed by the Unified Memory system. 
    cudaMallocManaged(&a,bytes);
    cudaMallocManaged(&b,bytes);
    cudaMallocManaged(&c,bytes);
    //Inciamos la matrices random 
    init_matrix(a,N);
    init_matrix(b,N);
    //Como el tamaño de la matrix es pequeña no es necesario preocuparse por el numero de threads y de bloques
    int threads =4;
    int blocks = 1;
    //La matrix es una estructura bidimensional, por lo tanto tenemos (X,Y).
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);
    //Iniciamos el kernel, y utilizamos cudaDeviceSynchronize el cual es mas optimo que cudaThreadSynchronize el cual es uan versión obsoleta de cudaDeviceSynchronize.
    matrixMul<<<BLOCKS,THREADS>>>(a,b,c,N);
    cudaDeviceSynchronize();
    
    


    //Verificamos el codigo con los asserts 
    verify_resul(a,b,c,N);

    //Prints de los resultados
    cout << "Matrix A:" << endl;
    print_matrix(a, N);
    cout << "Matrix B:" << endl;
    print_matrix(b, N);
    cout << "Matrix C:" << endl;
    print_matrix(c, N);

    cout<<"Program SUCCESFUL"<<endl;


    return 0;

}