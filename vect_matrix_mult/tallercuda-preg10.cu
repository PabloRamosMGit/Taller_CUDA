#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Función para imprimir una matriz
void printMatrix(const string& matrixName, const vector<vector<int>>& matrix) {
    cout << "\n" <<matrixName << ":" << endl;
    for (const auto& row : matrix) {
        for (int val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

// Función para imprimir un vector 
void printVector(const std::string& vectorName, const std::vector<int>& vector) {
    std::cout << "\n" << vectorName << ":" << std::endl;
    for (int val : vector) {
        std::cout << val << "\t";
    }
    std::cout << std::endl;
}

//-------------------- Ejecucion SERIALIZADA
// Función para multiplicar una matrix por un vector
vector<int> matrixVectorMultiply(const vector<vector<int>>& A, const vector<int>& B) {
    int m = A.size();
    int n = A[0].size();
    vector<int> result(m, 0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * B[j];
        }
    }
    return result;
}
 
//-------------------- Ejecucion CUDA
// Kernel para multiplicar una matrix por un vector en CUDA
__global__ void matrixVectorMultiplyKernel(const int* A, const int* B, int* C, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int dotProduct = 0;
        for (int j = 0; j < n; ++j) {
            dotProduct += A[row * n + j] * B[j];
        }
        C[row] = dotProduct;
    }
}

std::vector<int> matrixVectorMultiplyCUDA(const std::vector<std::vector<int>>& A, const std::vector<int>& B) {
    int m = A.size();
    int n = A[0].size();
    int* dev_A; // Matrix
    int* dev_B; // Vector
    int* dev_C; // Referencia la resultado
    std::vector<int> result(m); // Resultado

    // Alojar memoria en el dispositivo
    cudaMalloc((void**)&dev_A, m * n * sizeof(int));
    cudaMalloc((void**)&dev_B, n * sizeof(int));
    cudaMalloc((void**)&dev_C, m * sizeof(int));

    // Copiar datos de host a dispositivo
    cudaMemcpy(dev_A, &A[0][0], m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, &B[0], n * sizeof(int), cudaMemcpyHostToDevice);

    

    // Configurar tamaño y cantidad de bloques
    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    // Llamar al kernel 
    matrixVectorMultiplyKernel<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, m, n);


    // Copiar resultado de dispositivo a host
    cudaMemcpy(&result[0], dev_C, m * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria del dispositivo
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return result;
}


//---------------------------------------------------
int main() {
    const int SIZE = 500;
    vector<vector<int>> A(SIZE, vector<int>(SIZE, 1));
    vector<int> B(SIZE, 1);

    // Llena las matriz A y el vector B con números aleatorios entre min y max
    const int min = 1;
    const int max = 500;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            A[i][j] = rand() % max + min; 
        }
         B[i] = rand() % max + min;
    }

    //----- Serial
    auto startSerial = high_resolution_clock::now();
    vector<int> resultSerial = matrixVectorMultiply(A, B);
    auto stopSerial = high_resolution_clock::now();
    auto durationSerial = duration_cast<milliseconds>(stopSerial - startSerial);
    cout << "\nSerial tiempo de ejecución: " << durationSerial.count() << " milliseconds" << endl;

    //----- Cuda
    auto startCuda = high_resolution_clock::now();
    vector<int> resultCuda = matrixVectorMultiplyCUDA(A, B);
    auto stopCuda = high_resolution_clock::now();
    auto durationCuda = duration_cast<milliseconds>(stopCuda - startCuda);
    cout << "CUDA tiempo de ejecución: " << durationCuda.count() << " milliseconds" << endl;

    return 0;
}

