#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <random>
#include "math.h"

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(int numberOfHiddenLayers, int numberOfInputs, int numberOfOutputs, int numberOfNodesInHiddenLayers[]) {
    this->numberOfInputs = numberOfInputs;
    this->numberOfOutputs = numberOfOutputs;
    this->totalNumberOfLayers = numberOfHiddenLayers + 2;

    this->numberOfNodesInAllLayers = new int[this->totalNumberOfLayers];
    this->numberOfNodesInAllLayers[0] = numberOfInputs;
    this->numberOfNodesInAllLayers[this->totalNumberOfLayers - 1] = numberOfOutputs;
    for (int i=1; i<(this->totalNumberOfLayers-1); i++) {
        this->numberOfNodesInAllLayers[i] = numberOfNodesInHiddenLayers[i-1];
    }

    // Assign memory for the matrices, setting to zero
    this->transferMatrices = new double*[(numberOfHiddenLayers+1)]();
    for (int i=0; i<(this->totalNumberOfLayers-1); i++){
        // Note: the +1 is due to the bias term
        this->transferMatrices[i] = new double[(numberOfNodesInAllLayers[i]+1)*numberOfNodesInAllLayers[i+1]]();
    }

    this->L = numberOfHiddenLayers+1;
    this->a = new double*[L+1];
    this->z = new double*[L];
    this->delta = new double*[L];
    this->delta_w = new double*[L];
    this->y = new double[numberOfOutputs];

    a[0] = new double[numberOfNodesInAllLayers[0]];
    for (int i=0; i<(this->L); i++){
        a[i+1] = new double[numberOfNodesInAllLayers[i+1]];
        z[i] = new double[numberOfNodesInAllLayers[i+1]];
        delta[i] = new double[numberOfNodesInAllLayers[i+1]];
        delta_w[i] = new double[numberOfNodesInAllLayers[i+1]*(numberOfNodesInAllLayers[i]+1)];
    }
}

NeuralNet::~NeuralNet() {
    delete[] numberOfNodesInAllLayers;
    for (int i=0; i<(totalNumberOfLayers-1); i++) {
        delete[] transferMatrices[i];
    }
    delete[] transferMatrices;

    for (int i=0; i<L; i++) {
        delete[] a[i];
        delete[] z[i];
        delete[] delta[i];
        delete[] delta_w[i];
    }
    delete[] a[L];
    delete[] a;
    delete[] z;
    delete[] delta;
    delete[] delta_w;
    delete[] y;
}

void NeuralNet::randomiseMatrices() {
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    std::default_random_engine re;

    for (int l=0; l<L; l++) {
        int cols = numberOfNodesInAllLayers[l] + 1;
        int rows = numberOfNodesInAllLayers[l+1];
        double *matrix = transferMatrices[l];
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                matrix[i*cols + j] = unif(re);
            }
        }
    }
}

double NeuralNet::logisticFunction(double x) {
    return (1.0/(1.0 + exp(-x)));
}

double NeuralNet::logisticFunctionDeriv(double x) {
    // Avoid a layer of indirection
    double t = 1.0/(1.0 + exp(-x));
    return t*(1.0 - t);
}

void NeuralNet::softmaxLayer(int N, double *const z, double* y) {
    double total = 0.0;
    for (int i=0; i<N; i++) {
        y[i] = exp(z[i]);
        total += y[i];
    }
    for (int i=0; i<N; i++) { y[i] /= total; }
}

void NeuralNet::_feedForward() {
    for (int i=0; i<L; i++) {
        double *matrix = transferMatrices[i];
        int cols = numberOfNodesInAllLayers[i] + 1;
        int rows = numberOfNodesInAllLayers[i+1];
        for (int j=0; j<rows; j++) {
            z[i][j] = 0.0;
            z[i][j] += matrix[j*cols + 0];   // Bias term
            for (int k=1; k<cols; k++) {
                z[i][j] += matrix[j*cols + k]*a[i][k-1];
            }

            if (i < L-1) {
                a[i+1][j] = logisticFunction(z[i][j]);
            }
        }
        if (i == L-1) {
            softmaxLayer(rows, z[i], a[L]);
        }
    }
}

void NeuralNet::_backPropogate() {
    for (int i=0; i<numberOfOutputs; i++) {
        delta[L-1][i] = (a[L][i] - y[i]);// * logisticFunctionDeriv(z[L-1][i]);
    }
    for (int l=L-2; l>=0; l--) {
        double *matrix = transferMatrices[l+1];
        int cols = numberOfNodesInAllLayers[l+1] + 1;
        int rows = numberOfNodesInAllLayers[l+2];
        for (int i=1; i<cols; i++) {
            delta[l][i-1] = 0.0;
            for (int j=0; j<rows; j++) {
                delta[l][i-1] += matrix[j*cols + i]*delta[l+1][j];
            }
            delta[l][i-1] *= logisticFunctionDeriv(z[l][i-1]);
        }
    }

    for (int l=0; l<L; l++) {
        int cols = numberOfNodesInAllLayers[l] + 1;
        int rows = numberOfNodesInAllLayers[l+1];
        for (int i=0; i<rows; i++) {
            for (int j=1; j<cols; j++) {
                delta_w[l][i*cols + j] = delta[l][i]*a[l][j-1];
            }
            delta_w[l][i*cols + 0] = delta[l][i];
        }
    }
}

double* NeuralNet::feedForward(const double *const input) {
    std::memcpy(a[0], input, sizeof(double)*numberOfInputs);
    _feedForward();

    double *output = new double[numberOfOutputs];
    std::memcpy(output, a[L], sizeof(double)*numberOfOutputs);
    return output;
}

double** NeuralNet::getDerivatives(const double *const input, const double *const target_y) {
    std::memcpy(a[0], input, sizeof(double)*numberOfInputs);
    std::memcpy(y, target_y, sizeof(double)*numberOfOutputs);
    _feedForward();
    _backPropogate();

    double **output = new double*[L];
    for (int i=0; i<L; i++) {
        int cols = numberOfNodesInAllLayers[i] + 1;
        int rows = numberOfNodesInAllLayers[i+1];
        output[i] = new double[rows*cols];
        std::memcpy(output[i], delta_w[i], sizeof(double)*cols*rows);
    }

    return output;
}

void NeuralNet::setMatrix(int layer, double *matrix) {
    if (layer < 0 || layer >= totalNumberOfLayers) {
        throw std::domain_error("Invalid layer index.");
    }

    double *matrix_to_change = transferMatrices[layer];
    int rows = numberOfNodesInAllLayers[layer+1];
    int cols = numberOfNodesInAllLayers[layer] + 1;
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            matrix_to_change[i*cols + j] = matrix[i*cols + j];
        }
    }
}

double *NeuralNet::getMatrix(int layer) {
    if (layer < 0 || layer >= totalNumberOfLayers) {
        throw std::domain_error("Invalid layer index.");
    }

    int rows = numberOfNodesInAllLayers[layer+1];
    int cols = numberOfNodesInAllLayers[layer] + 1;
    double *matrix_to_return = new double[rows*cols];
    std::memcpy(matrix_to_return, transferMatrices[layer], sizeof(double)*rows*cols);

    return matrix_to_return;
    
}

void NeuralNet::learnStep(const double learningRate, double* const input, double *const target_y) {
    // Note: note calling getDerivatives because I don't want to allocate delta_w of memory
    std::memcpy(a[0], input, sizeof(double)*numberOfInputs);
    std::memcpy(y, target_y, sizeof(double)*numberOfOutputs);
    _feedForward();
    _backPropogate();
    
    // delta_w is now set to the correct derivatives
    for (int l=0; l<L; l++) {
        int cols = numberOfNodesInAllLayers[l] + 1;
        int rows = numberOfNodesInAllLayers[l+1];
        double *matrix = transferMatrices[l];
        double *dw = delta_w[l];
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                matrix[i*cols + j] -= learningRate * dw[i*cols + j];
            }
        }
    }
}

void NeuralNet::learnBatch(const double learningRate, const int numberOfTrainingCases, double** const input, double** const target_y) {

    double **acc_delta_w = new double*[L];
    for (int l=0; l<L; l++) {
        int rows = numberOfNodesInAllLayers[l+1];
        int cols = numberOfNodesInAllLayers[l] + 1;
        acc_delta_w[l] = new double[rows*cols]();
    }
    double acc_C = 0.0; // For comuting the cross-entropy

    for(int t=0; t<numberOfTrainingCases; t++) {
        std::memcpy(a[0], input[t], sizeof(double)*numberOfInputs);
        std::memcpy(y, target_y[t], sizeof(double)*numberOfOutputs);
        _feedForward();
        for (int i=0; i<numberOfOutputs; i++) {
            acc_C += -target_y[t][i]*log(a[L][i]); //0.5*(a[L][i] - target_y[t][i])*(a[L][i] - target_y[t][i]);
        }

        _backPropogate();
        for (int l=0; l<L; l++) {
            int rows = numberOfNodesInAllLayers[l+1];
            int cols = numberOfNodesInAllLayers[l] + 1;
            double *dw = delta_w[l];
            for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                    acc_delta_w[l][i*cols + j] += dw[i*cols + j];
                }
            }
        }

    }

    acc_C /= numberOfTrainingCases;

    // Compute the magnitude of the vector
    // Normalise by the number of training cases (otherwise smaller batches will learn faster per example! - FIXME - is this actually true anymore?
//    double acc_mag_delta_w = 0.0;
//    for (int l=0; l<L; l++) {
//        int rows = numberOfNodesInAllLayers[l+1];
//        int cols = numberOfNodesInAllLayers[l] + 1;
//        for (int i=0; i<rows; i++) {
//            for (int j=0; j<cols; j++) {
//                acc_delta_w[l][i*cols + j] /= numberOfTrainingCases;
//                //matrix[i*cols + j] -= learningRate * acc_delta_w[l][i*cols + j];
//                acc_mag_delta_w += acc_delta_w[l][i*cols + j]*acc_delta_w[l][i*cols + j];
//            }
//        }
//    }
//
//    acc_mag_delta_w = sqrt(acc_mag_delta_w);
//    double stepSize = learningRate * acc_C / acc_mag_delta_w;
    double stepSize = learningRate;
    stepSize = stepSize < 1e3 ? stepSize : 1e3;

    // Actually take the step
    for (int l=0; l<L; l++) {
        int rows = numberOfNodesInAllLayers[l+1];
        int cols = numberOfNodesInAllLayers[l] + 1;
        double *matrix = transferMatrices[l];
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                matrix[i*cols + j] -= stepSize * acc_delta_w[l][i*cols + j];
            }
        }
    }
}
