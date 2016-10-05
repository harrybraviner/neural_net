#include <algorithm>
#include <cstring>
#include <stdexcept>
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

double NeuralNet::transferFunction(double x) {
    return (1.0/(1.0 + exp(-x)));
}

double NeuralNet::transferFunctionDeriv(double x) {
    // Avoid a layer of indirection
    double t = 1.0/(1.0 + exp(-x));
    return t*(1.0 - t);
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

            a[i+1][j] = transferFunction(z[i][j]);
        }
    }
}

void NeuralNet::_backPropogate() {
    for (int i=0; i<numberOfOutputs; i++) {
        delta[L-1][i] = (a[L][i] - y[i]) * transferFunctionDeriv(z[L-1][i]);
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
            delta[l][i-1] *= transferFunctionDeriv(z[l][i-1]);
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
