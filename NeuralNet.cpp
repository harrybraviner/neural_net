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
    int maxNumberOfNodesInAnyLayer = std::max(numberOfInputs, numberOfOutputs);
    for (int i=1; i<(this->totalNumberOfLayers-1); i++) {
        this->numberOfNodesInAllLayers[i] = numberOfNodesInHiddenLayers[i-1];
        maxNumberOfNodesInAnyLayer = std::max(maxNumberOfNodesInAnyLayer, numberOfNodesInHiddenLayers[i-1]);
    }

    // Assign memory for the matrices, setting to zero
    this->transferMatrices = new double*[(numberOfHiddenLayers+1)]();
    for (int i=0; i<(this->totalNumberOfLayers-1); i++){
        this->transferMatrices[i] = new double[numberOfNodesInAllLayers[i]*numberOfNodesInAllLayers[i+1]]();
    }



    this->feedForwardScratch1 = new double[maxNumberOfNodesInAnyLayer];
    this->feedForwardScratch2 = new double[maxNumberOfNodesInAnyLayer];
}

double NeuralNet::transferFunction(double x) {
    return (1.0/(1.0 + exp(-x)));
}

double* NeuralNet::feedForward(double *input) {
    // Efficient feed-forward method - avoid repeated memory assignments
    int rows = numberOfNodesInAllLayers[1];
    int cols = numberOfNodesInAllLayers[0];
    double *prevActivations = input;
    double *nextActivations = feedForwardScratch1;
    double *matrix = transferMatrices[0];
    
    // i indexes the layer we are feeding forward *from*
    // FIXME - you haven't updated rows and cols here!
    for (int i=0; i<(totalNumberOfLayers-1); i++) {
        rows = numberOfNodesInAllLayers[i+1];
        cols = numberOfNodesInAllLayers[i];
        for(int j=0; j<rows; j++) {
            nextActivations[j] = 0.0;
            for(int k=0; k<cols; k++) {
                nextActivations[j] += matrix[j*cols + k]*prevActivations[k];
            }
            nextActivations[j] = transferFunction(nextActivations[j]);
        }

        // Note: yes, this results in pointers to non-existent matrices at the end,
        //       but we'll never try and access them.
        // Now swap the pointers to the next and prev layers
        if (i%2 == 0) {
            prevActivations = feedForwardScratch1;
            nextActivations = feedForwardScratch2;
        } else {
            prevActivations = feedForwardScratch2;
            nextActivations = feedForwardScratch1;
        }
        matrix = transferMatrices[i+1];
    }

    double *output = new double[numberOfOutputs];
    double *outputToCopy;
    if (totalNumberOfLayers%2 == 0) {
        outputToCopy = feedForwardScratch1;
    } else {
        outputToCopy = feedForwardScratch2;
    }
    for(int i=0; i<numberOfOutputs; i++){
        std::memcpy(output, outputToCopy, sizeof(double)*numberOfOutputs);
    }

    return output;
}

void NeuralNet::setMatrix(int layer, double *matrix) {
    if (layer < 0 || layer >= totalNumberOfLayers) {
        throw std::domain_error("Invalid layer index.");
    }

    double *matrix_to_change = transferMatrices[layer];
    int rows = numberOfNodesInAllLayers[layer+1];
    int cols = numberOfNodesInAllLayers[layer];
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            matrix_to_change[i*cols + j] = matrix[i*cols + j];
        }
    }
}
