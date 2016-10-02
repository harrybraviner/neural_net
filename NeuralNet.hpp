#ifndef _NEURALNET_HPP_
#define _NEURALNET_HPP_

class NeuralNet {
    public:
        NeuralNet(int numberOfHiddenLayers, int numberOfInputs, int numberOfOutputs, int numberOfNodesInHiddenLayers[]);
        double* feedForward(double *input);
        void setMatrix(int layer, double *matrix);
        ~NeuralNet();   // FIXME - write this, free all memory
    private:
        int numberOfInputs;
        int numberOfOutputs;
        int totalNumberOfLayers;    // Includes input and output
        int *numberOfNodesInAllLayers;   // Includes input and output
        double **transferMatrices;
        double transferFunction(double x);

        double *feedForwardScratch1;    // Memory to use to hold acitviations
        double *feedForwardScratch2;    // during feed-forward
};

#endif
