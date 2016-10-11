#ifndef _NEURALNET_HPP_
#define _NEURALNET_HPP_

class NeuralNet {
    public:
        NeuralNet(int numberOfHiddenLayers, int numberOfInputs, int numberOfOutputs, int numberOfNodesInHiddenLayers[]);

        void randomiseMatrices();

        double* feedForward(const double *const input);
        void setMatrix(int layer, double *matrix);
        double *getMatrix(int layer);
        ~NeuralNet();   // FIXME - write this, free all memory
        double** getDerivatives(const double *const input, const double *const target_y);
        void learnStep(const double learningRate, double *const input, double *const target_y);
        void learnBatch(const double learningRate, const int numberOfTrainingCases, double **const input, double **const target_y);
    private:
        int numberOfInputs;
        int numberOfOutputs;
        int totalNumberOfLayers;    // Includes input and output
        int *numberOfNodesInAllLayers;   // Includes input and output
        double **transferMatrices;
        double transferFunction(double x);
        double transferFunctionDeriv(double x);

        void _feedForward();    // Internal call to push input forward through all layers of memory
        void _backPropogate();  // Internal call to push the error terms back

        // Notation of Michael Nielsen's book
        int L;  // Index of last layer
        double **a; // Output of neurons in each layer
        double **z; // Input to neurons in each layer
        double **delta; // Stores the error in each a
        double **delta_w; // Stores the derivative of C w.r.t. w
        double *y;  // One-hot coded output vector
};

#endif
