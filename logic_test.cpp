#include <iostream>
#include "NeuralNet.hpp"

using namespace std;

int main() {

    int numberOfInputs = 2;
    int numberOfOutputs = 2;
    int numberOfHiddenLayers = 1;
    int numberOfNodesInHiddenLayers[] = {2};
    
    NeuralNet *nn = new NeuralNet(numberOfHiddenLayers, numberOfInputs, numberOfOutputs, numberOfNodesInHiddenLayers);

    // FIXME -these weights are nonsense
    double w1[] = {-150.0, 100.0, 100.0,
                    -50.0, 100.0, 100.0};
    double w2[] = {-50.0, -100.0,  100.0,
                    50.0,  100.0, -100.0};
    nn->setMatrix(0, w1);
    nn->setMatrix(1, w2);

    double input1[] = {0.0, 0.0};
    double *y = nn->feedForward(input1);
    cout << " x = {0.0, 0.0}, y = {" << y[0] << ", " << y[1] << "}\n"; 
    delete y;

    double input2[] = {0.0, 1.0};
    y = nn->feedForward(input2);
    cout << " x = {0.0, 1.0}, y = {" << y[0] << ", " << y[1] << "}\n"; 
    delete y;

    double input3[] = {1.0, 0.0};
    y = nn->feedForward(input3);
    cout << " x = {1.0, 0.0}, y = {" << y[0] << ", " << y[1] << "}\n"; 
    delete y;

    double input4[] = {1.0, 1.0};
    y = nn->feedForward(input4);
    cout << " x = {1.0, 1.0}, y = {" << y[0] << ", " << y[1] << "}\n"; 
    delete y;
    
    // Ok, but now can I train a net to do this?
    double **trainingInputs = new double*[4];
    trainingInputs[0] = input1; trainingInputs[1] = input2;
    trainingInputs[2] = input3; trainingInputs[3] = input4;

    double **trainingTargets = new double*[4];
    double target1[] = {0.0, 1.0};
    double target2[] = {1.0, 0.0};
    trainingTargets[0] = target1;
    trainingTargets[1] = target2;
    trainingTargets[2] = target2;
    trainingTargets[3] = target1;

    auto getErrorMSE = [&nn, trainingInputs, trainingTargets]() {
        double mse = 0.0;
        for (int i=0; i<4; i++) {
            double *y = nn->feedForward(trainingInputs[i]);
            //cout << "y[0]: " << y[0] << ", y[1]: " << y[1] << "\n";
            mse += (trainingTargets[i][0] - y[0])*(trainingTargets[i][0] - y[0]);
            //cout << "mse: " << mse << "\n";
            mse += (trainingTargets[i][1] - y[1])*(trainingTargets[i][1] - y[1]);
            //cout << "mse: " << mse << "\n";
            delete y;
        }
        mse /= 4.0;
        return mse;
    };

    nn->randomiseMatrices();
    cout << "Initial error rate: " << getErrorMSE() << "\n";

    double learningRate = 0.5;
    for (int i=0; i<10000; i++) {
        cout << "Training round " << i << "\n";
//        for (int j=0; j<4; j++) {
//            nn->learnStep(0.01, trainingInputs[j], trainingTargets[j]);
//        }
        nn->learnBatch(learningRate, 4, trainingInputs, trainingTargets);
        cout << "Error rate is now: " << getErrorMSE() << "\n";
    }

    double *w1_out = nn->getMatrix(0);
    double *w2_out = nn->getMatrix(1);

    cout << "w1: \n";
    for (int i=0; i<2; i++) {
        for (int j=0; j<3; j++) {
            cout << w1_out[i*3 + j] << "\t";
        }
        cout << "\n";
    };
    cout << "w2: \n";
    for (int i=0; i<2; i++) {
        for (int j=0; j<3; j++) {
            cout << w2_out[i*3 + j] << "\t";
        }
        cout << "\n";
    };

    return 0;
}

