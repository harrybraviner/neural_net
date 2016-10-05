#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNet_tests
#include <boost/test/unit_test.hpp>
#include "NeuralNet.hpp"
#include "math.h"


BOOST_AUTO_TEST_CASE ( Constructor_does_not_throw )
{
    int hiddenLayerCount[] = {5};
    NeuralNet *nn = new NeuralNet(1, 10, 15, hiddenLayerCount);
    (void)nn;   // To avoid a compiler warning

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Feed_forward_on_new_net_gives_halfs )
{
    int hiddenLayerCount[] = {5};
    NeuralNet *nn = new NeuralNet(1, 10, 15, hiddenLayerCount);
    double input[] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    double expected_output[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    double *actual_output;
    actual_output = nn->feedForward(input);
    BOOST_CHECK_EQUAL_COLLECTIONS( expected_output, expected_output + 15, actual_output, actual_output + 15 );

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Set_matrices_and_check_feed_forward )
{
    auto g = [](double x) { return 1.0/(1.0 + exp(-x)); };

    double m1[] = { 0.0, 1.0, 0.0, 0.0,
                   -3.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    5.0, 1.0, 1.0, 1.0};
    double m2[] = {0.0, 0.0, 1.0, 1.0, 0.0,
                   6.0, 1.0, 1.0, 0.0, 0.0};
    int hiddenLayerCount[] = {4};

    NeuralNet *nn = new NeuralNet(1, 3, 2, hiddenLayerCount);
    nn->setMatrix(0, m1);
    nn->setMatrix(1, m2);

    double input[] = {1.0, -0.1, 2.0, 1.0};
    double expected_output[] = {g(g(-3.0) + g(-0.1)), g(6.0 + g(1.0) + g(-3.0))};
    double *actual_output = nn->feedForward(input);

    BOOST_CHECK_EQUAL_COLLECTIONS( expected_output, expected_output + 2, actual_output, actual_output + 2);

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Compare_numeric_derivs_to_backpropogation_zero_matrices )
{
    double delta = 1e-3; // Step size for numeric derivative
    double epsilon = 1e-7;  // Tolerance for difference

    // Some matrices with arbitrary non-zero entries
    double m1[] = { 0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0};
    double m2[] = {0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0};
    int hiddenLayerCount[] = {4};

    double y[] = {1.0, 0.0};
    double input[] = {0.0, 0.0, 0.0};

    NeuralNet *nn = new NeuralNet(1, 3, 2, hiddenLayerCount);

    auto compute_numeric_deriv = [m1, m2, delta, nn, input, y](int row, int col, int layer) {
        double *m1_copy = new double[4*4];
        std::memcpy(m1_copy, m1, sizeof(double)*4*4);
        double *m2_copy = new double[2*5];
        std::memcpy(m2_copy, m2, sizeof(double)*2*5);

        if (layer==1) {
            m1_copy[4*row + col] += delta;
        } else {
            m2_copy[5*row + col] += delta;
        }
        nn->setMatrix(0, m1_copy);
        nn->setMatrix(1, m2_copy);

        double *y_plus = nn->feedForward(input);
        double C_plus = 0.0;
        for(int i = 0; i<2; i++) {
            C_plus += 0.5*(y_plus[i] - y[i])*(y_plus[i] - y[i]);
        }

        if (layer==1) {
            m1_copy[4*row + col] -= 2.0*delta;
        } else {
            m2_copy[5*row + col] -= 2.0*delta;
        }
        nn->setMatrix(0, m1_copy);
        nn->setMatrix(1, m2_copy);

        double *y_minus = nn->feedForward(input);
        double C_minus = 0.0;
        for(int i = 0; i<2; i++) {
            C_minus += 0.5*(y_minus[i] - y[i])*(y_minus[i] - y[i]);
        }

        return (C_plus - C_minus)/(2.0*delta);
    };

    nn->setMatrix(0, m1);
    nn->setMatrix(1, m2);
    double **w_derivs = nn->getDerivatives(input, y);

    double *w1_derivs = w_derivs[0];
    double *w2_derivs = w_derivs[1];

    double maxDiff = 0.0;

    for (int row=0; row<4; row++) {
        for (int col=0; col<4; col++) {
            double bp_result = w1_derivs[row*4 + col];
            double numeric_result = compute_numeric_deriv(row, col, 1);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }


    for (int row=0; row<2; row++) {
        for (int col=0; col<5; col++) {
            double bp_result = w2_derivs[row*5 + col];
            double numeric_result = compute_numeric_deriv(row, col, 2);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }

    BOOST_CHECK( maxDiff <= epsilon );
}

BOOST_AUTO_TEST_CASE ( Compare_numeric_derivs_to_backpropogation )
{
    double delta = 1e-3; // Step size for numeric derivative
    double epsilon = 1e-7;  // Tolerance for difference

    // Some matrices with arbitrary non-zero entries
    double m1[] = { 0.2,  1.0, -0.1,  0.4,
                   -3.0,  0.1,  2.0,  0.2,
                   -0.5, 0.25, -1.0, -0.6,
                    5.0, -0.7,  1.2,  1.0};
    double m2[] = {0.1, 0.2, 1.0, 1.0, -0.4,
                   6.0, 1.0, 1.6, 0.3, -0.2};
    int hiddenLayerCount[] = {4};

    double y[] = {1.0, 0.0};
    double input[] = {0.5, -3.5, 2.3};

    NeuralNet *nn = new NeuralNet(1, 3, 2, hiddenLayerCount);

    auto compute_numeric_deriv = [m1, m2, delta, nn, input, y](int row, int col, int layer) {
        double *m1_copy = new double[4*4];
        std::memcpy(m1_copy, m1, sizeof(double)*4*4);
        double *m2_copy = new double[2*5];
        std::memcpy(m2_copy, m2, sizeof(double)*2*5);

        if (layer==1) {
            m1_copy[4*row + col] += delta;
        } else {
            m2_copy[5*row + col] += delta;
        }
        nn->setMatrix(0, m1_copy);
        nn->setMatrix(1, m2_copy);

        double *y_plus = nn->feedForward(input);
        double C_plus = 0.0;
        for(int i = 0; i<2; i++) {
            C_plus += 0.5*(y_plus[i] - y[i])*(y_plus[i] - y[i]);
        }

        if (layer==1) {
            m1_copy[4*row + col] -= 2.0*delta;
        } else {
            m2_copy[5*row + col] -= 2.0*delta;
        }
        nn->setMatrix(0, m1_copy);
        nn->setMatrix(1, m2_copy);

        double *y_minus = nn->feedForward(input);
        double C_minus = 0.0;
        for(int i = 0; i<2; i++) {
            C_minus += 0.5*(y_minus[i] - y[i])*(y_minus[i] - y[i]);
        }

        return (C_plus - C_minus)/(2.0*delta);
    };

    nn->setMatrix(0, m1);
    nn->setMatrix(1, m2);
    double **w_derivs = nn->getDerivatives(input, y);

    double *w1_derivs = w_derivs[0];
    double *w2_derivs = w_derivs[1];

    double maxDiff = 0.0;

    for (int row=0; row<4; row++) {
        for (int col=0; col<4; col++) {
            double bp_result = w1_derivs[row*4 + col];
            double numeric_result = compute_numeric_deriv(row, col, 1);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }

    //printf("w2:\n\n");

    for (int row=0; row<2; row++) {
        for (int col=0; col<5; col++) {
            double bp_result = w2_derivs[row*5 + col];
            double numeric_result = compute_numeric_deriv(row, col, 2);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }

    BOOST_CHECK( maxDiff <= epsilon );
}
