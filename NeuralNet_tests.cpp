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
}
