#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNet_tests
#include <boost/test/unit_test.hpp>
#include "NeuralNet.hpp"


BOOST_AUTO_TEST_CASE ( Constructor_does_not_throw )
{
    int hiddenLayerCount[] = {5};
    NeuralNet *nn = new NeuralNet(1, 10, 15, hiddenLayerCount);
}

