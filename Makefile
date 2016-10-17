CPP=g++
CPPFLAGS=-Wall -std=c++17 -g

all : NeuralNet.o

test : NeuralNet_tests
	./NeuralNet_tests

logic_test: logic_test.cpp NeuralNet.o NeuralNet.hpp
	${CPP} ${CPPFLAGS} logic_test.cpp NeuralNet.o -o logic_test

integration_test : integration_test.cpp NeuralNet.o NeuralNet.hpp
	${CPP} ${CPPFLAGS} integration_test.cpp NeuralNet.o -o integration_test

NeuralNet_tests : NeuralNet_tests.cpp NeuralNet.o NeuralNet.hpp
	${CPP} ${CPPFLAGS} NeuralNet_tests.cpp NeuralNet.o -o NeuralNet_tests -lboost_unit_test_framework

%.o : %.cpp %.hpp
	${CPP} ${CPPFLAGS} -c $<
