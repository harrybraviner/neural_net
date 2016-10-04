CPP=g++
CPPFLAGS=-Wall -std=c++17 -g

all : NeuralNet.o

test : NeuralNet_tests
	./NeuralNet_tests

NeuralNet_tests : NeuralNet_tests.cpp NeuralNet.o NeuralNet.hpp
	${CPP} ${CPPFLAGS} NeuralNet_tests.cpp NeuralNet.o -o NeuralNet_tests -lboost_unit_test_framework

%.o : %.cpp %.hpp
	${CPP} ${CPPFLAGS} -c $<
