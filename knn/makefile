knn-parallel: knn-parallel.cpp
	g++ -fopenmp -o knn-parallel knn-parallel.cpp

knn: knn.cpp
	g++ -o knn knn.cpp

k-nearest-neighbours: k-nearest-neighbours.cpp
	g++ -fopenmp -o k-nearest-neighbours k-nearest-neighbours.cpp

all: knn-parallel.cpp knn.cpp
	g++ -fopenmp -o knn-parallel knn-parallel.cpp -o knn knn.cpp

clean:
	rm knn knn-parallel k-nearest-neighbours