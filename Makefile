build: Makefile main.cu
	nvcc -o cudalife main.cu -O3 -lSDL2 -l curand

run: build
	./cudalife
