build: Makefile main.cu
	nvcc -o cudabrot main.cu -O2 -lSDL2 -l curand

run: build
	./cudabrot