LDFLAGS=-lSDL2 -l curand
CFLAGS=-O3 -g

cudalife: main.o
	nvcc -o $@ $< ${CFLAGS} ${LDFLAGS}

main.o: main.cu
	nvcc -c $< -o $@ ${CFLAGS}

.PHONY:  clean
clean:
	rm cudalife
	rm main.o

run: cudalife
	./cudalife
