#include "cuda.h"
#include "SDL2/SDL.h"
#include "curand.h"

//#define pos(x, y) (x + 1920*y)

const dim3 threads(128, 8, 1);
const dim3 blocks(15, 135, 1);

__device__ uint32_t pos(uint32_t x, uint32_t y)
{
    return (x + 1920*y);
}

__global__ void solidify(uint8_t* arr)
{
    unsigned int ind = pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (arr[ind] % 2 == 0)
        arr[ind] = 0xff;
    else
        arr[ind] = 0;
}

__global__ void conway(const uint8_t* in, uint8_t* out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int neighbors = (in[pos(x, y)] > 0) ? -1 : 1;
    for (int i = -1; i < 2; i++)
        if (x >= -i && x + i < 1920)
        for (int j = -1; j < 2; j++)
            if (y >= -j && y + y < 1920)
                    neighbors += (in[pos(x + i,y + j)] > 0)  ? 1: 0;


    if (neighbors == 2)
    {
        out[pos(x, y)] = in[pos(x, y)];
        return;
    }

    if (neighbors == 3)
    {
        out[pos(x, y)] = 0xff;
        return;
    }

    out[pos(x,y)] = 0;
}

__global__ void draw(const uint8_t* in, uint32_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    out[pos(x, y)] = 0xffffffff * (in[pos(x, y)] == 0xff);
}


int main(int argc, char* argv[])
{
    SDL_Init(SDL_INIT_EVERYTHING);

    uint8_t* board;
    uint8_t* buffer;
    uint32_t* colours;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32) !=CURAND_STATUS_SUCCESS) SDL_Log("WE FUCKED UP");
    curandSetPseudoRandomGeneratorSeed(gen, SDL_GetTicks());
    cudaError_t err;

    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    SDL_Window* win = SDL_CreateWindow("conway's game of cuda", 0, 0, 1920, 1080, 0);

    SDL_Surface* sur = SDL_GetWindowSurface(win);
    SDL_Log("surdim : %i, surf: bpp: %i, fmt: %i, %s", sur->pitch * sur->h / 4, sur->format->BitsPerPixel, sur->format->format, SDL_GetPixelFormatName(sur->format->format));

    err = cudaMalloc(&board, 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
    err = cudaMalloc(&buffer, 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
    err = cudaMalloc(&colours, 4* 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));

    if (curandGenerate(gen, reinterpret_cast<unsigned int *>(board), 1080 * 1920 / 4) !=CURAND_STATUS_SUCCESS) SDL_Log("WE FUCKED UP2");
    // cudaMemset(board, 0xff, 1080 * 1920);
    solidify<<<blocks, threads>>>(board);
    err = cudaDeviceSynchronize();
    draw<<<blocks, threads>>>(board, colours);
    err = cudaDeviceSynchronize();

    cudaMemcpy(sur->pixels, colours, 4* 1080 * 1920, cudaMemcpyDeviceToHost);
    SDL_UpdateWindowSurface(win);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
    SDL_Delay(1000);

    bool running = true;
    while (running)
    {
        SDL_Event e;
        while(SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                running = false;
            if (e.type == SDL_MOUSEMOTION) {
                if (e.motion.state & SDL_BUTTON_LMASK) {
                    err = cudaMemset(board + e.motion.x + 1920 * e.motion.y, 0x00, 2);
                    if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
                }
                else if (e.motion.state & SDL_BUTTON_RMASK){
                    err = cudaMemset(board + e.motion.x + 1920 * e.motion.y, 0xff, 2);
                    if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
                }
            }
        }
        //SDL_Log("before");
        conway<<<blocks, threads>>>(board, buffer);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
        //SDL_Log("%s", cudaGetErrorString(cudaGetLastError()));
        draw<<<blocks, threads>>>(buffer, colours);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
        //SDL_Log("helo");
        uint8_t* temp = buffer;
        buffer = board;
        board = temp;
        SDL_LockSurface(sur);
        cudaMemcpy(sur->pixels, colours, 4* 1080 * 1920, cudaMemcpyDeviceToHost);
        SDL_UnlockSurface(sur);
        SDL_UpdateWindowSurface(win);
        SDL_Log("here\n\n");
        SDL_Delay(100);
    }

    curandDestroyGenerator(gen);
    cudaFree(board);
    cudaFree(buffer);
    cudaFree(colours);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}