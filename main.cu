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
    if (arr[ind] % 2 == 1)
        arr[ind] = 0xff;
    else
        arr[ind] = 0;
}

__global__ void conway(const uint8_t* in, uint8_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int neighbors = 0;

    if (x > 0)
    {
        if (y > 0) if (in[pos(x - 1, y - 1)] > 0)
                neighbors++;

        if (in[pos(x -1, y)] > 0)
            neighbors++;

        if (y < 1079) if (in[pos(x -1, y + 1)] > 0)
                neighbors++;

    }

    if (y > 0) if (in[pos(x, y - 1)] > 0)
            neighbors++;

    if (y < 1079) if (in[pos(x, y + 1)] > 0)
            neighbors++;

    if (x < 1919)
    {
        if (y > 0) if (in[pos(x + 1, y - 1)] > 0)
                neighbors++;

        if (in[pos(x + 1, y)] > 0)
            neighbors++;

        if (y < 1079) if (in[pos(x + 1, y + 1)] > 0)
                neighbors++;
    }

/*
    for (int i = -1; i < 2; i++)
        if (x >= -i && x + i < 1920)
            for (int j = -1; j < 2; j++)
                if (y >= -j && y + j < 1080)
                    if (in[pos(x + i, y + j)] > 0)
                        neighbors++;*/



    if (neighbors == 2)
        out[pos(x, y)] = in[pos(x, y)];

    else if (neighbors == 3)
        out[pos(x, y)] = 0xff;
    else out[pos(x,y)] = 0;
}

__global__ void draw(const uint8_t* in, uint32_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if(in[pos(x, y)] == 0xff)
        out[pos(x,y)] = 0x00006400;
    else
        out[pos(x, y)] = 0x00141414 ;
}


int main(int argc, char* argv[])
{
    SDL_Init(SDL_INIT_EVERYTHING);

    uint8_t* board;
    uint8_t* buffer;
    uint32_t* colours;
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW) !=CURAND_STATUS_SUCCESS) SDL_Log("WE FUCKED UP: %s", cudaGetErrorString(cudaGetLastError()));
    curandSetPseudoRandomGeneratorSeed(gen, rand());
    cudaError_t err;

    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    SDL_Window* win = SDL_CreateWindow("conway's game of cuda", 0, 0, 1920, 1080,
                                       SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_MOUSE_FOCUS |
                                       0 | SDL_WINDOW_ALLOW_HIGHDPI |
                                       SDL_WINDOW_SKIP_TASKBAR | SDL_WINDOW_SHOWN );

    SDL_Surface* sur = SDL_GetWindowSurface(win);
    SDL_Log("surdim : %i, surf: bpp: %i, fmt: %i, %s, h: %i, pitch: %i, pitch/4 : %f", sur->pitch * sur->h / 4, sur->format->BitsPerPixel, sur->format->format, SDL_GetPixelFormatName(sur->format->format), sur->h, sur->pitch, sur->pitch / 4.);

    err = cudaMalloc(&board, 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
    err = cudaMalloc(&buffer, 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
    err = cudaMalloc(&colours, 4 * 1080 * 1920); if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));

    if (curandGenerate(gen, reinterpret_cast<unsigned int *>(board), 1080 * 1920 / 4) !=CURAND_STATUS_SUCCESS) SDL_Log("RANDOM FUCKED UP");
    solidify<<<blocks, threads>>>(board);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));



    bool running = true;
    bool pause = false;
    int time;

    while (running)
    {
        time = SDL_GetTicks();
        SDL_Event e;
        while(SDL_PollEvent(&e)) {
            switch(e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYUP:
                    switch (e.key.keysym.sym) {
                        case SDLK_r:
                            if (curandGenerate(gen, reinterpret_cast<unsigned int *>(board), 1080 * 1920 / 4) !=
                                CURAND_STATUS_SUCCESS)
                                SDL_Log("RANDOM FUCKED UP");
                            solidify<<<blocks, threads>>>(board);
                            err = cudaDeviceSynchronize();
                            if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
                            break;

                        case SDLK_SPACE:
                            pause = !pause;
                            break;

                        case SDLK_q:
                            running = false;

                        case SDLK_c:
                            cudaMemset(board, 0, 1920 * 1080);

                        default:
                            break;

                    }
                    break;

                case SDL_MOUSEMOTION:
                    if (e.motion.state & SDL_BUTTON_RMASK) {
                        err = cudaMemset(board + e.motion.x + 1920 * e.motion.y, 0x00, 2);
                        if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
                    } else if (e.motion.state & SDL_BUTTON_LMASK) {
                        err = cudaMemset(board + e.motion.x + 1920 * e.motion.y, 0xff, 2);
                        if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
                    }
                    break;

                default:
                    break;
            }
        }
        //SDL_Log("before");
        if (!pause) {
            conway<<<blocks, threads>>>(board, buffer);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
            //SDL_Log("%s", cudaGetErrorString(cudaGetLastError()));
            uint8_t* temp = buffer;
            buffer = board;
            board = temp;
        }
        draw<<<blocks, threads>>>(board, colours);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
        //SDL_Log("helo");
        SDL_LockSurface(sur);
        cudaMemcpy(sur->pixels, colours, 4* 1080 * 1920, cudaMemcpyDeviceToHost);
        SDL_UpdateWindowSurface(win);
        SDL_UnlockSurface(sur);
        // while (SDL_GetTicks() - time < 4);
        // SDL_Delay(1);
        // SDL_Log("here\n\n");
        // SDL_Log("frametime: %i", SDL_GetTicks() - time);
    }

    curandDestroyGenerator(gen);
    cudaFree(board);
    cudaFree(buffer);
    cudaFree(colours);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}
