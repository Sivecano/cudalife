#include "cuda.h"
#include "SDL2/SDL.h"
#include "curand.h"
#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_render.h>

//#define pos(x, y) (x + 1920*y)

const dim3 threads(128, 8, 1);
const dim3 blocks(15, 135, 1);

#define n_colour 25

__device__ uint32_t pos(uint32_t x, uint32_t y)
{
    return (x + 1920*y);
}

__global__ void solidify(uint8_t* arr)
{
    unsigned int ind = pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    arr[ind] %= n_colour;
    return;
    if (arr[ind] % 2 == 1)
        arr[ind] = 0xff;
    else
        arr[ind] = 0;
}

__global__ void clear(uint8_t* arr)
{
  arr[pos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)] = 0;
}

__global__ void conway(const uint8_t* in, uint8_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int neighbors = 0;

    for (int i = - (x > 0 ? 1 : 0); i < 1 + (x < 1919 ? 1 : 0); i++)
        for (int j = - (y > 0 ? 1 : 0); j < 1 + (y < 1079 ? 1 : 0); j++)
            if (in[pos(x + i, y + j)] % 2)
                neighbors++;
    

    if (in[pos(x, y)] > 0)
        neighbors--;


    if (neighbors == 2)
        out[pos(x, y)] = in[pos(x, y)];

    else if (neighbors == 3)
        out[pos(x, y)] = 1;
    else out[pos(x,y)] = 0;

    //if (out[pos(x,y)]) out[pos(x-2,y+2)] = 0xff;
}

__global__ void cyclic(const uint8_t* in, uint8_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    uint8_t cur = in[pos(x, y)];
    uint8_t target = (cur + 1) % n_colour;
    out[pos(x, y)] = cur;

    for (int i = - (x > 0 ? 1 : 0); i < 1 + (x < 1919 ? 1 : 0); i++)
        for (int j = - (y > 0 ? 1 : 0); j < 1 + (y < 1079 ? 1 : 0); j++)
            if (in[pos(x + i, y + j)] == target)
                out[pos(x, y)] = target;
}

__global__ void cyclic_bugged(const uint8_t* in, uint8_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    uint8_t cur = in[pos(x, y)];
    uint8_t target = (cur + 1) % n_colour;
    out[pos(x, y)] = cur;

    for (int i = - (x > 0 ? 1 : 0); i < 1 + (x < 1919 ? 1 : 0); i++)
        for (int j = - (y > 0 ? 1 : 0); j < 1 + (y < 1079 ? 1 : 0); j++)
            if (i != 0 && j != 0 && in[pos(x + i, y + j)] == target)
                out[pos(x, y)] = target;
}

__global__ void draw(const uint8_t* in, uint32_t* out)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if(in[pos(x, y)])
        out[pos(x,y)] = 0x00006400;
    else
        out[pos(x, y)] = 0xff1a1a1a;
}

__global__ void cdraw(const uint8_t* in, uint32_t* out)
{
    const uint32_t colormap[] = {0xffbf3f3f, 0xffbf663f, 0xffbf8c3f, 0xffbfb23f, 0xffa5bf3f, 0xff7fbf3f, 0xff59bf3f, 0xff3fbf4c, 0xff3fbf72, 0xff3fbf99, 0xff3fbfbf, 0xff3f99bf, 0xff3f72bf, 0xff3f4cbf, 0xff593fbf, 0xff7f3fbf, 0xffa53fbf, 0xffbf3fb2, 0xffbf3f8c, 0xffbf3f66, 0xffbf3f3f, 0xffbf663f, 0xffbf8c3f, 0xffbfb23f, 0xffa5bf3f, 0xff7fbf3f, 0xff59bf3f, 0xff3fbf4c, 0xff3fbf72, 0xff3fbf99, 0xff3fbfbf, 0xff3f99bf, 0xff3f72bf, 0xff3f4cbf, 0xff3f3fbf, 0xff3f3fbf, 0xff3f3fbf, 0xff3f3fb2, 0xff3f3f8c, 0xff3f3f66};

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    out[pos(x,y)] = colormap[in[pos(x,y)] % n_colour];
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
    
    SDL_Log("set cuda device");
    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }
    
    SDL_Log("Window init");

    SDL_Window* win = SDL_CreateWindow("conway's game of cuda", 0, 0, 1920, 1080,
                                       SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_MOUSE_FOCUS |
                                       0 | SDL_WINDOW_ALLOW_HIGHDPI |
                                       SDL_WINDOW_SKIP_TASKBAR | SDL_WINDOW_SHOWN );
    SDL_Renderer* ren = SDL_CreateRenderer(win, 0, 0);

    SDL_Log("we rendering");
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
    bool pause = true;
    // int time = 0;
    uint8_t delay = 0;

    while (running)
    {
        // time = SDL_GetTicks();
        SDL_Event e;
        while(SDL_PollEvent(&e)) {
            switch(e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYUP:
                    switch (e.key.keysym.sym) {
                        case SDLK_r:
                            SDL_Log("rand");
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
                            break;

                        case SDLK_c:
                            cudaMemset(board, 0, 1920 * 1080);
                            break;
                        
                        case SDLK_UP:
                            delay += 1;
                            break;
                            
                        case SDLK_DOWN:
                            delay -= 1;
                            break;

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
            clear<<<blocks, threads>>>(buffer);
            cyclic_bugged<<<blocks, threads>>>(board, buffer);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) SDL_Log("%s", cudaGetErrorString(err));
            //SDL_Log("%s", cudaGetErrorString(cudaGetLastError()));
            uint8_t* temp = buffer;
            buffer = board;
            board = temp;
        }
        
        
        cdraw<<<blocks, threads>>>(board, colours);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
            SDL_Log("%s", cudaGetErrorString(err));
        //SDL_Log("helo");
        SDL_LockSurface(sur);
        if (cudaMemcpy(sur->pixels, colours, 4* 1080 * 1920, cudaMemcpyDeviceToHost) != cudaSuccess)
            SDL_Log("Oops!");
        SDL_UpdateWindowSurface(win);
        SDL_UnlockSurface(sur);
        // SDL_SetRenderDrawColor(ren, 255, 0, 0, 0);
        // SDL_RenderClear(ren);
        // SDL_RenderPresent(ren);
        // while (SDL_GetTicks() - time < 4);
        SDL_Delay(delay);
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
