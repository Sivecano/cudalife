#include "SDL2/SDL.h"
#include "curand.h"
#include <SDL2/SDL_error.h>
#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_log.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <deque>


#define HEIGHT 1440
#define WIDTH 2560
// #define pos(x, y) (x + 1920*y)

const dim3 threads(32, 32, 1); // for WIDTH == 2560
const dim3 blocks(80, 45, 1); // for HEIGHT == 1440
// const dim3 threads(64, 8, 1); // for WIDTH == 2880
// const dim3 blocks(45, 225, 1); // for HEIGHT == 1800
// const dim3 threads(128, 8, 1); // for WIDTH == 2880
// const dim3 blocks(15, 150, 1); // for HEIGHT == 1200
// const dim3 blocks(15, 135, 1); for HEIGHT == 1080

void (*update)(const uint8_t *in, uint8_t *out);

#define n_colour 22

inline __device__ uint32_t pos(uint32_t x, uint32_t y) {
  return (x + WIDTH * y);
}

__global__ void solidify(uint8_t *arr) {
  unsigned int ind = pos(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y);
  arr[ind] %= n_colour;
  // if (arr[ind] % 2 == 0)
  //   arr[ind] = 0xff;
  // else
  //   arr[ind] = 0;
}

__global__ void clear(uint8_t *arr) {
  arr[pos(blockIdx.x * blockDim.x + threadIdx.x,
          blockIdx.y * blockDim.y + threadIdx.y)] = 0;
}

__global__ void conway(const uint8_t *in, uint8_t *out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  int neighbors = 0;

  for (int i = -(x > 0 ? 1 : 0); i < 1 + (x < (WIDTH - 1) ? 1 : 0); i++)
    for (int j = -(y > 0 ? 1 : 0); j < 1 + (y < (HEIGHT - 1) ? 1 : 0); j++)
      if (in[pos(x + i, y + j)] % 2)
        neighbors++;

  if (in[pos(x, y)] > 0)
    neighbors--;

  if (neighbors == 2)
    out[pos(x, y)] = in[pos(x, y)];

  else if (neighbors == 3)
    out[pos(x, y)] = 1;
  else
    out[pos(x, y)] = 0;

  // if (out[pos(x,y)]) out[pos(x-2,y+2)] = 0xff;
}

__global__ void day_night(const uint8_t *in, uint8_t *out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  int neighbors = 0;

  for (int i = -(x > 0 ? 1 : 0); i < 1 + (x < (WIDTH - 1) ? 1 : 0); i++)
    for (int j = -(y > 0 ? 1 : 0); j < 1 + (y < (HEIGHT - 1) ? 1 : 0); j++)
      if (in[pos(x + i, y + j)] % 2)
        neighbors++;

  if (in[pos(x, y)] > 0)
    neighbors--;

  if (neighbors == 3 || neighbors == 6 || neighbors == 7 || neighbors == 8)
    out[pos(x, y)] = 1;
  else if (neighbors == 4)
    out[pos(x, y)] = in[pos(x, y)];
  else
    out[pos(x, y)] = 0;
}

__global__ void cyclic(const uint8_t *in, uint8_t *out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  uint8_t cur = in[pos(x, y)];
  uint8_t target = (cur + 1) % n_colour;
  out[pos(x, y)] = cur;

  for (int i = -(x > 0 ? 1 : 0); i < 1 + (x < (WIDTH - 1) ? 1 : 0); i++)
    for (int j = -(y > 0 ? 1 : 0); j < 1 + (y < (HEIGHT - 1) ? 1 : 0); j++)
      if (in[pos(x + i, y + j)] == target)
        out[pos(x, y)] = target;
}

__global__ void cyclic_bugged(const uint8_t *in, uint8_t *out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  uint8_t cur = in[pos(x, y)];
  uint8_t target = (cur + 1) % n_colour;
  out[pos(x, y)] = cur;

  for (int i = -(x > 0 ? 1 : 0); i < 1 + (x < (WIDTH - 1) ? 1 : 0); i++)
    for (int j = -(y > 0 ? 1 : 0); j < 1 + (y < (HEIGHT - 1) ? 1 : 0); j++)
      if (i != 0 && j != 0 && in[pos(x + i, y + j)] == target)
        out[pos(x, y)] = target;
}

__global__ void draw(const uint8_t *in, uint32_t *out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x,
               y = blockIdx.y * blockDim.y + threadIdx.y;
  if (in[pos(x, y)])
    out[pos(x, y)] = 0xff006400;
  else
    out[pos(x, y)] = 0xff1a1a1a;
}

__global__ void cdraw(const uint8_t *in, uint32_t *out) {
  const uint32_t colormap[] = {
      0xffbf3f3f, 0xffbf663f, 0xffbf8c3f, 0xffbfb23f, 0xffa5bf3f, 0xff7fbf3f,
      0xff59bf3f, 0xff3fbf4c, 0xff3fbf72, 0xff3fbf99, 0xff3fbfbf, 0xff3f99bf,
      0xff3f72bf, 0xff3f4cbf, 0xff593fbf, 0xff7f3fbf, 0xffa53fbf, 0xffbf3fb2,
      0xffbf3f8c, 0xffbf3f66, 0xffbf3f3f, 0xffbf663f, 0xffbf8c3f, 0xffbfb23f,
      0xffa5bf3f, 0xff7fbf3f, 0xff59bf3f, 0xff3fbf4c, 0xff3fbf72, 0xff3fbf99,
      0xff3fbfbf, 0xff3f99bf, 0xff3f72bf, 0xff3f4cbf, 0xff3f3fbf, 0xff3f3fbf,
      0xff3f3fbf, 0xff3f3fb2, 0xff3f3f8c, 0xff3f3f66};

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x,
               y = blockIdx.y * blockDim.y + threadIdx.y;
  out[pos(x, y)] = colormap[in[pos(x, y)] % n_colour];
}

int main(int argc, char *argv[]) {
  SDL_Init(SDL_INIT_EVERYTHING);

  uint8_t *board;
  uint8_t *buffer;
  uint32_t *colours;
  curandGenerator_t gen;

  update = &conway;

  if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW) !=
      CURAND_STATUS_SUCCESS)
    SDL_Log("WE FUCKED UP: %s", cudaGetErrorString(cudaGetLastError()));
  curandSetPseudoRandomGeneratorSeed(gen, rand());
  cudaError_t err;

  SDL_Log("set cuda device");
  if (cudaSetDevice(0) != cudaSuccess) {
    fprintf(stderr,
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    return 1;
  }

  SDL_Log("Window init");

  SDL_Window *win =
      SDL_CreateWindow("conway's game of cuda", 0, 0, WIDTH, HEIGHT, 0);
  // SDL_Renderer* ren = SDL_CreateRenderer(win, 0, 0);

  // if (ren)
  //     SDL_Log("we rendering");
  SDL_Surface *sur = SDL_GetWindowSurface(win);
  if (!sur)
    SDL_Log("why no window surface????: %s", SDL_GetError());
  SDL_Log(
      "surdim : %i, surf: bpp: %i, fmt: %i, %s, h: %i, pitch: %i, pitch/4 : %f",
      sur->pitch * sur->h / 4, sur->format->BitsPerPixel, sur->format->format,
      SDL_GetPixelFormatName(sur->format->format), sur->h, sur->pitch,
      sur->pitch / 4.);

  err = cudaMalloc(&board, HEIGHT * WIDTH);
  if (err != cudaSuccess)
    SDL_Log("%s", cudaGetErrorString(err));
  err = cudaMalloc(&buffer, HEIGHT * WIDTH);
  if (err != cudaSuccess)
    SDL_Log("%s", cudaGetErrorString(err));
  err = cudaMalloc(&colours, 4 * HEIGHT * WIDTH);
  if (err != cudaSuccess)
    SDL_Log("%s", cudaGetErrorString(err));

  if (curandGenerate(gen, reinterpret_cast<unsigned int *>(board),
                     HEIGHT * WIDTH / 4) != CURAND_STATUS_SUCCESS)
    SDL_Log("RANDOM FUCKED UP");
  solidify<<<blocks, threads>>>(board);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    SDL_Log("%s", cudaGetErrorString(err));

  bool running = true;
  bool pause = true;
  int time = 0;
  uint8_t delay = 0;
  uint8_t stencil = 0;
  std::deque<int> frametimes(1000);

  while (running) {
    time = SDL_GetTicks();
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      switch (e.type) {
      case SDL_QUIT:
        running = false;
        break;
      case SDL_KEYUP:
        switch (e.key.keysym.sym) {
        case SDLK_r:
          SDL_Log("rand");
          if (curandGenerate(gen, reinterpret_cast<unsigned int *>(board),
                             HEIGHT * WIDTH / 4) != CURAND_STATUS_SUCCESS)
            SDL_Log("RANDOM FUCKED UP");
          solidify<<<blocks, threads>>>(board);
          err = cudaDeviceSynchronize();
          if (err != cudaSuccess)
            SDL_Log("%s", cudaGetErrorString(err));

          break;

        case SDLK_SPACE:
          pause = !pause;
          if (pause)
            SDL_Log("pause");
          else
            SDL_Log("unpause");
          break;

        case SDLK_q:
          running = false;
          break;

        case SDLK_c:
          cudaMemset(board, 0, WIDTH * HEIGHT);
          break;

        case SDLK_f:
          cudaMemset(board, 1, WIDTH * HEIGHT);
          break;

        case SDLK_s:
          if (update == conway)
            update = day_night;
          else
            update = conway;

        case SDLK_UP:
          delay += 1;
          break;

        case SDLK_DOWN:
          delay -= 1;
          break;

        case SDLK_RIGHT:
          stencil += 1;
          break;

        case SDLK_LEFT:
          stencil -= 1;
          break;

        default:
          break;
        }
        break;

      case SDL_MOUSEMOTION:
        if (e.motion.state & SDL_BUTTON_RMASK) {
          for (int i = -stencil; i <= stencil; i++)
            for (int j = -stencil; j <= stencil; j++)
              err = cudaMemset(
                  board + (e.motion.x + i) + WIDTH * (e.motion.y + j), 0x00, 2);
          if (err != cudaSuccess)
            SDL_Log("%s", cudaGetErrorString(err));
        } else if (e.motion.state & SDL_BUTTON_LMASK) {
          for (int i = -stencil; i <= stencil; i++)
            for (int j = -stencil; j <= stencil; j++)
              err = cudaMemset(
                  board + (e.motion.x + i) + WIDTH * (e.motion.y + j), 0x01, 2);
          if (err != cudaSuccess)
            SDL_Log("%s", cudaGetErrorString(err));
        }
        break;

      default:
        break;
      }
    }
    // SDL_Log("before");
    if (!pause) {
      clear<<<blocks, threads>>>(buffer);
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess)
        SDL_Log("%s", cudaGetErrorString(err));
      update<<<blocks, threads>>>(board, buffer);
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess)
        SDL_Log("%s", cudaGetErrorString(err));
      // SDL_Log("%s", cudaGetErrorString(cudaGetLastError()));
      uint8_t *temp = buffer;
      buffer = board;
      board = temp;
    }

    draw<<<blocks, threads>>>(board, colours);

    // SDL_Log("helo");
    // SDL_SetRenderDrawColor(ren, 255, 0, 0, 0);
    // SDL_RenderClear(ren);
    // SDL_RenderPresent(ren);
    {
      SDL_Surface *sur2 = SDL_GetWindowSurface(win);
      if (sur2)
        sur = sur2;
    }
    if (SDL_LockSurface(sur))
      SDL_Log("error locking surface: %s", SDL_GetError());
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess)
    //   SDL_Log("%s", cudaGetErrorString(err));

    err = cudaMemcpy(sur->pixels, colours, 4 * HEIGHT * WIDTH, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      SDL_Log("%s", cudaGetErrorString(err));


    if (SDL_UpdateWindowSurface(win))
      SDL_Log("error updating window surface: %s", SDL_GetError());
    SDL_UnlockSurface(sur);

    // while (SDL_GetTicks() - time < 4);
    // SDL_Delay(delay);
    // SDL_Log("here\n\n");
    frametimes.pop_back();
    frametimes.push_front(SDL_GetTicks() - time);
    {
      uint64_t total = 0;
      for (int t : frametimes)
        total += t;
      SDL_Log("fps: %lu", (1000 * frametimes.size()) / total);
    }
  }

  curandDestroyGenerator(gen);
  cudaFree(board);
  cudaFree(buffer);
  cudaFree(colours);
  SDL_DestroyWindow(win);
  SDL_Quit();

  return 0;
}
