
#include <OpenCL/opencl.h>


#ifndef COMMON_H
#define COMMON_H

#define WIDTH 640
#define HEIGHT 480

cl_context context;               // context
cl_command_queue queue;           // command queue

cl_mem loadImage(cl_context context, unsigned char * buffer, int width, int height, int format, int channelOrder);

#endif