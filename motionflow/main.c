#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include "inc/pgm.h"
#include "inc/common.h"

#define DOWN_X_G 0
#define DOWN_Y_G 1
#define FILTER_G 2
#define SCHARR_X_H 3
#define SCHARR_X_V 4
#define SCHARR_Y_H 5
#define SCHARR_Y_V 6
#define OPTICAL_FLOW 7

#define WIDTH 640
#define HEIGHT 480


// Device output buffer
cl_mem d_c;
cl_mem d_d;
cl_mem d_g;
cl_mem d_h;

cl_platform_id cpPlatform;        // OpenCL platform
cl_device_id device_id;           // device ID
cl_program program;               // program
cl_kernel kernel[8];              // kernel
cl_context context;               // context
cl_command_queue queue;           // command queue

static inline size_t divUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}



cl_mem execute(cl_kernel *kernel, cl_mem *input_image, cl_int *width, cl_int *height, int format, const char * filepath) {
    int i, min, max, value;
    cl_int err;
    cl_int pixel = *width * *height;
    size_t globalSize[] = {WIDTH, HEIGHT};
    size_t formatSize;
    if(format == CL_SIGNED_INT16){
        formatSize = sizeof(cl_short);
    } else{
        formatSize = sizeof(cl_uchar);
    }
    
    cl_mem d_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixel*formatSize, NULL, NULL);
    unsigned char * h_buffer = (unsigned char *)malloc(pixel*formatSize);
    
    err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), input_image);
    err |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &d_buffer);
    err |= clSetKernelArg(*kernel, 2, sizeof(cl_int), width);
    err |= clSetKernelArg(*kernel, 3, sizeof(cl_int), height);
    
    err = clEnqueueNDRangeKernel(queue, *kernel, 2, 0, globalSize, 0, 0,  NULL, NULL);
    
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, pixel*formatSize, h_buffer, 0, NULL, NULL );
    cl_mem out = loadImage(context, h_buffer, *width, *height, format, CL_R);
    
    if(format == CL_SIGNED_INT16){
        min = 0;
        max = 0;
        for(i = 0; i < pixel; i++) {
            value = ((cl_short *)h_buffer)[i];
            if(value < min) min = value;
            if(value > max) max = value;
        }
        printf("min: %d\n", min);
        printf("max: %d\n", max);
    
        for(i = 0; i < pixel; i++) {
            value = ((cl_short *)h_buffer)[i];
            h_buffer[i] = ((value - min) * 255)/(max-min);
        }
    }
    writePGM(filepath, h_buffer, *width, *height);
    
    free(h_buffer);
    clReleaseMemObject(d_buffer);
    return out;
}



cl_mem loadImage(cl_context context, unsigned char * buffer, int width, int height, int format, int channelOrder) {
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order=channelOrder;
    clImageFormat.image_channel_data_type=format;
    
    cl_int errNum;
    cl_mem clImage;
    clImage=clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &clImageFormat, width, height, 0, buffer, &errNum);
    if(errNum != CL_SUCCESS) {
        printf("Error creating CL image object\n");
        return 0;
    }
    return clImage;
}



const char *ksrc(const char * filename) {
    char * buffer;
    long lSize;
    FILE * fp;
    
    fp = fopen (filename, "rb" );
    if( !fp ) perror(filename),exit(1);
    
    //get filesize
    fseek(fp , 0L , SEEK_END);
    lSize = ftell(fp);
    rewind(fp);
    
    buffer = (char *)calloc(1, lSize+1);
    if(!buffer) fclose(fp),fputs("memory alloc fails",stderr),exit(1);
    
    if( 1!=fread( buffer , lSize, 1 , fp) )
        fclose(fp),free(buffer),fputs("entire read fails",stderr),exit(1);
    
    fclose(fp);
    return buffer;;
}




int main() {
    const char *kernelSource = ksrc("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/motionflow.cl");
    
    //readImage() to Buffer
    unsigned char * raw_image1 = readPGM("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/img/frame10.pgm", WIDTH, HEIGHT);
    unsigned char * raw_image2 = readPGM("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/img/frame11.pgm", WIDTH, HEIGHT);
    //Test schreiben
    writePGM("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/original1.pgm", raw_image1, WIDTH, HEIGHT);
    writePGM("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/original2.pgm", raw_image2, WIDTH, HEIGHT);
    
    cl_int err;
    
    // Number of work items in each local work group
    size_t localSize[] = {16, 16};
    
    // Number of total work items - localSize must be devisor
    size_t globalSize[] = {640, 480};
    
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    cl_mem image1 = loadImage(context, raw_image1, WIDTH, HEIGHT, CL_UNSIGNED_INT8, CL_R);
    cl_mem image2 = loadImage(context, raw_image2, WIDTH, HEIGHT, CL_UNSIGNED_INT8, CL_R);
    
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
    
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    // Create the compute kernel in the program we wish to run
    kernel[DOWN_X_G] = clCreateKernel(program, "downfilter_x_g", &err);
    kernel[DOWN_Y_G] = clCreateKernel(program, "downfilter_y_g", &err);
    kernel[FILTER_G] = clCreateKernel(program, "filter_G", &err);
    kernel[SCHARR_X_H] = clCreateKernel(program, "scharr_x_horizontal", &err);
    kernel[SCHARR_X_V] = clCreateKernel(program, "scharr_x_vertical", &err);
    kernel[SCHARR_Y_H] = clCreateKernel(program, "scharr_y_horizontal", &err);
    kernel[SCHARR_Y_V] = clCreateKernel(program, "scharr_y_vertical", &err);
    kernel[OPTICAL_FLOW] = clCreateKernel(program, "optical_flow", &err);
    
    // Create the input and output arrays in device memory for our calculation
    cl_int w[2];
    cl_int h[2];
    cl_int pixel[2];
    
    w[0] = WIDTH;
    h[0] = HEIGHT;
    pixel[0] = w[0] * h[0];
    
    w[1] = divUp(WIDTH, 2);
    h[1] = divUp(HEIGHT, 2);
    pixel[1] = w[1] * h[1];
    
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixel[0]*sizeof(cl_uchar), NULL, NULL);
    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixel[1]*sizeof(cl_uchar), NULL, NULL);
    d_g = clCreateBuffer(context, CL_MEM_READ_WRITE, pixel[0]*sizeof(cl_int4), NULL, NULL);
    d_h = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixel[0]*sizeof(cl_float4), NULL, NULL);
    unsigned char * h_c = (unsigned char *)malloc(pixel[0]);
    unsigned char * h_d = (unsigned char *)malloc(pixel[1]);
    unsigned char * h_g = (unsigned char *)malloc(pixel[0]*sizeof(cl_int4));
    unsigned char * h_h = (unsigned char *)malloc(pixel[0]*sizeof(cl_float4));
    
    
    cl_mem Ix, Iy, tmp, downX, downY, out;
    
    downX = execute(&kernel[DOWN_X_G], &image1, &w[0], &h[0], CL_UNSIGNED_INT8,
                   "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/down_x.pgm");
    downY = execute(&kernel[DOWN_Y_G], &downX, &w[1], &h[1], CL_UNSIGNED_INT8,
                    "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/down_y.pgm");
    tmp = execute(&kernel[SCHARR_X_H], &image1, &w[0], &h[0], CL_SIGNED_INT16,
                        "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/scharr_x_h.pgm");
    Ix = execute(&kernel[SCHARR_X_V], &tmp, &w[0], &h[0], CL_SIGNED_INT16,
                "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/scharr_x_v.pgm");
    tmp = execute(&kernel[SCHARR_Y_H], &image1, &w[0], &h[0], CL_SIGNED_INT16,
                 "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/scharr_y_h.pgm");
    Iy = execute(&kernel[SCHARR_Y_V], &tmp, &w[0], &h[0], CL_SIGNED_INT16,
                "/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/scharr_y_v.pgm");
    
    //Filter G
    err  = clSetKernelArg(kernel[FILTER_G], 0, sizeof(cl_mem), &Ix);
    err |= clSetKernelArg(kernel[FILTER_G], 1, sizeof(cl_mem), &Iy);
    err |= clSetKernelArg(kernel[FILTER_G], 2, sizeof(cl_mem), &d_g);
    err |= clSetKernelArg(kernel[FILTER_G], 3, sizeof(cl_int), &w[0]);
    err |= clSetKernelArg(kernel[FILTER_G], 4, sizeof(cl_int), &h[0]);
    err = clEnqueueNDRangeKernel(queue, kernel[FILTER_G], 2, 0, globalSize, 0, 0,  NULL, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_g, CL_TRUE, 0, pixel[0]*sizeof(cl_int4), h_g, 0, NULL, NULL );
    
    out = loadImage(context, h_g, w[0], h[0], CL_SIGNED_INT32, CL_RGBA);

    int min = 0;
    int max = 0;
    int i = 0;
    int value = 0;
    for(i = 0; i < pixel[0]; i++) {
        value = ((cl_int4 *)h_g)[i].y;
        if(value < min) min = value;
        if(value > max) max = value;
    }
    printf("min: %d\n", min);
    printf("max: %d\n", max);
    
    for(i = 0; i < pixel[0]; i++) {
        value = ((cl_int4 *)h_g)[i].y;
        h_g[i] = ((value - min) * 255)/(max-min);
    }
    writePGM("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/filter_g.pgm", h_g, w[0], h[0]);
    

    //Optical Flow
    err = clSetKernelArg(kernel[OPTICAL_FLOW], 0, sizeof(cl_mem), &image1);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 1, sizeof(cl_mem), &Ix);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 2, sizeof(cl_mem), &Iy);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 3, sizeof(cl_mem), &out);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 4, sizeof(cl_int), &image2);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 5, sizeof(cl_mem), 0); //guess use
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 6, sizeof(cl_mem), 0);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 7, sizeof(cl_mem), &d_h);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 8, sizeof(cl_int), &w[0]);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 9, sizeof(cl_int), &h[0]);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 10, sizeof(cl_int), &w[0]);
    err |= clSetKernelArg(kernel[OPTICAL_FLOW], 11, sizeof(cl_int), &h[0]);
    err = clEnqueueNDRangeKernel(queue, kernel[OPTICAL_FLOW], 2, 0, globalSize, 0, 0,  NULL, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_h, CL_TRUE, 0, pixel[0]*sizeof(cl_float4), h_h, 0, NULL, NULL );
    
    FILE *file = fopen("/Users/michael/Dropbox/Fachhochschule/Master/Semester2/PPP/motionflow/motionflow/out/flow.txt","w+");
    cl_float4 fl;
    for(i = 0; i<pixel[0]; i++){
        fl = ((cl_float4*)h_h)[i];
        fprintf(file,"%f:%f:%f:%f\n",fl.x,fl.y,fl.z,fl.w);
    }
    fclose(file);
    
    // release OpenCL resources
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_g);
    clReleaseMemObject(d_h);
    clReleaseProgram(program);
    clReleaseKernel(kernel[DOWN_X_G]);
    clReleaseKernel(kernel[DOWN_Y_G]);
    clReleaseKernel(kernel[SCHARR_X_H]);
    clReleaseKernel(kernel[SCHARR_X_V]);
    clReleaseKernel(kernel[SCHARR_Y_H]);
    clReleaseKernel(kernel[SCHARR_Y_V]);
    clReleaseKernel(kernel[FILTER_G]);
    clReleaseKernel(kernel[OPTICAL_FLOW]);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    //release host memory
    free(h_c);
    return EXIT_SUCCESS;
}
