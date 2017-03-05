#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>



void getPlatformInfo(cl_platform_id id, cl_platform_info name, char* str)
{
    cl_int err;
    size_t paramValueSize;
    char * info;
    
    // Größe des Pattern holen
    err = clGetPlatformInfo(id,name,0,NULL,&paramValueSize);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to find OpenCL platform %s\n", str);
        return;
    }
    // Speicher in Größe des Pattern holen
    info = (char *)malloc(sizeof(char) * paramValueSize);
    // Inhalt der Größe des Pattern holen
    err = clGetPlatformInfo(id,name,paramValueSize,info,NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr,"Failed to find OpenCL platform %s\n",str);;
        return;
    }
    printf("%s\t%s\n", str, info);
    free(info);
}


void getDeviceInfo(cl_device_id id, cl_device_info name, char* str)
{
    cl_int err;
    size_t paramValueSize;
    cl_uint info;
    
    // Größe des Pattern holen
    err = clGetDeviceInfo(id,name,0,NULL,&paramValueSize);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to find OpenCL platform %s\n", str);
        return;
    }
    // Speicher in Größe des Pattern holen
    // Inhalt der Größe des Pattern holen
    
    err = clGetDeviceInfo(id,name,paramValueSize,&info,NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr,"Failed to find OpenCL platform %s\n",str);;
        return;
    }
    
    printf("%s\t%d\n", str, info);
}



int getInfos()
{
    cl_int err;
    cl_uint i;
    cl_uint numPlatforms;
    cl_uint numDevices = 0;
    cl_platform_id * platformIds;
    cl_device_id deviceIds[1];
    
    
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        fprintf(stderr, "Failed to find any OpenCL platform.\n");
        return 0;
    }
    printf("======= Plattform Infos ==========\n");
    printf("Number of platforms: %d\n", numPlatforms);
    platformIds = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platformIds, NULL);
    if(err != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to find any OpenCL platforms.\n");
        return 0;
    }
    for(i = 0; i < numPlatforms; i++)
    {
        getPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
        getPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
        getPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
        getPlatformInfo(platformIds[i], CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
        getPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
    }
    
    printf("\n======= Device Infos ==========\n");
    err = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);
    if (numDevices < 1)
    {
        fprintf( stderr, "keine GPU vorhanden\n");
        return 0;
    }
    else
    {
        deviceIds[0]= (cl_device_id) malloc(numDevices*sizeof(cl_device_id));
        printf("Anzahl der GPUs: %d\n",numDevices);
    }
    if(err != CL_SUCCESS)
    {
        fprintf( stderr, "Fail to find any OpenCL Device. \n");
        return 0;
    }
    err=clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_GPU,1,&deviceIds[0],NULL);
    if(err != CL_SUCCESS)
    {
        fprintf( stderr, "Failed by CL_DEVICE_TYPE_GPU");
        return 0;
    }
    for( i=0; i < numDevices; i++)
    {
        getDeviceInfo(deviceIds[0],CL_DEVICE_VENDOR_ID ,"CL_DEVICE_VENDOR_ID ");
        getDeviceInfo(deviceIds[0],CL_DEVICE_MAX_COMPUTE_UNITS ,"CL_DEVICE_MAX_COMPUTE_UNITS");
        getDeviceInfo(deviceIds[0],CL_DEVICE_TYPE ,"CL_DEVICE_TYPE ");
        getDeviceInfo(deviceIds[0],CL_DEVICE_IMAGE_SUPPORT ,"CL_DEVICE_IMAGE_SUPORT");
        getDeviceInfo(deviceIds[0],CL_DEVICE_ADDRESS_BITS  ,"CL_DEVICE_ADDRESS_BITS ");
    }
    
    free(platformIds);
    return 0;
}