//
//  information.h
//  hello
//
//  Created by Michael Nienhaus on 17.04.15.
//
//

#ifndef hello_information_h
#define hello_information_h


void getPlatformInfo(cl_platform_id id, cl_platform_info name, char* str);
void getDeviceInfo(cl_device_id id, cl_device_info name, char* str);
int getInfos();


#endif
