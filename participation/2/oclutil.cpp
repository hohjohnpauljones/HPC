/*
    OCLTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Dr. Zaius
    ClusterChimps.org
*/

// standard utilities and systems includes
#include <stdio.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <map>
#include <openssl/des.h>
#include <CL/cl.h>

extern "C" {
   extern char __ocl_code_start __attribute__((weak));
   extern char __ocl_code_end   __attribute__((weak));

   cl_int         ocltGetPlatformID(cl_platform_id* clSelectedPlatformID, const char* name);
   char*          ocltDecrypt(char *key, char *kernel, int size);
   void           ocltExtractKernels();
   char*          ocltLoadKernelSrc(const char* filename, size_t* length);
   unsigned char* ocltLoadKernelBin(const char* filename, char** compilerFlags, size_t* length);
   unsigned char* ocltGetEmbeddedKernelBin(char* kernelName, char** compilerFlags, size_t* length);
   unsigned char* ocltGetEmbeddedKernelSrc(char* kernelName, size_t* length);
   void           ocltLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice);
   const char* 	  oclErrorString(cl_int error);
};

std::map<std::string, std::string> __kernel_map__;
std::map<std::string, std::string> __flag_map__;

////////////////////////////////////////////////////////////////////////////////
// Decrypts encrypted kernel text (src || bin)
////////////////////////////////////////////////////////////////////////////////
#ifdef CRYPT
char* ocltDecrypt(char *key, char *kernel, int size)
{

   static char*    res;
   int             n=0;

   DES_cblock      key2;
   DES_key_schedule schedule;

   res = ( char * ) malloc( size );

    /* Prepare the key for use with DES_cfb64_encrypt */
   memcpy( key2, key, 8);
   DES_set_odd_parity( &key2 );
   DES_set_key_checked( &key2, &schedule );

   /* Decryption occurs here */
   DES_cfb64_encrypt( ( unsigned char * ) kernel, ( unsigned char * ) res,
                           size, &schedule, &key2, &n, DES_DECRYPT );

   return (res);

}
#endif

////////////////////////////////////////////////////////////////////////////////
// Extracts embedded kernels from application binary
////////////////////////////////////////////////////////////////////////////////
void ocltExtractKernels()
{
   size_t size = &__ocl_code_end - &__ocl_code_start;
   char *start = &__ocl_code_start;

   if(size < 5)
   {
      std::cout << "OCLTools[ERROR] In call to ocltExtractKernels" << std::endl;
      std::cout << "                Can't extract kernel from binary" << std::endl;
      std::cout << "                Did you forget to link in your kernel binary?" << std::endl;
      exit(1);
   }

   unsigned char* buffer = (unsigned char*) malloc (size);

   size_t length = size;
   memcpy(buffer, start, size);

   std::string blob((const char*)buffer, length);
   std::string::size_type start_flag   = 0;
   std::string::size_type kernel_start = 0;
   while((start_flag = blob.find("!@#~", start_flag)) != std::string::npos)
    {
      std::string::size_type end_flag = blob.find("!@#~", start_flag + 1);

      std::string compilerFlags;
      if((end_flag - start_flag) > 5)
      {
         compilerFlags = blob.substr(start_flag + 4, end_flag - start_flag - 4);
      }

      std::string::size_type end_name = blob.find("!@#~", end_flag + 1);

      std::string name;
      if((end_name - end_flag) > 5)
      {
         name = blob.substr(end_flag + 4, end_name - end_flag - 4);
      }

      std::string kernel = blob.substr(kernel_start, start_flag - kernel_start - 2);

      __kernel_map__[name] = kernel;
      __flag_map__[name]   = compilerFlags;

//std::cout << "NAME "   << name          << std::endl;
//std::cout << "FLAGS "  << compilerFlags << std::endl;
//std::cout << "KERNEL " << kernel        << std::endl;

      kernel_start = end_name + 4;
      start_flag   = end_name + 1;
   }

}

////////////////////////////////////////////////////////////////////////////////
// Gets the OpenCL cl_platform_id
////////////////////////////////////////////////////////////////////////////////
cl_int ocltGetPlatformID(cl_platform_id* clSelectedPlatformID, const char* name)
{
    char            buffer[2048];
    cl_uint         num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int          status;

    *clSelectedPlatformID = NULL;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Error " << status << "in clGetPlatformIDs" << std::endl;
        return -1;
    }
    if(num_platforms == 0)
    {
        std::cerr << "No OpenCL platform found!" << std::cout;
        return -2;
    }
    else
    {
        clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));

        status = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
        for(uint i = 0; i < num_platforms; ++i)
        {
            status = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 2048, &buffer, NULL);
            if(status == CL_SUCCESS)
            {
                if(strstr(buffer, name) != NULL)
                {
                    *clSelectedPlatformID = clPlatformIDs[i];
                    break;
                }
            }
        }

        if(*clSelectedPlatformID == NULL)
        {
            std::cerr << "WARNING: Requested OpenCL platform not found - defaulting to first platform!" << std::endl;
            *clSelectedPlatformID = clPlatformIDs[0];
        }
        free(clPlatformIDs);
    }

    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Loads a kernel from file system
////////////////////////////////////////////////////////////////////////////////
char* ocltLoadKernelSrc(const char* filename, size_t* length)
{
    FILE* ifp = NULL;

    ifp = fopen(filename, "rb");
    if(ifp == 0)
    {
        return NULL;
    }

    fseek(ifp, 0, SEEK_END);
    *length = ftell(ifp);
    fseek(ifp, 0, SEEK_SET);

    char* buffer = (char *)malloc(*length + 1);
    if (fread(buffer, *length, 1, ifp) != 1)
    {
        fclose(ifp);
        free(buffer);
        return 0;
    }

    fclose(ifp);
    buffer[*length] = '\0';

    return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// Gets embedded kernel binary from MAP
////////////////////////////////////////////////////////////////////////////////
unsigned char* ocltGetEmbeddedKernelBin(char* kernelName, char** compilerFlags, size_t* length)
{

   std::string kernel = __kernel_map__[kernelName];
   *length = kernel.length();

   if(*length < 5)
   {
      std::cerr << "OCLTools[ERROR] " << std::endl;
      std::cerr << "In call to ocltGetEmbeddedKernelBin" << std::endl;
      std::cerr << "The kernel name you are looking for (" << kernelName << ") is not embedded in this binary" << std::endl;
      std::cerr << "Either you forgot to call ocltExtractKernels or you have a typo in your kernel name" << std::endl;
      *compilerFlags = 0;
      *length = 0;
      return(NULL);
   }

   char* kernelCStr = (char *)malloc(*length + 1);
   kernel.copy(kernelCStr, *length);
   kernelCStr[*length] = '\0';

   std::string flags = __flag_map__[kernelName];
   size_t flagLength = flags.length();

   char* flagCStr = (char *)malloc(flagLength + 1);
   flags.copy(flagCStr, flagLength);
   flagCStr[flagLength] = '\0';
   *compilerFlags = flagCStr;

   return (unsigned char*)kernelCStr;
}

////////////////////////////////////////////////////////////////////////////////
// Gets Embedded Kernel Source from MAP
////////////////////////////////////////////////////////////////////////////////
unsigned char* ocltGetEmbeddedKernelSrc(char* kernelName, size_t* length)
{
   std::string kernel = __kernel_map__[kernelName];
   *length = kernel.length();

   if(*length < 5)
   {
      std::cerr << "OCLTools[ERROR] " << std::endl;
      std::cerr << "In call to ocltGetEmbeddedKernelSrc" << std::endl;
      std::cerr << "The kernel name you are looking for (" << kernelName << ") is not embedded in this binary" << std::endl;
      std::cerr << "Either you forgot to call ocltExtractKernels or you have a typo in your kernel name" << std::endl;
      *length = 0;
      return(NULL);
   }

   char* kernelCStr = (char *)malloc(*length + 1);
   kernel.copy(kernelCStr, *length);
   kernelCStr[*length + 1] = '\0';

   return (unsigned char*)kernelCStr;
}

////////////////////////////////////////////////////////////////////////////////
// Loads a kernel binary from file system
////////////////////////////////////////////////////////////////////////////////
unsigned char* ocltLoadKernelBin(const char* filename, char** compilerFlags, size_t* length)
{
   FILE* fp = fopen(filename, "r");
   if(fp == 0)
        return NULL;

   fseek (fp , 0 , SEEK_END);
   *length = ftell(fp);
   rewind(fp);
   unsigned char* buffer;
   buffer = (unsigned char*) malloc (*length);
   fread(buffer, 1, *length, fp);
   fclose(fp);

   std::string blob((const char*)buffer, *length);
   size_t start_cookie = blob.find("!@#~", 0);

   // remove the '//' from the compiler flags line
   *length = start_cookie - 2;
   size_t end_cookie   = blob.find("!@#~", start_cookie + 1);

   std::string flags = blob.substr(start_cookie+4, end_cookie) ;

   if(compilerFlags != NULL)
   {
      size_t flagLength = flags.length() + 1;
      char* flagBuffer = (char *)malloc(flagLength);
      flags.copy(flagBuffer, flagLength - 1);
      flagBuffer[flagLength] = '\0';

      *compilerFlags = flagBuffer;
   }

   return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// Get the build log from ocl
////////////////////////////////////////////////////////////////////////////////
void ocltLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice)
{
    char buildLog[10240];
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
    std::cout << buildLog << std::endl;
}

// Convert from error code to string
const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
    
    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

