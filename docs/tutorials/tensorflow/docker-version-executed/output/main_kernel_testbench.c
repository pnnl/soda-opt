#define _FILE_OFFSET_BITS 64
#define __Inf (1.0 / 0.0)
#define __Nan (0.0 / 0.0)
#ifdef __cplusplus
#undef printf
#include <cstdio>
#include <cstdlib>
typedef bool _Bool;
#else
#include <stdio.h>
#include <stdlib.h>
extern void exit(int status);
#endif
#include <sys/types.h>
#ifdef __AC_NAMESPACE
using namespace __AC_NAMESPACE;
#endif
#ifndef CDECL
#ifdef __cplusplus
#define CDECL extern "C"
#else
#define CDECL
#endif
#endif
#ifndef EXTERN_CDECL
#ifdef __cplusplus
#define EXTERN_CDECL extern "C"
#else
#define EXTERN_CDECL extern
#endif
#endif
#include <mdpi/mdpi_user.h>

CDECL void main_kernel(void*, void*, void*);

int main()
{
   void* P0;
   void* P1;
   void* P2;
   {
      float P0_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
      P0 = (void*)P0_temp;
      m_param_alloc(0, sizeof(P0_temp));
      float P1_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
      P1 = (void*)P1_temp;
      m_param_alloc(1, sizeof(P1_temp));
      float P2_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
      P2 = (void*)P2_temp;
      m_param_alloc(2, sizeof(P2_temp));
      main_kernel((void*) P0, (void*) P1, (void*) P2);
   }
   return 0;
}
