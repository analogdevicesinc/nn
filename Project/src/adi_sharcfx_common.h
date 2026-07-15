/**
********************************************************************************
*
* @file: adi_sharcfx_common.h
*
* @brief: common header file for optimized kernel files
*
* @details: contains all the required MACROS and common include files across all source files.
*
*******************************************************************************
 Copyright(c) 2024 Analog Devices, Inc. All Rights Reserved. This software is
 proprietary & confidential to Analog Devices, Inc. and its licensors. By using
 this software you agree to the terms of the associated Analog Devices License
 Agreement.
*******************************************************************************
*/

#ifndef ADI_SHARCFX_COMMON_H_
#define ADI_SHARCFX_COMMON_H_


/*============= I N C L U D E S =============*/
#ifdef ADI_DEBUG
#include <stdio.h>
#endif
#include <math.h>
#include <float.h>
#include <complex.h>
#include <matrix.h>
#include <filter.h>
#include <vector.h>
#include <sys/platform.h>

#include "math_fixedpoint_vec.h"
#include "libdsp_types.h" 			/* Cross-platform data type definitions. */

#ifdef __XTENSA__
#include <xtensa/sim.h>
#include <xtensa/tie/xt_pdxn.h>
#endif

/*============= D E F I N E S =============*/
#define NVEC 							32
#define USE_EXTRA_PRECISION
#define ROUNDING_MODE					1
#define ACT_MAX							127
#define ACT_MIN							-128
#define INT_32BIT_MAX             		0x007FFFFFFF
#define INT_32BIT_MIN             		0xFF80000000
#define ARRAY_INPUT_SIZE           		8192
#define INT_16BIT_MAX             		32767
#define INT_16BIT_MIN            		-32768
#define INT_U8BIT_MAX               	255
#define INT_3X3_FILTER_WIDTH_x_HEIGHT	9
#define INT_3x3_FILTER_WIDTH			3
#define ROUNDING_MODE_2					2
#define STRIDE_2						2
#define TEMP_BUFFER_SIZE_L1 			(180*1024)      //180KB of L1 scratch buffer.
#define TEMP_BUFFER_SIZE_L3_HALF        (1080*1024)     //1080KB of L3 scratch buffer.
#define TEMP_BUFFER_SIZE_L3 			(2160*1024)     //2160KB of L3 scratch buffer.

#define MIN(X, Y) 						(((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) 						(((X) > (Y)) ? (X) : (Y))

/*============= D A T A =============*/
extern int8_t pTempL1[TEMP_BUFFER_SIZE_L1];		/*scratch buffer used inside kernels*/
extern int8_t pTempL3[TEMP_BUFFER_SIZE_L3];		/*scratch buffer used inside kernels*/

/*============= F U N C T I O N P R O T O T Y P E S =============*/

#endif /* ADI_SHARCFX_COMMON_H_ */
