/**
********************************************************************************
*
* @file: adi_sharcfx_globals.cpp
*
* @brief: source file for global variable definitions
*
* @details: source file for global variable definitions
*
*******************************************************************************
 Copyright(c) 2024 Analog Devices, Inc. All Rights Reserved. This software is
 proprietary & confidential to Analog Devices, Inc. and its licensors. By using
 this software you agree to the terms of the associated Analog Devices License
 Agreement.
*******************************************************************************
*/

/*============= I N C L U D E S =============*/
#include "adi_sharcfx_common.h"


/*============= D A T A =============*/
int8_t pTempL1[TEMP_BUFFER_SIZE_L1]__attribute__((section(".L1.noload"), aligned(8)));        /*scratch buffer used inside kernels*/
int8_t pTempL3[TEMP_BUFFER_SIZE_L3]__attribute__((section(".L3.noload"), aligned(8)));       /*scratch buffer used inside kernels*/
