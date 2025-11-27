/**
********************************************************************************
*
* @file: adi_sharcfx_elementwise_ops.cpp
*
* @brief: contains optimized version of elementwise add and multiply
*
* @details: contains optimized version of elementwise add and multiply for int16 input data
*
*******************************************************************************
 Copyright(c) 2024 Analog Devices, Inc. All Rights Reserved. This software is
 proprietary & confidential to Analog Devices, Inc. and its licensors. By using
 this software you agree to the terms of the associated Analog Devices License
 Agreement.
*******************************************************************************
*/

/*============= I N C L U D E S =============*/
#include "adi_sharcfx_nn.h"

/*============= C O D E =============*/

/**
*******************************************************************************
* Function: adi_sharcfx_elementwise_mul_int8
* @brief vectorised elementwise multiplication for 16-bit integer input
*
* @details vectorised elementwise multiplication for 16-bit integer input
*
* Parameters:
* @param [in] pInput1 - input buffer 1 (16-bit)
* @param [in] pInput2 - input buffer 2 (16-bit)
* @param [in] nSize - input size
* @param [in] nQuantizedMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nQuantizedShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInOffset1 - input offset for input buffer 1
* @param [in] nInOffset2 - input offset for input buffer 2
* @param [in] nOutOffset - kernel width
* @param [in] output_activation_min - min value after activation function
* @param [in] output_activation_max - max value after activation function
*
* @param [out] pOutput - output data (8-bit)
*
* @return None
*
*
*******************************************************************************
*/
void adi_sharcfx_elementwise_mul_int8(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int8_t* pOutput,
                                       int32_t nInputLen,
                                       uint32_t nQuantizedMultiplier,
                                       int32_t nQuantizedShift,
                                       int32_t nInOffset1,
                                       int32_t nInOffset2,
                                       int32_t nOutOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max)
{
    const immediate Lane = 0;
    const immediate round_mode = 2;

    xb_vec2Mx16 vInOff1 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset1,Lane);
    xb_vec2Mx16 vInOff2 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset2,Lane);
    xb_vecMx32 vOutOff = PDX_REP_MX32((xb_vecMx32)nOutOffset,Lane);

    xb_vecMx32 vmin = PDX_REP_MX32((xb_vecMx32)output_activation_min,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vecMx32 vmax = PDX_REP_MX32((xb_vecMx32)output_activation_max,Lane);//Replicates the lane of data specified, across all lanes of a vector register

    xb_vec2Mx16 *inp1 = (xb_vec2Mx16 *)pInput1;
    xb_vec2Mx16 *inp2 = (xb_vec2Mx16 *)pInput2;
    valign ina1=PDX_LA_2MX16_PP (inp1);// define align vector
    valign ina2=PDX_LA_2MX16_PP (inp2);// define align vector


    valign outa = PDX_Z_ALIGN();
    xb_vecMx8 *outp = (xb_vecMx8 *)pOutput;

    xb_vecMx32 shift = PDX_REP_MX32((xb_vecMx32)nQuantizedShift,Lane);
    xb_vecMx32 multiplier = PDX_REP_MX32((xb_vecMx32)nQuantizedMultiplier,Lane);

    xb_vec2Mx16 vin1, vin2, product;
    xb_vec2Mx40 acc;
    xb_vecMx32 first8,last8;
    xb_vecMx80 quant_acc, quant_acc2;
    xb_vecMx32 conv_out, conv_out2;

    int32_t nPixLeft, nPixToWrite;

    if(nInputLen%16)
    {
        nPixLeft = nInputLen;
        while(nPixLeft > 2*PDX_M)
        {
            acc=0;
            //READ IP
            PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
            PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel
            vin1 += vInOff1;        //Add input offset
            vin2 += vInOff2;    //Add filter offset
            //MAC
            PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

            PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

            //first8
            quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //last8
            quant_acc2 = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //shift and round
            quant_acc2 = PDX_SLS_MX80(quant_acc2,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //round and shift and saturate
            conv_out2 = PDX_PACKQSRV_MX80   (quant_acc2, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //add output zero point
            conv_out2 += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);

            nPixLeft -= 2*PDX_M;
            PDX_SAV32_MX8_XP (conv_out, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
            PDX_SAV32_MX8_XP (conv_out2, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
        }
        acc=0;
        //READ IP
        PDX_LA_2MX16_XP (vin1, ina1, inp1, 0);//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
        PDX_LA_2MX16_XP (vin2, ina2, inp2, 0);//read 2*PDX_M number of channels for 1 pixel
        vin1 += vInOff1;        //Add input offset
        vin2 += vInOff2;    //Add filter offset
        //MAC
        PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

        PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

        //first8
        quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
        //shift and round
        quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
        //round and shift and saturate
        conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
        //add output zero point
        conv_out += vOutOff;
        //Saturate to 8 bit range output_activation_min to 127
        conv_out = PDX_MIN_MX32(conv_out,vmax);
        conv_out = PDX_MAX_MX32(conv_out,vmin);
        nPixToWrite = MIN(nPixLeft,PDX_M);
        PDX_SAV32_MX8_XP(conv_out, outa, outp, nPixToWrite);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
        PDX_SAPOS_MX8_FP(outa,outp);//flush
        nPixLeft -= nPixToWrite;
        if (nPixLeft>0)
        {
            //last8
            quant_acc = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            nPixToWrite = MIN(nPixLeft,PDX_M);
            PDX_SAV32_MX8_XP(conv_out, outa, outp, nPixToWrite);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
            nPixLeft -= nPixToWrite;
        }
    }
    else
    {
        nPixLeft = nInputLen;
        for (int32_t i = 0; i <= nInputLen; i+= (2*PDX_M))
        {
            acc=0;
            //READ IP
            PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
            PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel
            vin1 += vInOff1;        //Add input offset
            vin2 += vInOff2;    //Add filter offset
            //MAC
            PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

            PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

            //first8
            quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //last8
            quant_acc2 = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //shift and round
            quant_acc2 = PDX_SLS_MX80(quant_acc2,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //round and shift and saturate
            conv_out2 = PDX_PACKQSRV_MX80   (quant_acc2, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //add output zero point
            conv_out2 += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);

            nPixLeft -= 2*PDX_M;
            PDX_SAV32_MX8_XP (conv_out, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
            PDX_SAV32_MX8_XP (conv_out2, outa, outp, PDX_M);//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX8_FP(outa,outp);//flush
        }

    }

}



/**
*******************************************************************************
* Function: adi_sharcfx_elementwise_mul_int16
* @brief vectorised elementwise multiplication for 16-bit integer input
*
* @details vectorised elementwise multiplication for 16-bit integer input
*
* Parameters:
* @param [in] pInput1 - input buffer 1 (16-bit)
* @param [in] pInput2 - input buffer 2 (16-bit)
* @param [in] nSize - input size
* @param [in] nQuantizedMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nQuantizedShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInOffset1 - input offset for input buffer 1
* @param [in] nInOffset2 - input offset for input buffer 2
* @param [in] nOutOffset - kernel width
* @param [in] output_activation_min - min value after activation function 
* @param [in] output_activation_max - max value after activation function
* 
* @param [out] pOutput - output data (16-bit)
*
* @return None
*
*
*******************************************************************************
*/ 
void adi_sharcfx_elementwise_mul_int16(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int16_t* pOutput,
                                       int32_t nInputLen,
                                       uint32_t nQuantizedMultiplier,
                                       int32_t nQuantizedShift,
                                       int32_t nInOffset1,
                                       int32_t nInOffset2,
                                       int32_t nOutOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max)
{
    const immediate Lane = 0;
    const immediate round_mode = 2;

    xb_vec2Mx16 vInOff1 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset1,Lane);
    xb_vec2Mx16 vInOff2 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset2,Lane);
    xb_vecMx32 vOutOff = PDX_REP_MX32((xb_vecMx32)nOutOffset,Lane);

    xb_vecMx32 vmin = PDX_REP_MX32((xb_vecMx32)output_activation_min,Lane);//Replicates the lane of data specified, across all lanes of a vector register
    xb_vecMx32 vmax = PDX_REP_MX32((xb_vecMx32)output_activation_max,Lane);//Replicates the lane of data specified, across all lanes of a vector register

    xb_vec2Mx16 *inp1 = (xb_vec2Mx16 *)pInput1;
    xb_vec2Mx16 *inp2 = (xb_vec2Mx16 *)pInput2;
    valign ina1=PDX_LA_2MX16_PP (inp1);// define align vector
    valign ina2=PDX_LA_2MX16_PP (inp2);// define align vector


    valign outa = PDX_Z_ALIGN();
    xb_vecMx16 *outp = (xb_vecMx16 *)pOutput;

    xb_vecMx32 shift = PDX_REP_MX32((xb_vecMx32)nQuantizedShift,Lane);
    xb_vecMx32 multiplier = PDX_REP_MX32((xb_vecMx32)nQuantizedMultiplier,Lane);

    xb_vec2Mx16 vin1, vin2, product;
    xb_vec2Mx40 acc;
    xb_vecMx32 first8,last8;
    xb_vecMx80 quant_acc, quant_acc2;
    xb_vecMx32 conv_out, conv_out2;

    int32_t nPixLeft, nPixToWrite;

    if(nInputLen%16)
    {
        nPixLeft = nInputLen;
        while(nPixLeft > 2*PDX_M)
        {
            acc=0;
            //READ IP
            PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
            PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel
            vin1 += vInOff1;        //Add input offset
            vin2 += vInOff2;    //Add filter offset
            //MAC
            PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

            PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

            //first8
            quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //last8
            quant_acc2 = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //shift and round
            quant_acc2 = PDX_SLS_MX80(quant_acc2,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //round and shift and saturate
            conv_out2 = PDX_PACKQSRV_MX80   (quant_acc2, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //add output zero point
            conv_out2 += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);

            nPixLeft -= 2*PDX_M;
            PDX_SAV32_MX16_XP (conv_out, outa, outp, PDX_M* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX16_FP(outa,outp);//flush
            PDX_SAV32_MX16_XP (conv_out2, outa, outp, PDX_M* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX16_FP(outa,outp);//flush
        }
        acc=0;
        //READ IP
        PDX_LA_2MX16_XP (vin1, ina1, inp1, 0);//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
        PDX_LA_2MX16_XP (vin2, ina2, inp2, 0);//read 2*PDX_M number of channels for 1 pixel
        vin1 += vInOff1;        //Add input offset
        vin2 += vInOff2;    //Add filter offset
        //MAC
        PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

        PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

        //first8
        quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
        //shift and round
        quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
        //round and shift and saturate
        conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
        //add output zero point
        conv_out += vOutOff;
        //Saturate to 8 bit range output_activation_min to 127
        conv_out = PDX_MIN_MX32(conv_out,vmax);
        conv_out = PDX_MAX_MX32(conv_out,vmin);
        nPixToWrite = MIN(nPixLeft,PDX_M);
        PDX_SAV32_MX16_XP(conv_out, outa, outp, nPixToWrite* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
        PDX_SAPOS_MX16_FP(outa,outp);//flush
        nPixLeft -= nPixToWrite;
        if (nPixLeft>0)
        {
            //last8
            quant_acc = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            nPixToWrite = MIN(nPixLeft,PDX_M);
            PDX_SAV32_MX16_XP(conv_out, outa, outp, nPixToWrite* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX16_FP(outa,outp);//flush
            nPixLeft -= nPixToWrite;
        }
    }
    else
    {
        nPixLeft = nInputLen;
        for (int32_t i = 0; i <= nInputLen; i+= (2*PDX_M))
        {
            acc=0;
            //READ IP
            PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel, skip to adjoining pixel
            PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));//read 2*PDX_M number of channels for 1 pixel
            vin1 += vInOff1;        //Add input offset
            vin2 += vInOff2;    //Add filter offset
            //MAC
            PDX_MULAQW_2MX16(acc,vin1,vin2);//acc contains upto 2*PDX_M channel results for pixel

            PDX_CVT32D_2MX40(last8, first8, acc);    //Converting 40bit results to 32bit to prevent loss of accuracy from packing of 40bit -> 16bit

            //first8
            quant_acc = multiplier * first8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //last8
            quant_acc2 = multiplier * last8;    //Multiplying 2 32-bit vectors and storing result in 80bit vector
            //shift and round
            quant_acc = PDX_SLS_MX80(quant_acc,shift);//saturating left shift, right shift if negative
            //shift and round
            quant_acc2 = PDX_SLS_MX80(quant_acc2,shift);//saturating left shift, right shift if negative
            //round and shift and saturate
            conv_out = PDX_PACKQSRV_MX80   (quant_acc, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //round and shift and saturate
            conv_out2 = PDX_PACKQSRV_MX80   (quant_acc2, round_mode);    //pack 80bit result to 32bit with rounding and saturation.
            //add output zero point
            conv_out += vOutOff;
            //add output zero point
            conv_out2 += vOutOff;
            //Saturate to 8 bit range output_activation_min to 127
            conv_out = PDX_MIN_MX32(conv_out,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);

            nPixLeft -= 2*PDX_M;
            PDX_SAV32_MX16_XP (conv_out, outa, outp, PDX_M* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX16_FP(outa,outp);//flush
            PDX_SAV32_MX16_XP (conv_out2, outa, outp, PDX_M* sizeof(int16_t));//8-way 8-bit signed Aligning vector register variable-length store intrinsic, converting
            PDX_SAPOS_MX16_FP(outa,outp);//flush
        }

    }

}

/**
*******************************************************************************
* Function: adi_sharcfx_elementwise_add_int16
* @brief vectorised elementwise addition for 16-bit integer input
*
* @details vectorised elementwise addition for 16-bit integer input
*
* Parameters:
* @param [in] pInput1 - input buffer 1 (16-bit)
* @param [in] pInput2 - input buffer 2 (16-bit)
* @param [in] nBatches - number of batches
* @param [in] nInputLen - input size
* 
* @param [out] pOutput - output data (16-bit)
*
* @return None
*
*
*******************************************************************************
*/ 
void adi_sharcfx_elementwise_add_int16(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int32_t nBatches,
                                       int32_t nInputLen,
                                       int16_t* pOutput,
                                       int32_t kInt16Max,
                                       int32_t kInt16Min)
{
    xb_vec2Mx16 *inp1 = (xb_vec2Mx16 *)pInput1;
    xb_vec2Mx16 *inp2 = (xb_vec2Mx16 *)pInput2;
    valign ina1, ina2; // define align vector
    ina1=PDX_LA_2MX16_PP (inp1);
    ina2=PDX_LA_2MX16_PP (inp2);

    xb_vecMx32 vmin = kInt16Min;
    xb_vecMx32 vmax = kInt16Max;

    valign outa = PDX_Z_ALIGN();
    xb_vecMx16 *outp = (xb_vecMx16 *)pOutput;
//    valign outa = PDX_LA_2MX16_PP (outp);

    xb_vec2Mx16 vin1, vin2;
    xb_vec2Mx40 sum;
    xb_vecMx32 first8,last8;

    int32_t nPixLeft = nInputLen;

    if(nInputLen%16)
    {
        for (int batch = 0; batch < nBatches; batch++) 
        {
            //reset input pointer for each batch
            inp1 = (xb_vec2Mx16 *)(pInput1 + batch*nInputLen);
            inp2 = (xb_vec2Mx16 *)(pInput2 + batch*nInputLen);
            ina1=PDX_LA_2MX16_PP (inp1);
            ina2=PDX_LA_2MX16_PP (inp2);
            nPixLeft = nInputLen;
            while(nPixLeft > 2*PDX_M)
            {
                //load input
                PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));
                PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));
                //add with saturation
                sum = PDX_ADDW_2MX16(vin1, vin2);
                //
                PDX_CVT32D_2MX40(last8, first8, sum);
                first8 = PDX_MIN_MX32(first8,vmax);
                first8 = PDX_MAX_MX32(first8,vmin);
                last8 = PDX_MIN_MX32(last8,vmax);
                last8 = PDX_MAX_MX32(last8,vmin);
                //save sum and flush vector
                PDX_SAV32_MX16_XP(first8,outa,outp, PDX_M* sizeof(int16_t));
                PDX_SAPOS_MX16_FP(outa,outp);//flush
                PDX_SAV32_MX16_XP(last8,outa,outp, PDX_M* sizeof(int16_t));
                PDX_SAPOS_MX16_FP(outa,outp);//flush
                nPixLeft-= (2*PDX_M);
            }
            //load input
            PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));
            PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));
            //add with saturation
            sum = PDX_ADDW_2MX16 (vin1, vin2);
            //Divide into first 8 and last 8
            PDX_CVT32D_2MX40(last8, first8, sum);
            first8 = PDX_MIN_MX32(first8,vmax);
            first8 = PDX_MAX_MX32(first8,vmin);
            int nPixToWrite = MIN(PDX_M, nPixLeft);
            PDX_SAV32_MX16_XP(first8,outa,outp, nPixToWrite* sizeof(int16_t));
            PDX_SAPOS_MX16_FP(outa,outp);//flush
            nPixLeft-=nPixToWrite;
            if(nPixLeft>0){
                last8 = PDX_MIN_MX32(last8,vmax);
                last8 = PDX_MAX_MX32(last8,vmin);
                PDX_SAV32_MX16_XP(last8,outa,outp, nPixLeft* sizeof(int16_t));
                PDX_SAPOS_MX16_FP(outa,outp);//flush
            }
        }
    }
    else
    {
        for (int batch = 0; batch < nBatches; batch++) 
        {
            //reset input pointer for each batch
            inp1 = (xb_vec2Mx16 *)(pInput1 + batch*nInputLen);
            inp2 = (xb_vec2Mx16 *)(pInput2 + batch*nInputLen);
            ina1=PDX_LA_2MX16_PP (inp1);
            ina2=PDX_LA_2MX16_PP (inp2);
            for (int32_t i = 0; i <= nInputLen; i+= (2*PDX_M))
            {
                //load input
                PDX_LA_2MX16_XP (vin1, ina1, inp1, 2*PDX_M* sizeof(int16_t));
                PDX_LA_2MX16_XP (vin2, ina2, inp2, 2*PDX_M* sizeof(int16_t));
                //add with saturation
                sum = PDX_ADDW_2MX16(vin1, vin2);
                //
                PDX_CVT32D_2MX40(last8, first8, sum);
                first8 = PDX_MIN_MX32(first8,vmax);
                first8 = PDX_MAX_MX32(first8,vmin);
                last8 = PDX_MIN_MX32(last8,vmax);
                last8 = PDX_MAX_MX32(last8,vmin);
                //save sum and flush vector
                PDX_SAV32_MX16_XP(first8,outa,outp, PDX_M* sizeof(int16_t));
                PDX_SAPOS_MX16_FP(outa,outp);//flush
                PDX_SAV32_MX16_XP(last8,outa,outp, PDX_M* sizeof(int16_t));
                PDX_SAPOS_MX16_FP(outa,outp);//flush
            }
        }
    }
}
