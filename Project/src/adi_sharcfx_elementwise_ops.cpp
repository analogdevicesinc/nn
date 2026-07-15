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

/* Named constants for TFLM SaturatingRoundingDoublingHighMul scalar fallback.
 * Reference: tensorflow/lite/kernels/internal/common.h,
 *            SaturatingRoundingDoublingHighMul(). */
#define SRDH_NUDGE_SHIFT  30  /* nudge = 2^30 (= 0.5 * 2^31); selects round-to-nearest */
#define SRDH_DOUBLE_SHIFT 31  /* result = (a * b + nudge) >> 31; extracts high 32-bits of 63-bit product */


/*============= C O D E =============*/
/**
*******************************************************************************
* Function: adi_sharcfx_elementwise_mul_int8
* @brief vectorised elementwise multiplication for 8-bit integer input
*
* @details vectorised elementwise multiplication for 8-bit integer input
*
* Parameters:
* @param [in] pInput1 - input buffer 1 (8-bit)
* @param [in] pInput2 - input buffer 2 (8-bit)
* @param [in] nSize - input size
* @param [in] nQuantizedMultiplier - multiplier, corresponds to TFLM quantization scheme
* @param [in] nQuantizedShift - shift, corresponds to TFLM quantization scheme
* @param [in] nInOffset1 - input offset for input buffer 1
* @param [in] nInOffset2 - input offset for input buffer 2
* @param [in] nOutOffset - output zero-point offset
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
void adi_sharcfx_elementwise_mul_int8(const int8_t* pInput1,
                                       const int8_t* pInput2,
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

    // Broadcast parameters to vector registers
    xb_vec2Mx16 vInOff1 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset1, Lane);
    xb_vec2Mx16 vInOff2 = PDX_REP_2MX16((xb_vec2Mx16)nInOffset2, Lane);

    // Setup input/output pointers
    xb_vec2Mx8 *inp1 = (xb_vec2Mx8 *)pInput1;
    xb_vec2Mx8 *inp2 = (xb_vec2Mx8 *)pInput2;
    xb_vecMx8 *outp = (xb_vecMx8 *)pOutput;

    valign ina1 = PDX_LA_2MX8_PP(inp1);
    valign ina2 = PDX_LA_2MX8_PP(inp2);
    valign outa = PDX_Z_ALIGN();

    xb_vec2Mx16 vin1, vin2;
    xb_vec2Mx40 acc;
    xb_vecMx32 first8, last8;

    // Precompute shift parameters for double-rounding
    int32_t left_shift = (nQuantizedShift > 0) ? nQuantizedShift : 0;
    int32_t right_shift = (nQuantizedShift > 0) ? 0 : -nQuantizedShift;

    // Constants for double-rounding
    const int64_t pos_nudge = (int64_t)1 << SRDH_NUDGE_SHIFT;
    const int64_t neg_nudge = 1 - ((int64_t)1 << SRDH_NUDGE_SHIFT);
    const int32_t INT32_MIN_VAL = -2147483648;
    const int32_t INT32_MAX_VAL = 2147483647;
    int32_t signed_mult = (int32_t)nQuantizedMultiplier;

    int32_t nPixLeft = nInputLen;
    int32_t nPixToWrite;

    // Process 16 elements at a time (2*PDX_M)
    while (nPixLeft >= 2*PDX_M) {
        // Load 16 int8 values and sign-extend to 16-bit
        PDX_LA16_2MX8_XP(vin1, ina1, inp1, 2*PDX_M);
        PDX_LA16_2MX8_XP(vin2, ina2, inp2, 2*PDX_M);

        // Add input offsets
        vin1 += vInOff1;
        vin2 += vInOff2;

        // Element-wise widening multiply (16x16 -> 40-bit)
        acc = PDX_MULW_2MX16(vin1, vin2);

        // Convert 40-bit to two 32-bit vectors
        PDX_CVT32D_2MX40(last8, first8, acc);

        // Extract and process with double-rounding (scalar)
        xb_int32* f8_ptr = (xb_int32*)&first8;
        xb_int32* l8_ptr = (xb_int32*)&last8;
        xb_vecMx32 result1, result2;
        xb_int32* r1_ptr = (xb_int32*)&result1;
        xb_int32* r2_ptr = (xb_int32*)&result2;

        for (int j = 0; j < 8; j++) {
            int32_t product = (int32_t)f8_ptr[j];

            // Apply left shift
            if (left_shift > 0) product = product << left_shift;

            // SaturatingRoundingDoublingHighMul
            bool overflow = (product == signed_mult) && (product == INT32_MIN_VAL);
            int32_t srdh_result;
            if (overflow) {
                srdh_result = INT32_MAX_VAL;
            } else {
                int64_t ab = (int64_t)product * (int64_t)signed_mult;
                int64_t nudge = (ab >= 0) ? pos_nudge : neg_nudge;
                srdh_result = (int32_t)((ab + nudge) / ((int64_t)1 << SRDH_DOUBLE_SHIFT));
            }

            // RoundingDivideByPOT
            int32_t quantized;
            if (right_shift > 0) {
                int32_t mask = (1 << right_shift) - 1;
                int32_t threshold = (mask >> 1) + (srdh_result < 0 ? 1 : 0);
                int32_t remainder = srdh_result & mask;
                quantized = (srdh_result >> right_shift) + (remainder > threshold ? 1 : 0);
            } else {
                quantized = srdh_result;
            }

            // Add offset and clamp
            int32_t result = quantized + nOutOffset;
            result = (result < output_activation_min) ? output_activation_min : result;
            result = (result > output_activation_max) ? output_activation_max : result;
            r1_ptr[j] = (xb_int32)result;
        }

        for (int j = 0; j < 8; j++) {
            int32_t product = (int32_t)l8_ptr[j];

            if (left_shift > 0) product = product << left_shift;

            bool overflow = (product == signed_mult) && (product == INT32_MIN_VAL);
            int32_t srdh_result;
            if (overflow) {
                srdh_result = INT32_MAX_VAL;
            } else {
                int64_t ab = (int64_t)product * (int64_t)signed_mult;
                int64_t nudge = (ab >= 0) ? pos_nudge : neg_nudge;
                srdh_result = (int32_t)((ab + nudge) / ((int64_t)1 << SRDH_DOUBLE_SHIFT));
            }

            int32_t quantized;
            if (right_shift > 0) {
                int32_t mask = (1 << right_shift) - 1;
                int32_t threshold = (mask >> 1) + (srdh_result < 0 ? 1 : 0);
                int32_t remainder = srdh_result & mask;
                quantized = (srdh_result >> right_shift) + (remainder > threshold ? 1 : 0);
            } else {
                quantized = srdh_result;
            }

            int32_t result = quantized + nOutOffset;
            result = (result < output_activation_min) ? output_activation_min : result;
            result = (result > output_activation_max) ? output_activation_max : result;
            r2_ptr[j] = (xb_int32)result;
        }

        // Store 16 int8 outputs (8 at a time)
        PDX_SAV32_MX8_XP(result1, outa, outp, PDX_M);
        PDX_SAPOS_MX8_FP(outa, outp);
        PDX_SAV32_MX8_XP(result2, outa, outp, PDX_M);
        PDX_SAPOS_MX8_FP(outa, outp);

        nPixLeft -= 2*PDX_M;
    }

    // Handle remaining elements
    if (nPixLeft > 0) {
        // Load remaining elements
        PDX_LA16_2MX8_XP(vin1, ina1, inp1, 0);
        PDX_LA16_2MX8_XP(vin2, ina2, inp2, 0);

        vin1 += vInOff1;
        vin2 += vInOff2;

        acc = PDX_MULW_2MX16(vin1, vin2);
        PDX_CVT32D_2MX40(last8, first8, acc);

        xb_int32* f8_ptr = (xb_int32*)&first8;
        xb_int32* l8_ptr = (xb_int32*)&last8;
        xb_vecMx32 result1, result2;
        xb_int32* r1_ptr = (xb_int32*)&result1;
        xb_int32* r2_ptr = (xb_int32*)&result2;

        for (int j = 0; j < 8; j++) {
            int32_t product = (int32_t)f8_ptr[j];

            if (left_shift > 0) product = product << left_shift;

            bool overflow = (product == signed_mult) && (product == INT32_MIN_VAL);
            int32_t srdh_result;
            if (overflow) {
                srdh_result = INT32_MAX_VAL;
            } else {
                int64_t ab = (int64_t)product * (int64_t)signed_mult;
                int64_t nudge = (ab >= 0) ? pos_nudge : neg_nudge;
                srdh_result = (int32_t)((ab + nudge) / ((int64_t)1 << SRDH_DOUBLE_SHIFT));
            }

            int32_t quantized;
            if (right_shift > 0) {
                int32_t mask = (1 << right_shift) - 1;
                int32_t threshold = (mask >> 1) + (srdh_result < 0 ? 1 : 0);
                int32_t remainder = srdh_result & mask;
                quantized = (srdh_result >> right_shift) + (remainder > threshold ? 1 : 0);
            } else {
                quantized = srdh_result;
            }

            int32_t result = quantized + nOutOffset;
            result = (result < output_activation_min) ? output_activation_min : result;
            result = (result > output_activation_max) ? output_activation_max : result;
            r1_ptr[j] = (xb_int32)result;
        }

        nPixToWrite = MIN(nPixLeft, PDX_M);
        PDX_SAV32_MX8_XP(result1, outa, outp, nPixToWrite);
        PDX_SAPOS_MX8_FP(outa, outp);
        nPixLeft -= nPixToWrite;

        if (nPixLeft > 0) {
            for (int j = 0; j < 8; j++) {
                int32_t product = (int32_t)l8_ptr[j];

                if (left_shift > 0) product = product << left_shift;

                bool overflow = (product == signed_mult) && (product == INT32_MIN_VAL);
                int32_t srdh_result;
                if (overflow) {
                    srdh_result = INT32_MAX_VAL;
                } else {
                    int64_t ab = (int64_t)product * (int64_t)signed_mult;
                    int64_t nudge = (ab >= 0) ? pos_nudge : neg_nudge;
                    srdh_result = (int32_t)((ab + nudge) / ((int64_t)1 << SRDH_DOUBLE_SHIFT));
                }

                int32_t quantized;
                if (right_shift > 0) {
                    int32_t mask = (1 << right_shift) - 1;
                    int32_t threshold = (mask >> 1) + (srdh_result < 0 ? 1 : 0);
                    int32_t remainder = srdh_result & mask;
                    quantized = (srdh_result >> right_shift) + (remainder > threshold ? 1 : 0);
                } else {
                    quantized = srdh_result;
                }

                int32_t result = quantized + nOutOffset;
                result = (result < output_activation_min) ? output_activation_min : result;
                result = (result > output_activation_max) ? output_activation_max : result;
                r2_ptr[j] = (xb_int32)result;
            }

            nPixToWrite = MIN(nPixLeft, PDX_M);
            PDX_SAV32_MX8_XP(result2, outa, outp, nPixToWrite);
            PDX_SAPOS_MX8_FP(outa, outp);
        }
    }
}

/**
*******************************************************************************
* Function: adi_sharcfx_elementwise_mul_int16_input_int8_output
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
* @param [in] nOutOffset - output zero-point offset
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
void adi_sharcfx_elementwise_mul_int16_input_int8_output(  const int16_t* pInput1,
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
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);

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
        for (int32_t i = 0; i < nInputLen; i+= (2*PDX_M))
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
            conv_out = PDX_MAX_MX32(conv_out,vmin);
            conv_out2 = PDX_MIN_MX32(conv_out2,vmax);
            conv_out2 = PDX_MAX_MX32(conv_out2,vmin);

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
* @param [in] nOutOffset - output zero-point offset
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

/**
 *******************************************************************************
 * Function: adi_sharcfx_elementwise_add_int8
 * @brief Vectorised element-wise addition for int8 inputs (TFLM quantization scheme).
 *
 * @details Vectorised implementation of element-wise addition for two 8-bit
 *          integer input tensors following the TFLM double-quantization scheme:
 *          each input is independently scaled and left-shifted before summing,
 *          then the result is re-quantized and clamped to the activation range.
 *
 * Parameters:
 * @param [in]  pInput1                 First input buffer (int8).
 * @param [in]  pInput2                 Second input buffer (int8).
 * @param [out] pOutput                 Output buffer (int8).
 * @param [in]  nSize                   Number of elements.
 * @param [in]  input1_offset           Zero-point offset for input 1.
 * @param [in]  input1_multiplier       Quantization multiplier for input 1.
 * @param [in]  input1_shift            Quantization shift for input 1.
 * @param [in]  input2_offset           Zero-point offset for input 2.
 * @param [in]  input2_multiplier       Quantization multiplier for input 2.
 * @param [in]  input2_shift            Quantization shift for input 2.
 * @param [in]  left_shift              Shared left-shift applied before scaling.
 * @param [in]  output_multiplier       Output quantization multiplier.
 * @param [in]  output_shift            Output quantization shift.
 * @param [in]  output_offset           Output zero-point offset.
 * @param [in]  quantized_activation_min  Activation minimum clamp.
 * @param [in]  quantized_activation_max  Activation maximum clamp.
 * @return None
 *******************************************************************************
 */
void adi_sharcfx_elementwise_add_int8(
									const int8_t* pInput1,
									const int8_t* pInput2,
									int8_t* pOutput,
									int32_t nSize,
									int32_t input1_offset,
									int32_t input1_multiplier,
									int32_t input1_shift,
									int32_t input2_offset,
									int32_t input2_multiplier,
									int32_t input2_shift,
									int32_t left_shift,
									int32_t output_multiplier,
									int32_t output_shift,
									int32_t output_offset,
									int32_t quantized_activation_min,
									int32_t quantized_activation_max)
{
  // Vector registers for input/output
  xb_vecMx32  vin1, vin2, vout;
  xb_vecMx80  acc1, acc2, sum;
  xb_vecMx8 *inp1 = (xb_vecMx8 *)pInput1;
  xb_vecMx8 *inp2 = (xb_vecMx8 *)pInput2;
  xb_vecMx8 *outp = (xb_vecMx8 *)pOutput;

  // Alignment vectors for unaligned loads/stores
  valign ina1, ina2, outa;
  ina1 = PDX_LA_MX8_PP(inp1);   // prime alignment for input1 (corrected from 2MX8)
  ina2 = PDX_LA_MX8_PP(inp2);   // prime alignment for input2 (corrected from 2MX8)
  outa = PDX_Z_ALIGN();         // prime alignment for output

  // Broadcast parameters to vector registers
	xb_vecMx32 vInput1Offset = input1_offset;
	xb_vecMx32 vInput2Offset = input2_offset;
  xb_vecMx32 vInput1Mult = input1_multiplier;
  xb_vecMx32 vInput2Mult = input2_multiplier;
  xb_vecMx32 vOutputMult = output_multiplier;
  xb_vecMx32 vLeftShift = left_shift;
  xb_vecMx80 vOutOffset = output_offset;
  xb_vecMx32 vActMin = quantized_activation_min;
  xb_vecMx32 vActMax = quantized_activation_max;

  // Pre-compute total shifts for MultiplyByQuantizedMultiplier
  // Reference: total_shift = 31 - shift, result = (x * mult + round) >> total_shift
  // Since shifts are negative, total_shift = 31 - shift > 31
  int32_t input1_total_shift = 31 - input1_shift;
  int32_t input2_total_shift = 31 - input2_shift;
  int32_t output_total_shift = 31 - output_shift;

  // Pre-compute rounding bias: round = 1 << (total_shift - 1)
  // PDX_SRS_MX80 is a saturating shift, NOT rounding - we must add bias manually
  xb_vecMx80 vInput1Round = (int64_t)1 << (input1_total_shift - 1);
  xb_vecMx80 vInput2Round = (int64_t)1 << (input2_total_shift - 1);
  xb_vecMx80 vOutputRound = (int64_t)1 << (output_total_shift - 1);

  int32_t nElementsLeft = nSize;

  // Main vectorized loop - process PDX_M elements at a time
  if(nSize % PDX_M)  // Non-multiple of PDX_M
  {
	  while(nElementsLeft > PDX_M)
	  {
		// Load PDX_M int8 elements and convert to 32-bit
		PDX_LA32_MX8_XP(vin1, ina1, inp1, PDX_M);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, PDX_M);

		// De-quantize input1: offset -> left shift -> fixed-point multiply with rounding
		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);                   // add offset (reference: input1_offset + x)
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);                   // add offset (reference: input2_offset + y)
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);                      // shift left with saturation
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);                      // shift left with saturation

		// MultiplyByQuantizedMultiplierSmallerThanOneExp for input1:
		// Reference: result = (x * mult + round) >> total_shift
		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);                    // multiply with widen to 80-bit
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);                    // add rounding bias
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);              // shift right (saturating)

		// MultiplyByQuantizedMultiplierSmallerThanOneExp for input2:
		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);                    // multiply with widen to 80-bit
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);                    // add rounding bias
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);              // shift right (saturating)

		// Add the two scaled inputs
		sum = PDX_ADD_MX80(acc1, acc2);

		// Re-quantize: MultiplyByQuantizedMultiplierSmallerThanOneExp for output
		// First pack 80-bit sum to 32-bit, then multiply (PDX_MULW_MX32 requires 32-bit inputs)
		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);                // 80-bit to 32-bit
		sum = PDX_MULW_MX32(sum32, vOutputMult);                    // 32-bit * 32-bit -> 80-bit
		sum = PDX_ADD_MX80(sum, vOutputRound);                      // add rounding bias
		sum = PDX_SRS_MX80(sum, output_total_shift);                // shift right (saturating)

		// Add output offset (need to widen offset to 80-bit)
		sum = PDX_ADD_MX80(sum, vOutOffset);

		// Convert back to 32-bit with saturation
		vout = PDX_PACKSIV_MX80(sum, 0);

		// Clamp to activation range
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		// Convert 32-bit to 8-bit and store
		PDX_SAV32_MX8_XP(vout, outa, outp, PDX_M * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);  // flush
		nElementsLeft -= PDX_M;
	  }

	  // Handle remaining elements (< PDX_M)
	  if(nElementsLeft > 0)
	  {
		// Load remaining int8 elements
		PDX_LA32_MX8_XP(vin1, ina1, inp1, 0);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, 0);

		// De-quantize
		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);

		// MultiplyByQuantizedMultiplierSmallerThanOneExp for inputs
		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);                    // add rounding bias
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);
		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);                    // add rounding bias
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);

		// Add
		sum = PDX_ADD_MX80(acc1, acc2);

		// Re-quantize: pack 80-bit sum to 32-bit before multiply
		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);                // 80-bit to 32-bit
		sum = PDX_MULW_MX32(sum32, vOutputMult);                    // 32-bit * 32-bit -> 80-bit
		sum = PDX_ADD_MX80(sum, vOutputRound);                      // add rounding bias
		sum = PDX_SRS_MX80(sum, output_total_shift);
		sum = PDX_ADD_MX80(sum, vOutOffset);

		// Pack and clamp
		vout = PDX_PACKSIV_MX80(sum, 0);
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		// Store remaining
		int nPixToWrite = nElementsLeft;  // Removed MIN - nElementsLeft is already < PDX_M
		PDX_SAV32_MX8_XP(vout, outa, outp, nPixToWrite * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);  // flush
	  }
  }
  else  // Size is exact multiple of PDX_M
  {
	  for (int32_t i = 0; i < nSize; i += PDX_M)  // Fixed: changed <= to <
	  {
		// Load PDX_M int8 elements
		PDX_LA32_MX8_XP(vin1, ina1, inp1, PDX_M);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, PDX_M);

		// De-quantize
		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);                   // add offset (reference: input1_offset + x)
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);                   // add offset (reference: input2_offset + y)
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);

		// MultiplyByQuantizedMultiplierSmallerThanOneExp for inputs
		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);                    // add rounding bias
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);
		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);                    // add rounding bias
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);

		// Add
		sum = PDX_ADD_MX80(acc1, acc2);

		// Re-quantize (MultiplyByQuantizedMultiplierSmallerThanOneExp)
		// Pack 80-bit sum to 32-bit before multiply (PDX_MULW_MX32 requires 32-bit inputs)
		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);                // 80-bit to 32-bit
		sum = PDX_MULW_MX32(sum32, vOutputMult);                    // 32-bit * 32-bit -> 80-bit
		sum = PDX_ADD_MX80(sum, vOutputRound);                      // add rounding bias
		sum = PDX_SRS_MX80(sum, output_total_shift);
		sum = PDX_ADD_MX80(sum, vOutOffset);

		// Pack and clamp
		vout = PDX_PACKSIV_MX80(sum, 0);
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		// Store
		PDX_SAV32_MX8_XP(vout, outa, outp, PDX_M * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);  // flush
	  }
  }
}

/**
 *******************************************************************************
 * Function: adi_sharcfx_elementwise_sub_int8
 * @brief Vectorised element-wise subtraction for int8 inputs (TFLM quantization scheme).
 *
 * @details Vectorised implementation of element-wise subtraction for two 8-bit
 *          integer input tensors following the TFLM double-quantization scheme.
 *          pInput1 is the minuend and pInput2 is the subtrahend.
 *          Each input is independently scaled and left-shifted, the difference
 *          is re-quantized and clamped to the activation range.
 *
 * Parameters:
 * @param [in]  pInput1                 Minuend input buffer (int8).
 * @param [in]  pInput2                 Subtrahend input buffer (int8).
 * @param [out] pOutput                 Output buffer (int8).
 * @param [in]  nSize                   Number of elements.
 * @param [in]  input1_offset           Zero-point offset for input 1.
 * @param [in]  input1_multiplier       Quantization multiplier for input 1.
 * @param [in]  input1_shift            Quantization shift for input 1.
 * @param [in]  input2_offset           Zero-point offset for input 2.
 * @param [in]  input2_multiplier       Quantization multiplier for input 2.
 * @param [in]  input2_shift            Quantization shift for input 2.
 * @param [in]  left_shift              Shared left-shift applied before scaling.
 * @param [in]  output_multiplier       Output quantization multiplier.
 * @param [in]  output_shift            Output quantization shift.
 * @param [in]  output_offset           Output zero-point offset.
 * @param [in]  quantized_activation_min  Activation minimum clamp.
 * @param [in]  quantized_activation_max  Activation maximum clamp.
 * @return None
 *******************************************************************************
 */
void adi_sharcfx_elementwise_sub_int8(
									const int8_t* pInput1,
									const int8_t* pInput2,
									int8_t* pOutput,
									int32_t nSize,
									int32_t input1_offset,
									int32_t input1_multiplier,
									int32_t input1_shift,
									int32_t input2_offset,
									int32_t input2_multiplier,
									int32_t input2_shift,
									int32_t left_shift,
									int32_t output_multiplier,
									int32_t output_shift,
									int32_t output_offset,
									int32_t quantized_activation_min,
									int32_t quantized_activation_max)
{
  xb_vecMx32  vin1, vin2, vout;
  xb_vecMx80  acc1, acc2, sum;
  xb_vecMx8 *inp1 = (xb_vecMx8 *)pInput1;
  xb_vecMx8 *inp2 = (xb_vecMx8 *)pInput2;
  xb_vecMx8 *outp = (xb_vecMx8 *)pOutput;

  valign ina1, ina2, outa;
  ina1 = PDX_LA_MX8_PP(inp1);
  ina2 = PDX_LA_MX8_PP(inp2);
  outa = PDX_Z_ALIGN();

  xb_vecMx32 vInput1Offset = input1_offset;
  xb_vecMx32 vInput2Offset = input2_offset;
  xb_vecMx32 vInput1Mult = input1_multiplier;
  xb_vecMx32 vInput2Mult = input2_multiplier;
  xb_vecMx32 vOutputMult = output_multiplier;
  xb_vecMx32 vLeftShift = left_shift;
  xb_vecMx80 vOutOffset = output_offset;
  xb_vecMx32 vActMin = quantized_activation_min;
  xb_vecMx32 vActMax = quantized_activation_max;

  // Compute total shifts: total_shift = 31 - shift
  int32_t input1_total_shift = 31 - input1_shift;
  int32_t input2_total_shift = 31 - input2_shift;
  int32_t output_total_shift = 31 - output_shift;

  // Rounding bias: round = 1 << (total_shift - 1)
  xb_vecMx80 vInput1Round = (int64_t)1 << (input1_total_shift - 1);
  xb_vecMx80 vInput2Round = (int64_t)1 << (input2_total_shift - 1);
  xb_vecMx80 vOutputRound = (int64_t)1 << (output_total_shift - 1);

  int32_t nElementsLeft = nSize;

  if(nSize % PDX_M)
  {
	  while(nElementsLeft > PDX_M)
	  {
		PDX_LA32_MX8_XP(vin1, ina1, inp1, PDX_M);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, PDX_M);

		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);

		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);

		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);

		sum = PDX_SUB_MX80(acc1, acc2);

		// Pack 80-bit to 32-bit before multiply (PDX_MULW_MX32 requires 32-bit inputs)
		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);
		sum = PDX_MULW_MX32(sum32, vOutputMult);
		sum = PDX_ADD_MX80(sum, vOutputRound);
		sum = PDX_SRS_MX80(sum, output_total_shift);

		sum = PDX_ADD_MX80(sum, vOutOffset);

		vout = PDX_PACKSIV_MX80(sum, 0);
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		PDX_SAV32_MX8_XP(vout, outa, outp, PDX_M * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);
		nElementsLeft -= PDX_M;
	  }

	  // Remainder
	  if(nElementsLeft > 0)
	  {
		PDX_LA32_MX8_XP(vin1, ina1, inp1, 0);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, 0);

		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);

		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);
		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);

		sum = PDX_SUB_MX80(acc1, acc2);

		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);
		sum = PDX_MULW_MX32(sum32, vOutputMult);
		sum = PDX_ADD_MX80(sum, vOutputRound);
		sum = PDX_SRS_MX80(sum, output_total_shift);
		sum = PDX_ADD_MX80(sum, vOutOffset);

		vout = PDX_PACKSIV_MX80(sum, 0);
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		PDX_SAV32_MX8_XP(vout, outa, outp, nElementsLeft * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);
	  }
  }
  else
  {
	  for (int32_t i = 0; i < nSize; i += PDX_M)
	  {
		PDX_LA32_MX8_XP(vin1, ina1, inp1, PDX_M);
		PDX_LA32_MX8_XP(vin2, ina2, inp2, PDX_M);

		vin1 = PDX_ADD_MX32(vin1, vInput1Offset);
		vin2 = PDX_ADD_MX32(vin2, vInput2Offset);
		vin1 = PDX_SLS_MX32(vin1, vLeftShift);
		vin2 = PDX_SLS_MX32(vin2, vLeftShift);

		acc1 = PDX_MULW_MX32(vin1, vInput1Mult);
		acc1 = PDX_ADD_MX80(acc1, vInput1Round);
		acc1 = PDX_SRS_MX80(acc1, input1_total_shift);
		acc2 = PDX_MULW_MX32(vin2, vInput2Mult);
		acc2 = PDX_ADD_MX80(acc2, vInput2Round);
		acc2 = PDX_SRS_MX80(acc2, input2_total_shift);

		sum = PDX_SUB_MX80(acc1, acc2);

		xb_vecMx32 sum32 = PDX_PACKSIV_MX80(sum, 0);
		sum = PDX_MULW_MX32(sum32, vOutputMult);
		sum = PDX_ADD_MX80(sum, vOutputRound);
		sum = PDX_SRS_MX80(sum, output_total_shift);
		sum = PDX_ADD_MX80(sum, vOutOffset);

		vout = PDX_PACKSIV_MX80(sum, 0);
		vout = PDX_MIN_MX32(vout, vActMax);
		vout = PDX_MAX_MX32(vout, vActMin);

		PDX_SAV32_MX8_XP(vout, outa, outp, PDX_M * sizeof(int8_t));
		PDX_SAPOS_MX8_FP(outa, outp);
	  }
  }
}

