/**
********************************************************************************
*
* @file: adi_sharcfx_nn.h
*
* @brief: header file for adi NN library
*
* @details: primary header file for adi NN library
*
*******************************************************************************
 Copyright(c) 2024 Analog Devices, Inc. All Rights Reserved. This software is
 proprietary & confidential to Analog Devices, Inc. and its licensors. By using
 this software you agree to the terms of the associated Analog Devices License
 Agreement.
*******************************************************************************
*/

#ifndef ADI_SHARCFX_NN_H_
#define ADI_SHARCFX_NN_H_


/*============= I N C L U D E S =============*/
#include "adi_sharcfx_common.h"
#include "libdsp_types.h"
#ifdef ADI_DEBUG
#include "debug.h"
#endif

/*============= D E F I N E S =============*/
/*General defines for functions*/
#if PDX_M==8
#define PDX_2M  (PDX_M*2)
#define PDX_4M  (PDX_M*4)
#define LOG2_PDX_2M    (4)    /* log2(PDX_M * 2) = log2(16) = 4 when PDX_M == 8 */
#endif

/* Optimisation-enable flags.
 * Flag                           Enables
 * ----                           -------
 * USE_OPTIMIZED_DEPTHCONV        ADI vectorised depthwise conv2d (int8)
 * USE_OPTIMIZED_3x3_CONV         ADI vectorised 3x3 conv2d (int8)
 * USE_OPTIMIZED_1x1_CONV         ADI vectorised 1x1 conv2d (int8 / int16)
 * USE_OPTIMIZED_FC               ADI vectorised fully-connected (int8 / int16)
 * USE_OPTIMIZED_RELU             ADI vectorised ReLU (int8)
 * USE_OPTIMIZED_LSTM             ADI vectorised LSTM
 * USE_OPTIMIZED_ELEMENTWISE_OPS  ADI vectorised elementwise mul / add / sub (int8)
 * USE_OPTIMIZED_MAXPOOL          ADI vectorised max-pooling (int8)
 * USE_OPTIMIZED_TANH_INT8        ADI LUT tanh int8 — limited resolution LUT implementation.
 *                                 Resolution is documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 *                                 In testing, the resolution is not an issue for most applications. However, the quantisation
 *                                 error can accumulate across layers in certain models and alter inference results
 *                                 (e.g. observed in the DFN denoiser). Disabled by default; evaluate on your model before enabling.
 * USE_OPTIMIZED_LOGISTIC_INT8    ADI LUT logistic int8 — limited resolution LUT implementation.
 *                                 Resolution is documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 *                                 In testing, the resolution is not an issue for most applications. However, the quantisation
 *                                 error can accumulate across layers in certain models and alter inference results
 *                                 (e.g. observed in the DFN denoiser). Disabled by default; evaluate on your model before enabling.
 * USE_OPTIMIZED_TANH_INT16       ADI LUT tanh int16  — used internally by the TFLM LSTM block (int16 path).
 *                                 Note: currently supported apps do not include an LSTM block in the TFLite/TFLM model,
 *                                 so this path is not exercised at runtime. Tested with existing test cases; passes.
 *                                 Test coverage is limited. Tolerance documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 * USE_OPTIMIZED_LOGISTIC_INT16   ADI LUT logistic int16 — used internally by the TFLM LSTM block (int16 path).
 *                                 Note: currently supported apps do not include an LSTM block in the TFLite/TFLM model,
 *                                 so this path is not exercised at runtime. Tested with existing test cases; passes.
 *                                 Test coverage is limited. Tolerance documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 */
#define USE_OPTIMIZED_DEPTHCONV
#define USE_OPTIMIZED_3x3_CONV
#define USE_OPTIMIZED_1x1_CONV
#define USE_OPTIMIZED_FC
#define USE_OPTIMIZED_RELU
#define USE_OPTIMIZED_LSTM
#define USE_OPTIMIZED_ELEMENTWISE_OPS
// #define USE_OPTIMIZED_MAXPOOL
// #define USE_OPTIMIZED_TANH_INT8         /* Limited resolution LUT. Not an issue for most apps but error can propagate in some models (e.g. DFN denoiser) and alter results. See ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf. Evaluate on your model before enabling. */
// #define USE_OPTIMIZED_LOGISTIC_INT8     /* Limited resolution LUT. Not an issue for most apps but error can propagate in some models (e.g. DFN denoiser) and alter results. See ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf. Evaluate on your model before enabling. */
// #define USE_OPTIMIZED_TANH_INT16        /* TFLM LSTM int16 path only. Not exercised by current apps (no LSTM in TFLite/TFLM model). Limited test coverage. See ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf for tolerance. */
// #define USE_OPTIMIZED_LOGISTIC_INT16    /* TFLM LSTM int16 path only. Not exercised by current apps (no LSTM in TFLite/TFLM model). Limited test coverage. See ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf for tolerance. */

// #define USE_REORDERED_WEIGHTS_SCHEME		/*will enable the optimizations for FC and conv1x1 layers that expect the reordered weights. To be used when using model files with reordered weights. */

/* Enable or disable profiling */
#ifdef DISPLAY_CYCLE_COUNTS
#define __PRE_FX_COMPATIBILITY
#define DO_CYCLE_COUNTS
#include <cycle_count.h>
#endif
/*============= D A T A =============*/


/*============= F U N C T I O N P R O T O T Y P E S =============*/

/**
 * @brief Optimised max-pooling for int8 input.
 * @param input_y      Input height.
 * @param input_x      Input width.
 * @param output_y     Output height.
 * @param output_x     Output width.
 * @param stride_y     Stride in the y direction.
 * @param stride_x     Stride in the x direction.
 * @param kernel_y     Kernel height.
 * @param kernel_x     Kernel width.
 * @param pad_y        Padding in y.
 * @param pad_x        Padding in x.
 * @param act_min      Activation minimum clamp value.
 * @param act_max      Activation maximum clamp value.
 * @param ch_src       Number of input channels.
 * @param src          Pointer to input data (int8).
 * @param dst          Pointer to output data (int8).
 * @return None
 */
void adi_sharcfx_maxpool_int8(const int32_t input_y,
                              const int32_t input_x,
                              const int32_t output_y,
                              const int32_t output_x,
                              const int32_t stride_y,
                              const int32_t stride_x,
                              const int32_t kernel_y,
                              const int32_t kernel_x,
                              const int32_t pad_y,
                              const int32_t pad_x,
                              const int32_t act_min,
                              const int32_t act_max,
                              const int32_t ch_src,
                              const int8_t  *src,
                              int8_t        *dst);

/**
 * @brief Optimised 1x1 conv2d with pre-reordered weights (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Pre-reordered weight buffer (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nBatches             Batch count.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nKernelHeight        Kernel height.
 * @param nKernelWidth         Kernel width.
 * @param nNumKernels          Number of kernels.
 * @param nInputWidth          Input width.
 * @param nInputHeight         Input height.
 * @param stride_height        Stride in height direction.
 * @param stride_width         Stride in width direction.
 * @param nPadHeight           Padding in height direction.
 * @param nPadWidth            Padding in width direction.
 * @param nOutHeight           Output height.
 * @param nOutWidth            Output width.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @param nFilterZeroPoint     Filter zero point.
 * @param nActMin              Activation minimum clamp.
 * @param nActMax              Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_conv2d_kernel1x1_int8_reordered_weights(   const int8_t* pInputBuffer,
                                                            const int8_t* pWeightsBuffer,
                                                            const int32_t* pBiasBuffer,
                                                            int8_t* pOutputBuffer,
                                                            int32_t nBatches,
                                                            int32_t nInChannels,
                                                            int32_t nOutChannels,
                                                            int32_t nKernelHeight,
                                                            int32_t nKernelWidth,
                                                            int32_t nNumKernels,
                                                            int32_t nInputWidth,
                                                            int32_t nInputHeight,
                                                            int32_t stride_height,
                                                            int32_t stride_width,
                                                            int32_t nPadHeight,
                                                            int32_t nPadWidth,
                                                            int32_t nOutHeight,
                                                            int32_t nOutWidth,
                                                            int32_t *pQuantizedMultiplier,
                                                            int32_t *pQuantizedShift,
                                                            int32_t pInZeroPoint,
                                                            int32_t pOutZeroPoint,
                                                            int32_t nFilterZeroPoint,
                                                            int32_t nActMin,
                                                            int32_t nActMax);

/**
 * @brief Optimised stride-2 depthwise conv2d, non-interleaved layout (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pOutputBuffer        Output feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param nInputWidth          Input width.
 * @param nDepthMult           Depth multiplier.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nKernelSize          Kernel size.
 * @param nTotalPadding        Total padding applied.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_depthconv2d_stride2_noninterleaved_int8(const int8_t *pInputBuffer,
                                                         int8_t *pOutputBuffer,
                                                         const int8_t *pWeightsBuffer,
                                                         const int32_t *pBiasBuffer,
                                                         int32_t nInputWidth,
                                                         int32_t nDepthMult,
                                                         int32_t nInChannels,
                                                         int32_t nOutChannels,
                                                         int8_t nKernelSize,
                                                         int8_t nTotalPadding,
                                                         uint32_t *pQuantizedMultiplier,
                                                         int32_t *pQuantizedShift,
                                                         int32_t pInZeroPoint,
                                                         int32_t pOutZeroPoint);

/**
 * @brief Optimised stride-1 depthwise conv2d, non-interleaved layout (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pOutputBuffer        Output feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param nInputWidth          Input width.
 * @param nDepthMult           Depth multiplier.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nKernelSize          Kernel size.
 * @param nTotalPadding        Total padding applied.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_depthconv2d_stride1_noninterleaved_int8(const int8_t *pInputBuffer,
                                                         int8_t *pOutputBuffer,
                                                         const int8_t *pWeightsBuffer,
                                                         const int32_t *pBiasBuffer,
                                                         int32_t nInputWidth,
                                                         int32_t nDepthMult,
                                                         int32_t nInChannels,
                                                         int32_t nOutChannels,
                                                         int8_t nKernelSize,
                                                         int8_t nTotalPadding,
                                                         uint32_t *pQuantizedMultiplier,
                                                         int32_t *pQuantizedShift,
                                                         int32_t pInZeroPoint,
                                                         int32_t pOutZeroPoint);

/**
 * @brief Optimised stride-2, 8x10 kernel depthwise conv2d, non-interleaved (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nInputWidth          Input width.
 * @param nInputLength         Input length.
 * @param nInChannels          Number of input channels.
 * @param nOutputWidth         Output width.
 * @param nOutputLength        Output length.
 * @param nOutChannels         Number of output channels.
 * @param nKernelWidth         Kernel width.
 * @param nKernelLength        Kernel length.
 * @param nPaddingWidth        Padding width.
 * @param nPaddingLength       Padding length.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_depthconv2d_stride2_kernel8x10_noninterleaved_int8(const int8_t *pInputBuffer,
                                                                    const int8_t *pWeightsBuffer,
                                                                    const int32_t *pBiasBuffer,
                                                                    int8_t *pOutputBuffer,
                                                                    int32_t nInputWidth,
                                                                    int32_t nInputLength,
                                                                    int32_t nInChannels,
                                                                    int32_t nOutputWidth,
                                                                    int32_t nOutputLength,
                                                                    int32_t nOutChannels,
                                                                    int8_t nKernelWidth,
                                                                    int8_t nKernelLength,
                                                                    int8_t nPaddingWidth,
                                                                    int8_t nPaddingLength,
                                                                    uint32_t *pQuantizedMultiplier,
                                                                    int32_t *pQuantizedShift,
                                                                    int32_t pInZeroPoint,
                                                                    int32_t pOutZeroPoint,
                                                                    int32_t output_activation_min,
                                                                    int32_t output_activation_max);

/**
 * @brief Optimised general depthwise conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pOutputBuffer        Output feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param nInputWidth          Input width.
 * @param nInputHeight         Input height.
 * @param nDepthMult           Depth multiplier.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nKernelSizeWidth     Kernel width.
 * @param nKernelSizeHeight    Kernel height.
 * @param nTotalPaddingWidth   Total padding in width.
 * @param nTotalPaddingHeight  Total padding in height.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @param nStrideWidth         Stride in width direction.
 * @param nStrideHeight        Stride in height direction.
 * @param nActMin              Activation minimum clamp.
 * @param nActMax              Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_depthconv2d_int8(const int8_t *pInputBuffer,
                                  int8_t *pOutputBuffer,
                                  const int8_t *pWeightsBuffer,
                                  const int32_t *pBiasBuffer,
                                  int32_t nInputWidth,
                                  int32_t nInputHeight,
                                  int32_t nDepthMult,
                                  int32_t nInChannels,
                                  int32_t nOutChannels,
                                  int32_t nKernelSizeWidth,
                                  int32_t nKernelSizeHeight,
                                  int32_t nTotalPaddingWidth,
                                  int32_t nTotalPaddingHeight,
                                  int32_t *pQuantizedMultiplier,
                                  int32_t *pQuantizedShift,
                                  int32_t pInZeroPoint,
                                  int32_t pOutZeroPoint,
                                  int32_t nStrideWidth,
                                  int32_t nStrideHeight,
                                  int32_t nActMin,
                                  int32_t nActMax);

/**
 * @brief Optimised fully-connected layer for int8 input.
 * @param pInputBuffer         Input activation buffer (int8).
 * @param pWeightsBuffer       Weight buffer (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output buffer (int8).
 * @param nFilterDepth         Number of input features (filter depth).
 * @param nOutsize             Number of output neurons.
 * @param nBatches             Batch count.
 * @param nQuantizedMultiplier Quantization multiplier.
 * @param nQuantizedShift      Quantization shift.
 * @param nInputOffset         Input zero point offset.
 * @param nFilterOffset        Filter zero point offset.
 * @param nOutputOffset        Output zero point offset.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_fully_connected_int8(const int8_t* pInputBuffer,
                                      const int8_t* pWeightsBuffer,
                                      const int32_t* pBiasBuffer,
                                      int8_t* pOutputBuffer,
                                      int32_t nFilterDepth,
                                      int32_t nOutsize,
                                      int32_t nBatches,
                                      uint32_t nQuantizedMultiplier,
                                      int32_t nQuantizedShift,
                                      int32_t nInputOffset,
                                      int32_t nFilterOffset,
                                      int32_t nOutputOffset,
                                      int32_t output_activation_min,
                                      int32_t output_activation_max);

/**
 * @brief Optimised fully-connected layer with pre-reordered weights (int8).
 * @param pInputBuffer         Input activation buffer (int8).
 * @param pWeightsBuffer       Pre-reordered weight buffer (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output buffer (int8).
 * @param nFilterDepth         Number of input features.
 * @param nOutsize             Number of output neurons.
 * @param nBatches             Batch count.
 * @param nQuantizedMultiplier Quantization multiplier.
 * @param nQuantizedShift      Quantization shift.
 * @param nInputOffset         Input zero point offset.
 * @param nFilterOffset        Filter zero point offset.
 * @param nOutputOffset        Output zero point offset.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_fully_connected_int8_reordered_weights(const int8_t* pInputBuffer,
                                                        const int8_t* pWeightsBuffer,
                                                        const int32_t* pBiasBuffer,
                                                        int8_t* pOutputBuffer,
                                                        int32_t nFilterDepth,
                                                        int32_t nOutsize,
                                                        int32_t nBatches,
                                                        uint32_t nQuantizedMultiplier,
                                                        int32_t nQuantizedShift,
                                                        int32_t nInputOffset,
                                                        int32_t nFilterOffset,
                                                        int32_t nOutputOffset,
                                                        int32_t output_activation_min,
                                                        int32_t output_activation_max);

/**
 * @brief Optimised fully-connected layer for int16 input.
 * @param pInputBuffer         Input activation buffer (int16).
 * @param pWeightsBuffer       Weight buffer (int8).
 * @param pBiasBuffer          Bias buffer (int64).
 * @param pOutputBuffer        Output buffer (int16).
 * @param nFilterDepth         Number of input features.
 * @param nOutsize             Number of output neurons.
 * @param nBatches             Batch count.
 * @param nQuantizedMultiplier Quantization multiplier.
 * @param nQuantizedShift      Quantization shift.
 * @param nInputOffset         Input zero point offset.
 * @param nFilterOffset        Filter zero point offset.
 * @param nOutputOffset        Output zero point offset.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_fully_connected_int16(const int16_t* pInputBuffer,
                                       const int8_t* pWeightsBuffer,
                                       const int64_t* pBiasBuffer,
                                       int16_t* pOutputBuffer,
                                       int32_t nFilterDepth,
                                       int32_t nOutsize,
                                       int32_t nBatches,
                                       uint32_t nQuantizedMultiplier,
                                       int32_t nQuantizedShift,
                                       int32_t nInputOffset,
                                       int32_t nFilterOffset,
                                       int32_t nOutputOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max);

/**
 * @brief Optimised tanh activation for int16 input (Q3.12 in, Q0.15 out).
 * @details Used internally by the TFLM LSTM block (int16 activation path). Currently supported
 *          applications do not contain an LSTM block in their TFLite/TFLM model, so this function
 *          is not exercised at runtime. Tested with existing test cases; passes. Test coverage is
 *          limited. Numerical tolerance is documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 * @param nInputMultiplier  Input scaling multiplier.
 * @param nInputLeftShift   Input left-shift amount.
 * @param nLength           Number of elements.
 * @param pInputData        Input buffer (int16, Q3.12).
 * @param pOutputData       Output buffer (int16, Q0.15).
 * @return None
 */
void adi_sharcfx_tanh_int16(int32_t nInputMultiplier, 
                            int32_t nInputLeftShift, 
                            int32_t nLength,
                            const int16_t* pInputData, 
                            int16_t* pOutputData);

/**
 * @brief Optimised tanh activation for int8 input.
 * @param nInputZeroPoint   Input zero point.
 * @param nInputMultiplier  Input scaling multiplier.
 * @param nInputLeftShift   Input left-shift amount.
 * @param nInputSize        Number of elements.
 * @param pInputData        Input buffer (int8).
 * @param pOutputData       Output buffer (int8).
 * @return None
 */
void adi_sharcfx_tanh_int8(int32_t nInputZeroPoint,
                           int32_t nInputMultiplier,
                           int32_t nInputLeftShift,
                           int32_t nInputSize,
                           const int8_t* pInputData,
                           int8_t* pOutputData);

/**
 * @brief Optimised logistic (sigmoid) activation for int8 input.
 * @param nInputZeroPoint   Input zero point.
 * @param nInputMultiplier  Input scaling multiplier.
 * @param nInputLeftShift   Input left-shift amount.
 * @param nInputSize        Number of elements.
 * @param pInputData        Input buffer (int8).
 * @param pOutputData       Output buffer (int8).
 * @return None
 */
void adi_sharcfx_logistic_int8 (int32_t nInputZeroPoint, 
                                int32_t nInputMultiplier, 
                                int32_t nInputLeftShift, 
                                int32_t nInputSize, 
                                const int8_t* pInputData, 
                                int8_t* pOutputData);

/**
 * @brief Optimised logistic (sigmoid) activation for int16 input.
 * @details Used internally by the TFLM LSTM block (int16 activation path). Currently supported
 *          applications do not contain an LSTM block in their TFLite/TFLM model, so this function
 *          is not exercised at runtime. Tested with existing test cases; passes. Test coverage is
 *          limited. Numerical tolerance is documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 * @param nInputMultiplier  Input scaling multiplier.
 * @param nInputLeftShift   Input left-shift amount.
 * @param nInputSize        Number of elements.
 * @param pInputData        Input buffer (int16).
 * @param pOutputData       Output buffer (int16).
 * @return None
 */
void adi_sharcfx_logistic_int16(int32_t nInputMultiplier, 
                                int32_t nInputLeftShift, 
                                int32_t nInputSize, 
                                const int16_t* pInputData, 
                                int16_t* pOutputData);

/**
 * @brief Optimised logistic (sigmoid) activation for int16 input using lookup table.
 * @details Used internally by the TFLM LSTM block (int16 activation path). Currently supported
 *          applications do not contain an LSTM block in their TFLite/TFLM model, so this function
 *          is not exercised at runtime. Tested with existing test cases; passes. Test coverage is
 *          limited. Numerical tolerance is documented in ADI_TFLITE_MICRO_SHARCFX_Library_Product_Reference_Guide.pdf.
 * @param nInputMultiplier  Input scaling multiplier.
 * @param nInputLeftShift   Input left-shift amount.
 * @param nInputSize        Number of elements.
 * @param pInputData        Input buffer (int16).
 * @param pOutputData       Output buffer (int16).
 * @return None
 */
void adi_sharcfx_logistic_int16_optimized_LUT(int32_t nInputMultiplier,
                                              int32_t nInputLeftShift,
                                              int32_t nInputSize,
                                              const int16_t* pInputData,
                                              int16_t* pOutputData);

/**
 * @brief Vectorised element-wise addition for int16 inputs.
 * @param pInput1       First input buffer (int16).
 * @param pInput2       Second input buffer (int16).
 * @param nBatches      Number of batches.
 * @param nInputLen     Number of elements per batch.
 * @param pOutput       Output buffer (int16).
 * @param kInt16Max     Upper saturation clamp.
 * @param kInt16Min     Lower saturation clamp.
 * @return None
 */
void adi_sharcfx_elementwise_add_int16(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int32_t nBatches,
                                       int32_t nInputLen,
                                       int16_t* pOutput,
                                       int32_t kInt16Max,
                                       int32_t kInt16Min);

/**
 * @brief Vectorised element-wise multiplication for int16 inputs (int16 output).
 * @param pInput1               First input buffer (int16).
 * @param pInput2               Second input buffer (int16).
 * @param pOutput               Output buffer (int16).
 * @param nSize                 Number of elements.
 * @param pQuantizedMultiplier  Quantization multiplier.
 * @param pQuantizedShift       Quantization shift.
 * @param pInOffset1            Zero-point offset for input 1.
 * @param pInOffset2            Zero-point offset for input 2.
 * @param pOutOffset            Zero-point offset for output.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_elementwise_mul_int16(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int16_t* pOutput,
                                       int32_t nSize,
                                       uint32_t pQuantizedMultiplier,
                                       int32_t pQuantizedShift,
                                       int32_t pInOffset1,
                                       int32_t pInOffset2,
                                       int32_t pOutOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max);

/**
 * @brief Vectorised element-wise multiplication for int8 inputs (int8 output).
 * @param pInput1               First input buffer (int8).
 * @param pInput2               Second input buffer (int8).
 * @param pOutput               Output buffer (int8).
 * @param nInputLen             Number of elements.
 * @param nQuantizedMultiplier  Quantization multiplier.
 * @param nQuantizedShift       Quantization shift.
 * @param nInOffset1            Zero-point offset for input 1.
 * @param nInOffset2            Zero-point offset for input 2.
 * @param nOutOffset            Zero-point offset for output.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
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
                                       int32_t output_activation_max);

/**
 * @brief Vectorised element-wise multiplication for int16 inputs with int8 output.
 * @param pInput1               First input buffer (int16).
 * @param pInput2               Second input buffer (int16).
 * @param pOutput               Output buffer (int8).
 * @param nInputLen             Number of elements.
 * @param nQuantizedMultiplier  Quantization multiplier.
 * @param nQuantizedShift       Quantization shift.
 * @param nInOffset1            Zero-point offset for input 1.
 * @param nInOffset2            Zero-point offset for input 2.
 * @param nOutOffset            Zero-point offset for output.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_elementwise_mul_int16_input_int8_output(const int16_t* pInput1,
                                       const int16_t* pInput2,
                                       int8_t* pOutput,
                                       int32_t nInputLen,
                                       uint32_t nQuantizedMultiplier,
                                       int32_t nQuantizedShift,
                                       int32_t nInOffset1,
                                       int32_t nInOffset2,
                                       int32_t nOutOffset,
                                       int32_t output_activation_min,
                                       int32_t output_activation_max);


/**
 * @brief Vectorised element-wise addition for int8 inputs (TFLM quantization scheme).
 * @param pInput1               First input buffer (int8).
 * @param pInput2               Second input buffer (int8).
 * @param pOutput               Output buffer (int8).
 * @param nSize                 Number of elements.
 * @param input1_offset         Zero-point offset for input 1.
 * @param input1_multiplier     Quantization multiplier for input 1.
 * @param input1_shift          Quantization shift for input 1.
 * @param input2_offset         Zero-point offset for input 2.
 * @param input2_multiplier     Quantization multiplier for input 2.
 * @param input2_shift          Quantization shift for input 2.
 * @param left_shift            Shared left-shift applied before scaling.
 * @param output_multiplier     Output quantization multiplier.
 * @param output_shift          Output quantization shift.
 * @param output_offset         Output zero-point offset.
 * @param quantized_activation_min Activation minimum clamp.
 * @param quantized_activation_max Activation maximum clamp.
 * @return None
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
									int32_t quantized_activation_max);

/**
 * @brief Vectorised element-wise subtraction for int8 inputs (TFLM quantization scheme).
 * @param pInput1               Minuend input buffer (int8).
 * @param pInput2               Subtrahend input buffer (int8).
 * @param pOutput               Output buffer (int8).
 * @param nSize                 Number of elements.
 * @param input1_offset         Zero-point offset for input 1.
 * @param input1_multiplier     Quantization multiplier for input 1.
 * @param input1_shift          Quantization shift for input 1.
 * @param input2_offset         Zero-point offset for input 2.
 * @param input2_multiplier     Quantization multiplier for input 2.
 * @param input2_shift          Quantization shift for input 2.
 * @param left_shift            Shared left-shift applied before scaling.
 * @param output_multiplier     Output quantization multiplier.
 * @param output_shift          Output quantization shift.
 * @param output_offset         Output zero-point offset.
 * @param quantized_activation_min Activation minimum clamp.
 * @param quantized_activation_max Activation maximum clamp.
 * @return None
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
									int32_t quantized_activation_max);

/**
 * @brief Optimised ReLU activation for int8 input.
 * @param pInput                Input buffer (int8).
 * @param pOutput               Output buffer (int8).
 * @param nSize                 Number of elements.
 * @param nQuantizedMultiplier  Quantization multiplier.
 * @param nQuantizedShift       Quantization shift.
 * @param nInOffset             Input zero-point offset.
 * @param nOutOffset            Output zero-point offset.
 * @param output_activation_min Activation minimum clamp.
 * @param output_activation_max Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_relu_int8(const int8_t* pInput,
                           int8_t* pOutput,
                           const uint32_t nSize,
                           uint32_t pQuantizedMultiplier,
                           int32_t pQuantizedShift,
                           int32_t pInOffset,
                           int32_t pOutOffset,
                           int32_t output_activation_min,
                           int32_t output_activation_max);

/**
 * @brief Optimised dilated 1x1 conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nBatches             Batch count.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nKernelHeight        Kernel height.
 * @param nKernelWidth         Kernel width.
 * @param nNumKernels          Number of kernels.
 * @param nInputWidth          Input width.
 * @param nInputHeight         Input height.
 * @param stride_height        Stride in height direction.
 * @param stride_width         Stride in width direction.
 * @param nPadHeight           Padding in height direction.
 * @param nPadWidth            Padding in width direction.
 * @param nOutHeight           Output height.
 * @param nOutWidth            Output width.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @param nFilterZeroPoint     Filter zero point.
 * @param nActMin              Activation minimum clamp.
 * @param nActMax              Activation maximum clamp.
 * @return None
 */
void adi_sharcfx_conv2d_dilation1x1_int8(const int8_t* pInputBuffer,
                                         const int8_t* pWeightsBuffer,
                                         const int32_t* pBiasBuffer,
                                         int8_t* pOutputBuffer,
                                         int32_t nBatches,
                                         int32_t nInChannels,
                                         int32_t nOutChannels,
                                         int32_t nKernelHeight,
                                         int32_t nKernelWidth,
                                         int32_t nNumKernels,
                                         int32_t nInputWidth,
                                         int32_t nInputHeight,
                                         int32_t stride_height,
                                         int32_t stride_width,
                                         int32_t nPadHeight,
                                         int32_t nPadWidth,
                                         int32_t nOutHeight,
                                         int32_t nOutWidth,
                                         int32_t *pQuantizedMultiplier,
                                         int32_t *pQuantizedShift,
                                         int32_t pInZeroPoint,
                                         int32_t pOutZeroPoint,
                                         int32_t nFilterZeroPoint,
                                         int32_t nActMin,
                                         int32_t nActMax);

/**
 * @brief Optimised 3x3 stride-1 valid-padding conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nWidth               Input width.
 * @param nHeight              Input height.
 * @param nFilters             Number of filters.
 * @param nQuantizedMultiplier Per-channel quantization multipliers.
 * @param nQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_conv2d_kernel3x3_stride1_valid_pad_int8(const int8_t* pInputBuffer,
                                                         const int8_t* pWeightsBuffer,
                                                         const int32_t* pBiasBuffer,
                                                         int8_t* pOutputBuffer,
                                                         int32_t nInChannels,
                                                         int32_t nOutChannels,
                                                         int32_t nWidth,
                                                         int32_t nHeight,
                                                         int32_t nFilters,
                                                         int32_t *nQuantizedMultiplier,
                                                         int32_t *nQuantizedShift,
                                                         int32_t pInZeroPoint,
                                                         int32_t pOutZeroPoint);

/**
 * @brief Optimised 3x3 stride-1 same-padding conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nWidth               Input width.
 * @param nHeight              Input height.
 * @param nFilters             Number of filters.
 * @param nQuantizedMultiplier Per-channel quantization multipliers.
 * @param nQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_conv2d_kernel3x3_stride1_same_pad_int8(const int8_t* pInputBuffer,
                                                        const int8_t* pWeightsBuffer,
                                                        const int32_t* pBiasBuffer,
                                                        int8_t* pOutputBuffer,
                                                        int32_t nInChannels,
                                                        int32_t nOutChannels,
                                                        int32_t nWidth,
                                                        int32_t nHeight,
                                                        int32_t nFilters,
                                                        int32_t *nQuantizedMultiplier,
                                                        int32_t *nQuantizedShift,
                                                        int32_t pInZeroPoint,
                                                        int32_t pOutZeroPoint);

/**
 * @brief Optimised 3x3 stride-2 valid-padding conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nWidth               Input width.
 * @param nHeight              Input height.
 * @param nFilters             Number of filters.
 * @param nQuantizedMultiplier Per-channel quantization multipliers.
 * @param nQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_conv2d_kernel3x3_stride2_valid_pad_int8(const int8_t* pInputBuffer,
                                                        const int8_t* pWeightsBuffer,
                                                        const int32_t* pBiasBuffer,
                                                        int8_t* pOutputBuffer,
                                                        int32_t nInChannels,
                                                        int32_t nOutChannels,
                                                        int32_t nWidth,
                                                        int32_t nHeight,
                                                        int32_t nFilters,
                                                        int32_t *nQuantizedMultiplier,
                                                        int32_t *nQuantizedShift,
                                                        int32_t pInZeroPoint,
                                                        int32_t pOutZeroPoint);

/**
 * @brief Optimised 1x1 conv2d (int8).
 * @param pInputBuffer         Input feature map (int8).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param pOutputBuffer        Output feature map (int8).
 * @param nBatches             Batch count.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param nSize                Spatial size (height * width).
 * @param nQuantizedMultiplier Per-channel quantization multipliers.
 * @param nQuantizedShift      Per-channel quantization shifts.
 * @param nInputOffset         Input zero point.
 * @param nOutputOffset        Output zero point.
 * @return None
 */
void adi_sharcfx_conv2d_kernel1x1_int8(const int8_t* pInputBuffer,
                                       const int8_t* pWeightsBuffer,
                                       const int32_t* pBiasBuffer,
                                       int8_t* pOutputBuffer,
                                       int32_t nBatches,
                                       int32_t nInChannels,
                                       int32_t nOutChannels,
                                       int32_t nSize,
                                       int32_t *nQuantizedMultiplier,
                                       int32_t *nQuantizedShift,
                                       int32_t nInputOffset,
                                       int32_t nOutputOffset);

/**
 * @brief Optimised 1x1 conv2d with int16 input, non-interleaved layout.
 * @param pInputBuffer         Input feature map (int16).
 * @param pOutputBuffer        Output feature map (int16).
 * @param pWeightsBuffer       Kernel weights (int8).
 * @param pBiasBuffer          Bias buffer (int32).
 * @param nWidth               Spatial width.
 * @param nInChannels          Number of input channels.
 * @param nOutChannels         Number of output channels.
 * @param pQuantizedMultiplier Per-channel quantization multipliers.
 * @param pQuantizedShift      Per-channel quantization shifts.
 * @param pInZeroPoint         Input zero point.
 * @param pOutZeroPoint        Output zero point.
 * @return None
 */
void adi_sharcfx_conv2d_kernel1x1_noninterleaved_int16(const int16_t *pInputBuffer,
                                                       int16_t *pOutputBuffer,
                                                       const int8_t *pWeightsBuffer,
                                                       const int32_t *pBiasBuffer,
                                                       int32_t nWidth,
                                                       int32_t nInChannels,
                                                       int32_t nOutChannels,
                                                       uint32_t *pQuantizedMultiplier,
                                                       int32_t *pQuantizedShift,
                                                       int32_t pInZeroPoint,
                                                       int32_t pOutZeroPoint);


#endif /* ADI_SHARCFX_NN_H_ */
