// RUN: tpp-run %s -e entry --entry-point-result=void --splat-to-random --init-type quant --print --seed 123 | FileCheck %s

// CHECK: ( 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 0, 0 )


#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

!twoDimf32 = tensor<3x5xf32>
!twoDimi8 = tensor<3x5xi8>
!oneDimf32 = tensor<3xf32>
!oneDimi8 = tensor<3xi8>

module {

  // ========================== dequantize =================================
  func.func @dequantize(%quantizedi8 : !twoDimi8, %scale: !oneDimf32) -> !twoDimf32 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0_f32 = arith.constant 0.0 : f32
    %castToF32 = tensor.empty() : !twoDimf32

    %castToF32_val = linalg.generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel", "parallel"]
    } ins(%quantizedi8 : !twoDimi8) outs(%castToF32 : !twoDimf32) {
      ^bb0(%in_val: i8, %out_val: f32):
        %1 = arith.sitofp %in_val: i8 to f32
        linalg.yield %1 : f32
    } -> !twoDimf32

    %bcastScale = tensor.empty() : !twoDimf32
    %bcastScale_val = linalg.broadcast ins(%scale:!oneDimf32) outs(%bcastScale:!twoDimf32) dimensions = [1]

    %scaleRes_fp32 = tensor.empty() : !twoDimf32
    %r = linalg.mul ins(%castToF32_val, %bcastScale_val : !twoDimf32, !twoDimf32) outs(%scaleRes_fp32 : !twoDimf32) -> !twoDimf32
    
    return %r : !twoDimf32
  }

  // ========================== computeScalingFactors =================================
  func.func @computeScalingFactors(%dquant_val: !twoDimf32) -> (!oneDimf32, !oneDimf32) {
    %c0_f32 = arith.constant 0.0 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %cst0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32

    %c_neg_inf = arith.constant 0.0 : f32
    %init = tensor.empty() : !oneDimf32
    %init_val = linalg.fill ins(%c_neg_inf : f32) outs(%init : !oneDimf32) -> !oneDimf32
 
    // Reduce along dimension 1 (rows) to find max of each row
    %absmax = linalg.reduce ins(%dquant_val : !twoDimf32) outs(%init_val : !oneDimf32) dimensions = [1]
      (%in: f32, %out: f32) {
      %abs = llvm.intr.fabs(%in) : (f32) -> f32
      %max = arith.maximumf %abs, %out : f32
      linalg.yield %max : f32
    }

    // Compute exponent of the max value
    %res_exp = tensor.empty() : tensor<3xi32>
    %neg_exp2 = tensor.empty() : !oneDimf32
	  %res_exp_val = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%absmax : !oneDimf32) outs(%res_exp : tensor<3xi32>) {
	    ^bb0(%in: f32, %out: i32):
      %fr_exp = llvm.intr.frexp(%in) : (f32) -> !llvm.struct<(f32, i32)>
      %a = llvm.extractvalue %fr_exp[1] : !llvm.struct<(f32, i32)>
      %b = llvm.extractvalue %fr_exp[0] : !llvm.struct<(f32, i32)>
      %unbiased_exp = arith.subi %a, %c7 : i32

      linalg.yield %unbiased_exp : i32
	  } -> tensor<3xi32>

    // Compute 2 raise to the power of -ve value of the exponent
    %neg_exp2_val = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%res_exp_val : tensor<3xi32>) outs(%neg_exp2 : !oneDimf32) {
      ^bb0(%in: i32, %out: f32):
        %neg = arith.subi %cst0, %in : i32
        %castToFloat = arith.sitofp %neg : i32 to f32
        %n_exp2 = math.exp2 %castToFloat : f32
        linalg.yield %n_exp2 : f32
    } -> !oneDimf32

    %pos_exp2 = tensor.empty() : !oneDimf32
    %pos_exp2_val = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%res_exp_val : tensor<3xi32>) outs(%pos_exp2 : !oneDimf32) {
      ^bb0(%in: i32, %out: f32):
        %castToFloat = arith.sitofp %in : i32 to f32
        %p_exp2 = math.exp2 %castToFloat : f32
        linalg.yield %p_exp2 : f32
    } -> !oneDimf32

    return %neg_exp2_val, %pos_exp2_val : !oneDimf32, !oneDimf32
  }

  // ========================== Quantize =================================
  func.func @quantize(%dquant_val: !twoDimf32) -> (!twoDimi8, !oneDimf32) {
    %c0_f32 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    %neg_exp2_val, %pos_exp2_val = call @computeScalingFactors(%dquant_val) : (!twoDimf32) -> (!oneDimf32, !oneDimf32)

    // Broadcast and Scale Cf32
    %scaleRes = tensor.empty() : !twoDimf32
    %bcastScaleRes = linalg.broadcast ins(%neg_exp2_val:!oneDimf32) outs(%scaleRes:!twoDimf32) dimensions = [1]

    %scaledC = tensor.empty() : !twoDimf32
    %scaleRes_val = linalg.mul ins(%dquant_val, %bcastScaleRes : !twoDimf32, !twoDimf32) outs(%scaledC : !twoDimf32) -> !twoDimf32
    
    // Down-convert Cf32 -> Ci8
    %castToi8 = tensor.empty() : !twoDimi8
    %castToi8_val = linalg.generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel", "parallel"]
    } ins(%scaleRes_val : !twoDimf32) outs(%castToi8 : !twoDimi8) {
      ^bb0(%in_val: f32, %out_val: i8):
        %clamped_i8 = arith.fptosi %in_val : f32 to i8
        linalg.yield %clamped_i8 : i8
    } -> !twoDimi8

    return %castToi8_val, %pos_exp2_val : !twoDimi8, !oneDimf32
  }

  func.func @entry(%input : !twoDimi8, %scale : !oneDimf32, %output : !twoDimi8) -> (!twoDimi8) {
    %c0_f32 = arith.constant 0.0 : f32
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    // ========================== DeQuantize =================================
    %dequantized_val = call @dequantize(%input, %scale) : (!twoDimi8, !oneDimf32) -> !twoDimf32

    // ========================== Quantize =================================
    %castToi8, %pos_exp2 = call @quantize(%dequantized_val) : (!twoDimf32) -> (!twoDimi8, !oneDimf32)

    // Perform elementwise Subraction of original input from quantized output to see the difference
    %diffTensor_val = linalg.generic {
      indexing_maps = [#map1, #map1, #map1],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %castToi8 : !twoDimi8, !twoDimi8) outs(%output : !twoDimi8) {
      ^bb0(%in1: i8, %in2: i8, %out: i8):
        %sub = arith.subi %in1, %in2 : i8
        linalg.yield %sub : i8
    } -> !twoDimi8

    return %diffTensor_val : !twoDimi8
  }
}