// RUN: tpp-opt %s -map-linalg-to-tpp -tile-consumer-and-fuse-producers -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, 
                  %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>, 
                  %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>, 
                  %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>, 
                  %arg8: tensor<1024xf32>, %output: tensor<128x1024xf32>,
                  %output1: tensor<128x2048xf32>, %output2: tensor<128x1024xf32>, 
                  %ouput3: tensor<128x512xf32>) -> tensor<128x1024xf32> {
    %c0 = arith.constant 0.0 : f32
    // %0 = linalg.init_tensor [128, 512] : tensor<128x512xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    // %4 = linalg.init_tensor [128, 1024] : tensor<128x1024xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 512]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = arith.maxf %arg9, %c0 : f32 
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    // %8 = linalg.init_tensor [128, 2048] : tensor<128x2048xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      linalg.yield %arg9 : f32
    } -> tensor<128x2048xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %arg5 : tensor<128x1024xf32>, tensor<1024x2048xf32>) outs(%9 : tensor<128x2048xf32>) attrs =  {iterator_ranges = [128, 2048, 1024]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x2048xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<128x2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> tensor<128x2048xf32>
    // %12 = linalg.init_tensor [128, 1024] : tensor<128x1024xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xf32>) outs(%output : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11, %arg7 : tensor<128x2048xf32>, tensor<2048x1024xf32>) outs(%13 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 2048]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<128x1024xf32>) outs(%output : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = arith.maxf %arg9, %c0 : f32 
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    return %15 : tensor<128x1024xf32>
  }
}

