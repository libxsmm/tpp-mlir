// We don't need to print because we use the check dialect
// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>


func.func private @generate_1D_source(%buff: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%buff : tensor<?xf32>) {
  ^bb0(%out: f32):
    %0 = linalg.index 0 : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.sitofp %1 : i32 to f32
    linalg.yield %2 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func.func @entry() {
  %cst = arith.constant 32: index
  %c0 = arith.constant 0: index
  %cf = arith.constant 0.0: f32
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c12 = arith.constant 12: index
  %c56 = arith.constant 56: index
  %buf = tensor.empty(%cst) {alignment = 128 : i64} : tensor<?xf32>
  %gen = call @generate_1D_source(%buf): (tensor<?xf32>) -> (tensor<?xf32>)
  %arg0 = tensor.cast %gen: tensor<?xf32> to tensor<32xf32>

  %out_shape = tensor.empty() {alignment = 128 : i64} : tensor<12x2x56x56x32xf32>
  %0 = linalg.generic {indexing_maps=[#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<32xf32> ) outs(%out_shape : tensor<12x2x56x56x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } ->  tensor<12x2x56x56x32xf32>

  // Check if subview extraction gives us the same result when comparing with the second relu op.
  %relu1 = scf.for %arg3 = %c0 to %c12 step %c1 iter_args(%ia1 = %0) -> tensor<12x2x56x56x32xf32> {
    %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%ia2 = %ia1) -> tensor<12x2x56x56x32xf32> {
      %3 = scf.for %arg5 = %c0 to %c56 step %c1 iter_args(%ia3 = %ia2) -> tensor<12x2x56x56x32xf32> {
        %extracted_slice = tensor.extract_slice %ia3[%arg3, %arg4, %arg5, 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
          : tensor<12x2x56x56x32xf32> to tensor<56x32xf32>
        %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<56x32xf32>) {
          ^bb0(%out: f32):
            %5 = arith.maximumf %out, %cf : f32
            linalg.yield %5 : f32
        } -> tensor<56x32xf32>
        %inserted_slice = tensor.insert_slice %4 into %ia3 [%arg3, %arg4, %arg5, 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
        : tensor<56x32xf32> into tensor<12x2x56x56x32xf32>
        scf.yield %inserted_slice : tensor<12x2x56x56x32xf32>
      }
      scf.yield %3 : tensor<12x2x56x56x32xf32>
    }
    scf.yield %2 : tensor<12x2x56x56x32xf32>
  }

  %relu2 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<12x2x56x56x32xf32>) {
      ^bb0(%out: f32):
        %6 = arith.maximumf %out, %cf : f32
        linalg.yield %6 : f32
  } -> tensor<12x2x56x56x32xf32>

  %threshold = arith.constant 0.0:f32
  check.expect_almost_eq(%relu1, %relu2, %threshold) : tensor<12x2x56x56x32xf32>, tensor<12x2x56x56x32xf32>, f32

  return
}
