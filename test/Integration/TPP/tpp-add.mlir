// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXE

// RUN: tpp-opt %s -convert-linalg-to-loops | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXE

func.func private @generate_1D_source(%init_source : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<?xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val_f32 = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val_f32 :  f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func.func @entry() {
  %arg0 = tensor.empty() : tensor<56x32xf32>

  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c56 = arith.constant 56 : index
  %c0_cst = arith.constant 0.0 : f32
  %c1_cst = arith.constant 1.0 : f32

  %c32 = arith.constant 32 : index
  %seed = tensor.empty(%c32) : tensor<?xf32>
  %seedGen = call @generate_1D_source(%seed) : (tensor<?xf32>) -> (tensor<?xf32>)
  %cast_seed = tensor.cast %seedGen : tensor<?xf32> to tensor<32xf32>
  %b0 = linalg.broadcast ins(%cast_seed : tensor<32xf32>)
                         outs(%arg0 : tensor<56x32xf32>)
                         dimensions = [0]
  %out_shape = tensor.empty() : tensor<2x56x56x32xf32>
  %3 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%ia1 = %out_shape) -> (tensor<2x56x56x32xf32>) {
    %4 = scf.for %arg4 = %c0 to %c56 step %c1 iter_args(%ia2 = %ia1) -> (tensor<2x56x56x32xf32>) {
      %out_slice = tensor.empty() : tensor<56x32xf32>
      %fill = linalg.fill ins(%c0_cst : f32) outs(%out_slice : tensor<56x32xf32>) -> tensor<56x32xf32>
      %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%b0, %b0 : tensor<56x32xf32>, tensor<56x32xf32>)
      outs(%fill : tensor<56x32xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %0 = arith.addf %in, %in_0 : f32
          linalg.yield %0 : f32
      } -> tensor<56x32xf32>
      %inserted_slice = tensor.insert_slice %res into %ia2 [%arg3, %arg4, 0, 0] [1, 1, 56, 32] [1, 1, 1, 1]
        : tensor<56x32xf32> into tensor<2x56x56x32xf32>
      scf.yield %inserted_slice : tensor<2x56x56x32xf32>
    }
    scf.yield %4 : tensor<2x56x56x32xf32>
  }
  %zeroCst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  scf.for %i1 = %c0 to %c2 step %c1 {
    scf.for %i2 = %c0 to %c56 step %c1 {
      scf.for %i3 = %c0 to %c56 step %c1 {
        %v0 = vector.transfer_read %3[%i1, %i2, %i3, %zeroCst], %d1 : tensor<2x56x56x32xf32>, vector<32xf32>
        vector.print %v0 : vector<32xf32>
      }
    }
  }
  // EXE-COUNT-6272: ( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62 )

  return
}
