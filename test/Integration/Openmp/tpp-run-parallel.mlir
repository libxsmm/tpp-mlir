// RUN: tpp-run %s -def-parallel=0 -e entry -entry-point-result=void -print-mlir=late -print 2>&1 | \
// RUN: FileCheck %s --check-prefix=SERIAL

// RUN: tpp-run %s -def-parallel=1 -e entry -entry-point-result=void -print-mlir=late -print 2>&1 | \
// RUN: FileCheck %s --check-prefix=PARALLEL

func.func @entry(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %0 = scf.forall (%arg1, %arg2) = (0, 0) to (8, 8) step(1, 1)
                                          shared_outs(%o = %empty) -> (tensor<8x8xf32>) {
    %slice = tensor.extract_slice %arg0[%arg1, %arg2] [1, 1] [1, 1]
      : tensor<8x8xf32> to tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %o[%arg1, %arg2] [1, 1] [1, 1]
        : tensor<1x1xf32> into tensor<8x8xf32>
    }
  }
  return %0 : tensor<8x8xf32>
}

// SERIAL-LABEL: func.func @_entry
// SERIAL: scf.parallel

// PARALLEL-LABEL: func.func @_entry
// PARALLEL: omp.parallel
// PARALLEL:   omp.wsloop

// SERIAL-COUNT-8: ( 1, 1, 1, 1, 1, 1, 1, 1 )
// PARALLEL-COUNT-8: ( 1, 1, 1, 1, 1, 1, 1, 1 )
