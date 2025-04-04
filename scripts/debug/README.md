# Debug Scripts

These scripts are used to debug the compiler output.

## Check Accuracy

### Purpose

To run an MLIR program through `tpp-run` and compare the output with the baseline (`linalg-to-loops`).

### Usage

Options:
* `-b bin_dir`: The binary directory (usually `build/bin`)
* `-o "opt1 opt2 ..."`: `tpp-run` options
* `-d 0.005`: FP Delta allowance (default `0.01`)
* `-e entry`: Name of entry function (default `entry`)
* `-i file.mlir`: Input MLIR file

Examples:
```
// Runs tpp-run on file.mlir and compares linalg-to-loops with the default pipeline
./scripts/debug/check_accuracy.sh -i file.mlir

// Compares linalg-to-loops with vector-to-kernels to 0.005 precision in outputs
./scripts/debug/check_accuracy.sh -i file.mlir -o "--vector-to-kernels" -d 0.005
```

## Debug All Passes

### Purpose

To run MLIR through the compiler with `--mlir-print-ir-after-all`,
split the output into multiple files (`NNN.mlir`) and run a `diff` program
between any of those files that change the IR.

### Usage

Options:
* `-b bin_dir`: The binary directory (usually `build/bin`)
* `-i file.mlir`: Input MLIR file (optional)
* `-d tool`: Specifies a diff tool (default `diff`)
* `-m "opt1 opt2 ..."`: `mlir-gen` options
* `-o "opt1 opt2 ..."`: `tpp-opt` options

Examples:
```
// Run compiler over an MLIR file, uses `vimdiff`
./scripts/debug/debug_all_passes.sh \
  -b ./build/bin \
  -i file.mlir \
  -d vimdiff

// Generates an MLP with `mlir-gen`, uses `meld`
./scripts/debug/debug_all_passes.sh \
  -b ./build/bin \
  -m "--kernel=const --bias --relu --float-type=bf16 --batch=256 --layers=1024,1024,1024,1024" \
  -d meld

// Default behaviour, runs `mlir-gen` & `tpp-opt` without args, uses `diff`
./scripts/debug/debug_all_passes.sh \
  -b ./build/bin
```

### Helpers

`split.py`: Splits the output of `--mlir-print-ir-after-all` into multiple files.

`diff.py`: Looks through a list of `NNN.mlir` files and shows the diff of each
pair of files when the IR changes (ex. `003.mlir -> 007.mlir`, `007.mlir -> 013.mlir`
etc.).
