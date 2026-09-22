# Development and build shell for NInfer (Tesla V100 / sm_70)
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  cudaPkg = pkgs.symlinkJoin {
    name = "cuda-minimal-12.9";
    paths = with pkgs.cudaPackages; [
      cuda_nvcc
      cuda_cudart
      cuda_cccl
      cuda_nvtx.dev
      cuda_nvtx.include
      cuda_nvtx.lib
    ];
  };
in
pkgs.mkShell {
  name = "ninfer-v100-build-env";

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
    git
  ];

  buildInputs = with pkgs; [
    cudaPkg
    ffmpeg
    curl
  ];

  CUDA_PATH = "${cudaPkg}";
  CUDAToolkit_ROOT = "${cudaPkg}";
  CMAKE_CUDA_COMPILER = "${cudaPkg}/bin/nvcc";

  shellHook = ''
    export PKG_CONFIG_PATH="${pkgs.ffmpeg.dev}/lib/pkgconfig:${pkgs.curl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
    export CPATH="${cudaPkg}/include:$CPATH"
    echo "=========================================="
    echo " NInfer V100 (sm_70) Build Environment"
    echo " CUDA: $(${cudaPkg}/bin/nvcc --version | grep release)"
    echo "=========================================="
  '';
}
