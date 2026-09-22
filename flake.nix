{
  description = "NInfer Volta (sm_70 / Tesla V100) engine and server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    cutlass-src = {
      url = "github:nvidia/cutlass/v4.4.2";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, cutlass-src }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
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
      ninfer = pkgs.stdenv.mkDerivation {
        pname = "ninfer-v100";
        version = "0.1.0";
        src = ./.;

        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          pkg-config
          git
          makeWrapper
        ];

        buildInputs = with pkgs; [
          cudaPkg
          ffmpeg
          curl
        ];

        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=70"
          "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=${cutlass-src}"
          "-DNINFER_BUILD_APPS=ON"
          "-DBUILD_TESTING=OFF"
          "-DNINFER_BUILD_BENCHMARKS=OFF"
        ];

        CUDA_PATH = "${cudaPkg}";
        CUDAToolkit_ROOT = "${cudaPkg}";
        CMAKE_CUDA_COMPILER = "${cudaPkg}/bin/nvcc";

        installPhase = ''
          runHook preInstall
          mkdir -p $out/bin
          cp -a apps/ninfer apps/ninfer-serve apps/ninfer-perplexity $out/bin/
          for bin in $out/bin/*; do
            wrapProgram "$bin" \
              --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:${pkgs.cudaPackages.cuda_cudart}/lib"
          done
          runHook postInstall
        '';
      };
    in {
      packages.${system} = {
        default = ninfer;
        ninfer = ninfer;
      };

      devShells.${system}.default = pkgs.mkShell {
        name = "ninfer-v100-build-env";
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config git ];
        buildInputs = with pkgs; [ cudaPkg ffmpeg curl ];
        CUDA_PATH = "${cudaPkg}";
        CUDAToolkit_ROOT = "${cudaPkg}";
        CMAKE_CUDA_COMPILER = "${cudaPkg}/bin/nvcc";
        shellHook = ''
          export PKG_CONFIG_PATH="${pkgs.ffmpeg.dev}/lib/pkgconfig:${pkgs.curl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
          export CPATH="${cudaPkg}/include:$CPATH"
        '';
      };
    };
}
