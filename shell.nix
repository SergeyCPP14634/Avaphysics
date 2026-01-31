{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    sdl3
    assimp
    vulkan-loader
    vulkan-validation-layers
    vulkan-headers
    vulkan-tools
    shaderc
    glslang
    spirv-tools
    zlib
    libxcb
    minizip
    gcc
    cmake
    ninja
    git
    clang
    llvmPackages.libclang
    rustup
    openssl
    pkg-config
  ];

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
  
  NIX_CFLAGS_COMPILE = "-isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.lib.getVersion pkgs.clang}/include";
  
  VULKAN_SDK = "${pkgs.vulkan-headers}";
  SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";
  SHADERC_INCLUDE_DIR = "${pkgs.shaderc.dev}/include";

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.sdl3}/lib:${pkgs.shaderc.lib}/lib"
    export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
    
    echo "Настраиваем Rust через rustup..."
    
    if ! rustup toolchain list 2>/dev/null | grep -q 'stable.*(default)'; then
      echo "Устанавливаем стабильный toolchain Rust..."
      rustup default stable
    fi
    
    echo "Проверяем наличие rust-src..."
    if ! rustup component list 2>/dev/null | grep -q 'rust-src.*installed'; then
      rustup component add rust-src
      echo "rust-src установлен"
    else
      echo "rust-src уже установлен"
    fi
    
    export RUST_SRC_PATH="$(rustc --print sysroot)/lib/rustlib/src/rust/library"
    
    if [ -n "$LIBCLANG_PATH" ]; then
      echo "LIBCLANG_PATH установлен: $LIBCLANG_PATH"
      
      if ls "$LIBCLANG_PATH"/libclang*.so 1>/dev/null 2>&1; then
        echo "✓ libclang найден"
      else
        echo "⚠ Внимание: libclang не найден по пути $LIBCLANG_PATH"
        echo "Попробуем найти в других местах..."
        FIND_PATH=$(find ${pkgs.llvmPackages.libclang} -name "libclang*.so" 2>/dev/null | head -1)
        if [ -n "$FIND_PATH" ]; then
          export LIBCLANG_PATH="$(dirname "$FIND_PATH")"
          echo "Найден альтернативный путь: $LIBCLANG_PATH"
        fi
      fi
    fi
    
    echo ""
    echo "========================================"
    echo "Окружение готово к работе!"
    echo "Версии инструментов:"
    echo "   Rust: $(rustc --version | cut -d' ' -f2-)"
    echo "   Cargo: $(cargo --version | cut -d' ' -f2-)"
    echo "   GCC: $(gcc --version | head -n1)"
    echo "   Clang: $(clang --version | head -n1)"
    echo "   CMake: $(cmake --version | head -n1)"
    echo "   Shaderc: $(echo ${pkgs.shaderc.version})"
    echo "========================================"
    echo ""
    echo "Для VS Code добавьте в settings.json:"
    echo '  "rust-analyzer.rustc.source": "discover"'
    echo ""
  '';
}

