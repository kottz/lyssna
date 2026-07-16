cmake_minimum_required(VERSION 3.16)

project(lyssna VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LYSSNA_VULKAN "Use the GGML Vulkan backend" ON)

set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${LYSSNA_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)

add_executable(lyssna src/lyssna.cpp)
target_link_libraries(lyssna PRIVATE
    parakeet
    PkgConfig::PIPEWIRE
    Threads::Threads
)

include(GNUInstallDirs)
set_target_properties(lyssna PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS lyssna RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS
    parakeet
    ggml
    ggml-base
    ggml-cpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if (LYSSNA_VULKAN)
    install(TARGETS ggml-vulkan
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

configure_file(
    packaging/lyssna.service.in
    packaging/lyssna.service
    @ONLY
)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/packaging/lyssna.service
    DESTINATION ${CMAKE_INSTALL_DATADIR}/systemd/user
)
install(
    FILES prompts/dictation-cleanup.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/lyssna/prompts
)
