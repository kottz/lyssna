# Lyssna

Lyssna is local, push-to-talk dictation for Linux and Sway. It records from
PipeWire, transcribes with NVIDIA Parakeet through whisper.cpp's GGML runtime,
and types Unicode text into the focused Wayland application using `wtype`.

The model and microphone are not active while Lyssna is cold. On the first
completed recording, the model is loaded and retained for a configurable idle
period before its CPU/Vulkan resources are released.

## Dependencies

- CMake 3.16 or newer
- A C++17 compiler
- PipeWire development files
- Vulkan development tools and driver, unless built with `-DLYSSNA_VULKAN=OFF`
- `wtype`

On Arch Linux, the application dependencies are available with:

```sh
sudo pacman -S --needed base-devel cmake pipewire vulkan-headers shaderc wtype
```

## Clone and build

```sh
git clone --recursive <lyssna-repository-url>
cd lyssna
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lyssna -j
```

If the repository was cloned without submodules:

```sh
git submodule update --init --recursive
```

Run the daemon directly during development:

```sh
./build/lyssna --daemon \
    --model /absolute/path/to/ggml-parakeet-tdt-0.6b-v3-q4_k.bin \
    --threads 8 \
    --max-speech-ms 30000 \
    --model-idle 300
```

In another terminal:

```sh
./build/lyssna start
./build/lyssna stop
./build/lyssna status
./build/lyssna cancel
./build/lyssna unload
```

Both daemon and clients use `$XDG_RUNTIME_DIR/lyssna.sock` by default.

## Install

Install the binary and user-service definition under `~/.local`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --target lyssna -j
cmake --install build
```

The install includes Lyssna's pinned Parakeet and GGML runtime libraries under
`~/.local/lib`. The executable uses an installation-relative runtime search
path, so no `LD_LIBRARY_PATH` configuration is needed.

Put the model where the supplied service expects it:

```sh
mkdir -p ~/.local/share/lyssna/models
cp /path/to/ggml-parakeet-tdt-0.6b-v3-q4_k.bin \
   ~/.local/share/lyssna/models/
```

Reload and enable the service:

```sh
systemctl --user daemon-reload
systemctl --user enable --now lyssna.service
```

Because `wtype` needs the graphical-session environment, add this to the Sway
configuration before the dictation bindings:

```text
exec sh -c 'systemctl --user import-environment WAYLAND_DISPLAY SWAYSOCK XDG_CURRENT_DESKTOP XDG_RUNTIME_DIR && systemctl --user restart lyssna.service'
```

Example push-to-talk bindings:

```text
bindsym --no-repeat $mod+apostrophe exec $HOME/.local/bin/lyssna start
bindsym --release $mod+apostrophe exec $HOME/.local/bin/lyssna stop
```

Reload Sway with `swaymsg reload`.

Follow service logs with:

```sh
journalctl --user -u lyssna.service -f
```

## Dependency policy

whisper.cpp is pinned as the `third_party/whisper.cpp` Git submodule. Lyssna
links only to its public Parakeet and GGML APIs; it does not depend on any
whisper.cpp example code. Update the dependency deliberately with:

```sh
git submodule update --remote third_party/whisper.cpp
```

Review and commit the new submodule revision after building and testing it.

## Text cleanup

The prompt in `prompts/dictation-cleanup.md` is intended for a future optional
local-LLM cleanup stage. Lyssna currently types Parakeet's result directly.
