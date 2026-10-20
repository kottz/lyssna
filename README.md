# Lyssna

Local push-to-talk dictation for Linux and Wayland. Lyssna records through
PipeWire, transcribes with NVIDIA Parakeet using Whisper.cpp, and types the result with `wtype`.

## Setup

Install the dependencies on Arch Linux:

```sh
sudo pacman -S --needed base-devel cmake pipewire vulkan-headers shaderc wtype
```

```sh
git clone --recursive https://github.com/kottz/lyssna.git
cd lyssna
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --target lyssna -j
cmake --install build
```

Install the model:

```sh
install -Dm644 /path/to/ggml-parakeet-tdt-0.6b-v3-q4_k.bin \
  ~/.local/share/lyssna/models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin
```

Start Lyssna:

```sh
systemctl --user daemon-reload
systemctl --user enable --now lyssna.service
```

## Sway

Add to `~/.config/sway/config`:

```text
exec sh -c 'systemctl --user import-environment WAYLAND_DISPLAY SWAYSOCK XDG_CURRENT_DESKTOP XDG_RUNTIME_DIR && systemctl --user restart lyssna.service'
bindsym --no-repeat $mod+apostrophe exec $HOME/.local/bin/lyssna start
bindsym --release $mod+apostrophe exec $HOME/.local/bin/lyssna stop
```
