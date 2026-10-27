# WaterSimCUDA

A Windows C++17/CUDA shallow-water basin rendered with raylib and OpenGL 3.3. CUDA owns the water state and writes the shared mesh buffers. Falling drops trigger volume-balanced ripples; the simulation uses fixed stable substeps and a bounded catch-up clock.

Build with a supported MSVC/CUDA pair, CMake, an NVIDIA GPU, and a desktop session. See [the checked build environment](docs/build-environment.md) for exact versions, environment setup, and the pinned raylib offline override.

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build\water_demo.exe
build\water_demo.exe --nx 320 --ny 192
```

Keep the generated `assets` directory beside the executable. Launching from another working directory is supported. The OpenGL adapter must support CUDA; startup reports an error if it cannot use that adapter.

| Control | Action |
| --- | --- |
| Left click on water | Drop at the selected point |
| Right drag / wheel | Orbit / zoom, including while paused |
| Space | Pause or resume simulation and drops |
| R | Clear water and active drops; restart seeded rain |
| T | Toggle rain |
| Up / Down | Rain rate ±1 drop/s, clamped to 0–60 |
| Left / Right | Impact strength ±0.001 m, clamped to 0–0.02 × depth |
| [ / ] | Damping ±0.05/s, clamped to 0–5/s |
| Escape | Close |

`water_demo --help` lists startup options with units and defaults. The default grid is 512 × 512 in a 20 m × 12 m basin, depth 1 m, damping 0.15/s, impact radius 0.35 m, strength 0.01 m, rain 8/s, seed 1. Grid sizes are 8–2048 per axis. Radius must span at least two cell widths and stay below one quarter of the shorter basin dimension. Startup damping accepts any finite nonnegative value; bracket controls use the range above.

The overlay reports render FPS and the last completed CUDA simulation duration per frame. Timing queries do not wait for unfinished samples. A stalled frame advances at most eight 1/120 s quanta; simulated time slows under overload. There is no CPU water fallback.

The desktop smoke test opens a bounded real window, renders from a different working directory, and exports a capture. For longer validation, set `WATER_VALIDATE_SECONDS` to the sample duration and `WATER_VALIDATE_WARMUP` to warm-up seconds (default 10). The application closes automatically and prints frame mean/p95 and FPS. Optional `WATER_VALIDATE_CAPTURE` gives an image output path; this explicitly downloads the framebuffer only after sampling. For example, from PowerShell:

```powershell
$env:WATER_VALIDATE_SECONDS = '60'
$env:WATER_VALIDATE_WARMUP = '10'
$env:WATER_VALIDATE_CAPTURE = "$PWD/build/water.png"
./build/water_demo.exe
Remove-Item Env:WATER_VALIDATE_SECONDS, Env:WATER_VALIDATE_WARMUP, Env:WATER_VALIDATE_CAPTURE
```

These runs retain the normal 60 FPS cap. See [validation results and remaining manual checks](docs/validation.md).

`ctest --test-dir build -C Release -L soak --output-on-failure` runs 120 simulated seconds of maximum seeded rain on each rectangular test grid. It uses the production controller and CUDA solver without opening a window, checking finite state, mean height, amplitude and the final 30-second envelope. The soak requires `WATER_BUILD_DEMO=ON` for raylib's scheduler types; `WATER_BUILD_DEMO=OFF` retains the raylib-free configuration, numerical and impact groups.
