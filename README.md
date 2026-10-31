# MinecraftAI

MinecraftAI is a client-side Fabric mod for Minecraft 1.21.11. It streams player and target observations and accepts controls from a separate Python process over local TCP. The repository also contains Python PPO training scripts, saved checkpoints, and a frozen-policy runner; policies run separately and are not activated automatically by the mod.

## Requirements

- Java 21 or newer
- The Gradle wrapper included in this repository
- Python 3.12 and the packages in `python/requirements.txt` for the Python tools

The Minecraft, Fabric Loader, Fabric API, and Loom versions are configured in `gradle.properties`.

## Build and run

From PowerShell in the project folder:

```powershell
.\gradlew.bat build
.\gradlew.bat runClient
```

The built mod JAR is written to `build/libs/`. Import the project as a Gradle project in an IDE for development.

## Project layout

- `src/main/` — mod entry point, metadata, and shared resources.
- `src/client/` — client initialization, observations, diagnostics, TCP control server, and Minecraft input/attack mixins.
- `scripts/` — Python control demo and protocol test scripts.
- `python/` — PPO training and inference scripts, shared policy helpers, requirements, and checkpoints.
- `gradle/`, `build.gradle`, `gradle.properties`, `settings.gradle` — Gradle wrapper and build configuration.
- `.github/workflows/build.yml` — GitHub Actions build workflow.

## In-game controls

- **K** toggles external AI control. It is off by default. When off, remote commands are rejected and normal player controls are left alone.
- **F12** toggles the diagnostics overlay independently of AI control.

The overlay includes detected opponent count, each opponent's visibility and distance, the selected target, target projection and crosshair information, player state, attack timing, and input diagnostics. Other living, non-spectator players are considered opponents regardless of scoreboard team. While AI control is on, the overlay also displays the latest remote input state.

## Local control protocol

The mod opens a TCP server on `127.0.0.1:8765`; it listens only on loopback, not on the network. To use another port, add `-Dminecraftai.controlPort=8766` to the development client's JVM arguments and connect to port `8766`.

Messages are newline-delimited JSON. The protocol is version 2 and observations use the `minecraftai.pvp.v2` schema. While AI control is enabled, the server streams observations each client tick. A controller sends a command with an increasing `seq`, the source observation's `tick`, `yaw_delta`, and `pitch_delta`; optional booleans control forward, backward, left, right, jump, sprint, sneak, attack, and use. Commands tied to invalid or stale observations are rejected. Remote input expires after 500 ms without an update.

Observations include player movement and combat state, a visible player target when available, projected target bounds, screen-center target offsets, crosshair-hit state, hit timing and damage events, and a 20-row action history. Target bounds use normalized screen coordinates; `target_dx` and `target_dy` are offsets from screen center, so `(0, 0)` means centered. Target selection and projection are client-side estimates, not image recognition.

### Python policy compatibility

The Python transport accepts v2 observations and adapts them to the v1 feature contract used by the existing policies and checkpoints. To install the Python dependencies, run:

```powershell
py -3.12 -m pip install -r .\python\requirements.txt
```

With the development client running, join a world and press **K** to enable control. Run the frozen combo policy from another PowerShell window:

```powershell
py -3.12 .\python\run_combo_policy.py --port 8765
```

The runner uses `python/checkpoints/combo_attack_best.pt` by default. It controls aiming, movement, and attacks; use it in a suitable world with the intended target visible. Other `train_*.py` scripts provide online PPO training.

## Control demo

With the development client running, join a world and press **K**, then run:

```powershell
py -3 .\scripts\control_test.py
```

The demo sends view rotation commands for a full horizontal turn. It does not move the player or send attack/use actions. The script accepts `--port` if the server is configured to use a non-default port.

## License

The project is licensed under CC0. See [LICENSE](LICENSE).
