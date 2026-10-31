# MinecraftAI

MinecraftAI is a client-side Fabric mod for Minecraft 1.21.11. It exposes player and target observations and accepts optional controls from a local Python process over TCP. It does not include or train an AI policy.

## Requirements

- Java 21 or newer
- The Gradle wrapper included in this repository

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
- `gradle/`, `build.gradle`, `gradle.properties`, `settings.gradle` — Gradle wrapper and build configuration.
- `.github/workflows/build.yml` — GitHub Actions build workflow.

## In-game controls

- **K** toggles external AI control. It is off by default. When off, remote commands are rejected and normal player controls are left alone.
- **F8** toggles the diagnostics overlay independently of AI control.

The overlay includes target projection and crosshair information, player/target state, attack timing, and input diagnostics. While AI control is on, it also displays the latest remote input state.

## Local control protocol

The mod opens a TCP server on `127.0.0.1:8765`; it listens only on loopback, not on the network. To use another port, add `-Dminecraftai.controlPort=8766` to the development client's JVM arguments and connect to port `8766`.

Messages are newline-delimited JSON. The protocol is version 2 and observations use the `minecraftai.pvp.v1` schema. While AI control is enabled, the server streams observations each client tick. A controller sends a command with an increasing `seq`, the source observation's `tick`, `yaw_delta`, and `pitch_delta`; optional booleans control forward, backward, left, right, jump, sprint, sneak, attack, and use. Commands tied to invalid or stale observations are rejected. Remote input expires after 500 ms without an update.

Observations include player movement and combat state, a visible non-allied player target when available, projected target bounds and aim point, crosshair-hit state, hit timing, and a 20-row action history. The `target_aim_dx` and `target_aim_dy` values are normalized screen coordinates, with `(0, 0)` at the top-left and `(0.5, 0.5)` at the center. Target selection and projection are client-side estimates; they are not based on image recognition.

## Control demo

With the development client running, join a world and press **K**, then run:

```powershell
py -3 .\scripts\control_test.py
```

The demo sends view rotation commands for a full horizontal turn. It does not move the player or send attack/use actions. The script accepts `--port` if the server is configured to use a non-default port.

## License

The project is licensed under CC0. See [LICENSE](LICENSE).
