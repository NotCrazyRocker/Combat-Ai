# Combat-Ai — observation feature patch (Phase 1)

Designed for the repository's current Fabric template: Minecraft 1.21.11, official Mojang mappings, Java 21, and split `main`/`client` source sets.

## What this adds

- Captures one structured observation at `ClientTickEvents.END_CLIENT_TICK` (20 observations per second at Minecraft's normal tick rate).
- Reads the integrated server tick counter from Fabric's server end-tick event in single-player and LAN-hosted worlds. On a remote multiplayer client, the authoritative server tick is not exposed and remains `null`.
- Uses Minecraft entity/block state directly; there is **no OCR, screenshot capture, or pixel processing**.
- Selects the nearest alive, non-spectator player within 32 blocks as the provisional opponent. If none is available, `opponent` and target-derived features are `null`.
- Reports player/opponent position, velocity, rotation, health, absorption, armor, equipment, effects, movement state and hurt time.
- Computes target distance, relative position/velocity, yaw/pitch error, line of sight and a heuristic 3-block reachability estimate.
- Samples nearby collision shapes and water/lava/hazard candidates around the local player.
- Sends one newline-delimited JSON object per client tick to a local Python TCP receiver, without waiting on Python on the Minecraft tick thread.
- Uses a bounded queue and retries connection attempts if Python is not running yet.
- Writes validated observations to timestamped `.jsonl` files.

## Apply to the repo

Copy the patch's `src/client/java/com/grim/swordpvpai/client/SwordPvPAIClient.java` over the existing file, then copy the new `src/client/java/com/grim/swordpvpai/client/observation/` directory into the repository's matching `src/client/java` path. Copy the patch's `python/` and `protocol/` directories to the repository root.

No Gradle dependency change is required; the implementation uses Fabric API and Gson already available to the Minecraft runtime.

## Run it

From the repository root, in terminal 1:

```powershell
py -3 python\bridge\receiver.py
```

In terminal 2:

```powershell
.\gradlew.bat runClient
```

Join a world/server with another player-like entity nearby. The Python terminal shows a periodic status line with the observation rate, client tick, and (in integrated-server worlds) server tick; JSONL recordings are written under `python/recordings/`. The receiver binds to `127.0.0.1` only by default.

The mod defaults to `127.0.0.1:8765`. You can override this with JVM properties `-Dswordpvpai.observation.host=127.0.0.1` and `-Dswordpvpai.observation.port=8765`.

## Important measurement limits

1. The client sees the world state available to its Minecraft client. In integrated-server worlds, `server_tick` comes from `MinecraftServer.getTickCount()` at the end of each server tick. A remote multiplayer client cannot inspect the remote server's counter, so `server_tick` remains `null` there; `level_game_time` is the client's world-time value.
2. `damage_observed_this_tick` is inferred from health plus absorption changes between observations. It can miss multiple damage changes that resolve within a single tick and does not prove the local player caused the damage.
3. Critical/sweep/knockback labels are explicitly low-confidence candidates inferred from observable state. Exact attack subtype attribution requires server-side instrumentation on the server that adjudicates the hit; this patch does not pretend those labels are authoritative.
4. Opponent selection is currently nearest-player selection, not team/faction-aware targeting. Nearby collision geometry is centered on the local player and sampled in a 7×4×7 block window; it is not a full world map.
