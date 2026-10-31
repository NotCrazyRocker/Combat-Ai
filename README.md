# MinecraftAI

## Setup

This is the MinecraftAI Fabric mod project. It targets Minecraft 1.21.11 and Java 21.

## Development

- Import this folder as a Gradle project in IntelliJ IDEA or another Gradle-compatible IDE.
- Use the `runClient` Gradle task to launch a development client.
- Use `build` to produce the mod JAR in `build/libs/`.
- Keep common code under `src/main` and client-only code under `src/client`.

## Python control

The client listens over TCP on `127.0.0.1:8765`. The listener is loopback-only. Enable AI control in game with the default **K** key before sending actions. When AI control is off, the server rejects commands and the mod leaves the player's normal controls alone. While it is on, fresh Python snapshots control movement, jump, sneak, sprint, attack, use, and view direction; input expires after 500 ms without an update. Restart the development client after changing the control protocol.

For a separate Mineflayer player connecting to a Minecraft server on `localhost:25565`, see the [Mineflayer frozen-policy runner](./mineflayer/README.md). It defaults to `combo_attack_best.pt` and the `minecraftai.pvp.v1` schema, but server-side estimates are not identical to client-mod telemetry.

Run the included control demo from the project folder while `runClient` is open:

```powershell
py -3 .\scripts\control_test.py
```

The demo leaves movement and action buttons released and sends a 360-degree horizontal turn in 12-degree steps. It does not move or spawn a target. To use another port, set `-Dminecraftai.controlPort=8766` in the `runClient` JVM configuration and run the script with `--port 8766`.

## Client diagnostics and tests

Press **F8** to toggle the full diagnostics panel; it is separate from **K**, which toggles AI control. The panel and visual markers are hidden together when F8 is off, while the small AI on/off indicator remains visible. A green box shows the target's projected 3D bounding box; a cyan cross shows its projected center; a magenta cross shows the projected head point. The panel reports target distance, whether the crosshair ray hit the selected target, camera yaw/pitch, player and target movement state, attack charge, button states, remote click counters, and the last confirmed outgoing hit plus incoming damage timing. Projection is calculated from the entity box and camera geometry, not detected from rendered pixels.

While **K** AI control is on, the top-left arcade input matrix shows W/A/S/D movement, jump, sprint, sneak, attack, and use keycaps. Lit keycaps indicate the current fresh AI command; attack/use and nonzero yaw/pitch briefly pulse so one-tick actions remain visible. The yaw and pitch line shows the most recently applied view deltas.

Use the panel for these in-game checks:

- Keep a target in view and move it to the left and right of the screen. The box, cyan center, and magenta head point should move in the same direction as the target.
- Move the target partly out of view and look sharply up/down or stand very close. The box should remain bounded to the viewport instead of stretching across unrelated screen areas.
- Move the crosshair onto and away from the target. `crosshair hit` should switch between YES and no based on the entity ray hit.
- Walk, jump, sprint, and sneak; the player-state line and key-state line should change with the action. Observe target movement to check relative velocity signs.
- In a safe test world, land ordinary, sprinting, falling critical, and sweep attacks. `Last outgoing hit` should change only after a confirmed hit and show the corresponding detected type. Incoming damage timing should update when another player hits you.

Run the offline observation-contract tests with:

```powershell
py -3.12 .\scripts\test_client_suite.py
```

For live checks, start `runClient`, join a world, and press **K**. Validate live observations without sending actions with:

```powershell
py -3.12 .\scripts\test_client_suite.py --live
```

To probe yaw, pitch, movement, jump, sneak, and sprint input routing, use `--live --actions`. These probes move and jump the player. Add `--allow-interactions` to also send one attack and one use click; run those probes only in a disposable test world because they can damage entities or activate the held item. The live suite waits for the mod to report the specific command sequence as applied before comparing its result. It restores the camera and sends neutral controls on completion.

Protocol version 2 streams one newline-delimited `minecraftai.pvp.v1` observation per client tick while AI control is on. A control message includes its increasing `seq`, the source observation's `tick`, `yaw_delta`, and `pitch_delta`; optional movement and action booleans default to false. Controls with an already-used, future, or more than two ticks old observation are rejected. The view deltas are applied once when the client consumes that sequence. Yaw stays continuous across the ±180° boundary to avoid a render interpolation jump, while pitch is clamped. Observations report the last applied command sequence so clients can match an action to a post-action frame. A control state expires after 500 ms without a new message, releasing remote inputs. The TCP connection stays open while control is off and sends a `control_state` message when **K** toggles, so a trainer can pause and resume without reconnecting.

Each observation has a `features` object and a 20-row `action_history`. The feature names are explicit rather than packed into a hidden positional vector. They include player position/height, velocity and movement state, camera yaw/pitch, health/food/attack charge, whether the main hand holds a sword, target distance and normalized screen offset (`target_dx`, `target_dy`), projected normalized bounding box, line-of-sight-filtered target motion/state, and whether the crosshair ray currently hits that target. `target_aim_dx` and `target_aim_dy` are absolute normalized screen coordinates: `(0, 0)` is the top-left and `(0.5, 0.5)` is screen center. The trainer converts the head point to reticle-relative error by subtracting `0.5` from its vertical coordinate, so aligned means zero error. If there is no visible, non-allied player target, `target_valid` is false. The target box is a client-side projection of the 3D player box; it is an estimate of the rendered screen bounds, not a screenshot detector.

Hit timing advances in game ticks and stays at `-1` until the corresponding event has occurred. The `last_hit_*` flags describe the most recently confirmed outgoing hit. `last_hit_type` gives a primary label (`sweep`, `critical`, `knockback`, or `normal`); flags can overlap, since a critical hit can also receive knockback. Critical and sweep eligibility use vanilla's attack checks; knockback includes sprint knockback and a main-hand Knockback enchantment. This identifies the hit mechanics that applied, not measured knockback distance. `hit_combo` counts confirmed outgoing hits since the last incoming player hit; `knockback_combo` counts only uninterrupted knockback hits and resets after a non-knockback hit or incoming player hit. `action_history` is oldest to newest, padded to 20 rows, with each row ordered as `[forward, backward, left, right, jump, sprint, sneak, attack, use, yaw_delta, pitch_delta]`; movement/action values are `0` or `1`.

The project uses Fabric Loader, Fabric API, Fabric Loom, and Mojang mappings. Version settings are in `gradle.properties`.

## PPO aim training

### Joint yaw and pitch PPO

The combined trainer is `python/train_aim_ppo.py`. It loads the pretrained yaw policy from `yaw_ppo_latest.pt` and the pretrained pitch policy from `pitch_ppo_latest.pt`, then runs both controllers together. Yaw and pitch keep separate observation encoders, feature networks, value functions, and PPO optimizers so each starts with its trained behavior and receives only its own axis reward. It issues yaw deltas up to ±10° and pitch deltas up to ±8° per tick; movement, jump, sprint, sneak, and use remain disabled. Scripted sword attacks require a valid target, a sword in hand, full attack cooldown, and the crosshair on target. Pitch is rewarded for vertical overlap with the target bounding box and for camera pitch near 0° on every applied tick; it receives no pitch reward for a crosshair hit.

Start Minecraft with the mod, join a world with Theobald's training bot visible, press **K**, then run:

```powershell
py -3.12 python/train_aim_ppo.py
```

Each PPO update contains exactly 1,024 applied control steps. The script logs separate yaw and pitch rewards and losses, horizontal and head-point centering, projection coverage, crosshair-hit rate, scripted attack count, action magnitudes, and exploration standard deviations. It resumes `python/checkpoints/aim_ppo_latest.pt` when present; otherwise, it initializes each axis from its corresponding pretrained checkpoint and starts fresh optimizer state. It saves the latest checkpoint every five updates by default, on **Ctrl+C**, and when AI control is turned off with **K**. It also saves `python/checkpoints/aim_ppo_best.pt` whenever a completed rollout reaches a new best mean combined yaw-plus-pitch reward. This checkpoint contains both policies and optimizer states, and can resume training with `--checkpoint .\python\checkpoints\aim_ppo_best.pt`; its best score is carried across runs. Pressing **K** again resumes with a fresh rollout. Use `--no-resume` to reload both pretrained policies or `--checkpoint-every 1` to save the latest checkpoint every update.

`python/train_yaw_ppo.py` is retained as the earlier shared-network joint trainer for reference. Its default output is now `joint_yaw_pitch_legacy_latest.pt`, so it cannot overwrite the new combined trainer's checkpoint.

### W and sprint combo PPO

`python/train_combo_ppo.py` continues the combined aim policy while learning W, sprint, and attack decisions. When starting fresh (or using `--no-resume`), it initializes yaw and pitch from `aim_ppo_best.pt` when available, otherwise from `aim_ppo_latest.pt`; `--init-from` overrides this choice. It saves its own resumable `combo_latest.pt` and `combo_best.pt`. Yaw receives the same centering, progress, and action-smoothing reward as aim training, scaled to 0.5. Pitch receives the same target-box progress/overlap and near-zero camera-pitch reward as pitch training, also scaled to 0.5, in addition to its combat reward. Pitch gets no bonus for `crosshair_on_target`. A confirmed hit earns +0.5 only when the corresponding scripted attack satisfied the ground-truth gate (valid target, sword, full cooldown, crosshair on target). A gated attack that misses costs -0.25; a premature attack costs -0.25; other false attacks cost -0.10; and skipping an eligible attack costs -0.02. Confirmed ground-truth knockback hits add +1.0. Consecutive eligible knockback hits earn +0.2 per hit after the first, capped at +1.0 extra per hit. An incoming hit resets the rewarded knockback streak and costs -0.5 per hit, increasing for each hit in a streak up to 10 hits. Forward and sprint toggle penalties are currently disabled. The arcade keystroke panel shows the general combo and the observed knockback-only combo.

Start Minecraft with the mod and Theobald's training bot, press **K**, then run:

```powershell
py -3.12 .\python\train_combo_ppo.py
```

The trainer sends yaw, pitch, W, sprint, and scripted attack actions on each applied control step; it leaves backward, left, right, jump, sneak, and use disabled. It logs action rates, missed and false clicks, toggle counts, confirmed knockback hits, incoming hits, peak combo, aim metrics, and separate PPO losses. It saves every five updates by default, on **Ctrl+C**, and when AI control is turned off with **K**. Use `--no-resume` to start fresh from the best aim checkpoint (falling back to latest), `--init-from` to select another aim checkpoint, or `--checkpoint-every 1` to save after every update.

To run a frozen policy without PPO updates or checkpoint writes, start Minecraft with the mod, enter a suitable world with the intended target visible, press **K**, then run:

```powershell
py -3.12 .\python\run_combo_policy.py
```

The inference script loads `python/checkpoints/combo_best.pt` by default. It uses deterministic actions (Gaussian means for yaw/pitch and the most likely forward/sprint actions); scripted attacks still require a valid target, sword, full cooldown, and crosshair hit. The policy can move and attack in-game. Use `--checkpoint PATH` to select a different frozen checkpoint.
### Standalone pitch-only PPO

`python/train_pitch_ppo.py` temporarily trains only vertical centering. It sends pitch deltas with yaw fixed at zero, and never presses attack, movement, jump, sprint, sneak, or use. The pitch actor starts with a zero mean and small exploration; it copies only the compatible shared feature layers from `yaw_ppo_latest.pt`. It does not run the yaw policy or modify the yaw checkpoint, so begin with the camera already facing the target. Its objective is to drive the projected head point's signed vertical screen error toward zero, with up/down corrections and a smooth centering reward.

Start Minecraft with the mod, join a world with the target visible, press **K**, then run:

```powershell
py -3.12 python/train_pitch_ppo.py
```

This uses a separate `python/checkpoints/pitch_ppo_latest.pt` checkpoint, resumes it automatically, and otherwise initializes from the yaw checkpoint. It logs head-point error, valid projection coverage, signed pitch actions, corrective direction, reward, entropy, and PPO losses every 1,024 applied steps. It saves every five updates by default, on **Ctrl+C**, and when AI control is turned off with **K**. Use `--no-resume` to restart pitch training from the yaw feature layers or `--checkpoint-every 1` to save each update.

The mod selects the nearest visible, non-allied player as the policy target, so Theobald's training bot must appear as a player in the same world with line of sight. The yaw-only and pitch-only checkpoints remain unchanged as initialization sources; joint and pitch-only training save to their own checkpoints. If inference sees a fresher frame before sending an action, each trainer recomputes rather than sending an action from an expired observation.
## License

The project is licensed under CC0. See [LICENSE](LICENSE).
