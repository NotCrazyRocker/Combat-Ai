Repository and environment setup
Foundation
Establish the GitHub repository, the Fabric development environment, the Python project, and the communication protocol.

Tasks
- Set up the mod project using the official Fabric development toolchain.

  Fabric Documentation
  +1
- Create separate configurations for Theobald's practice map and the self-play arena.
- Establish a version compatibility matrix before implementing mod-specific hooks.
- Set up logging, configuration files and reproducible builds.
Completion criterion: The mod compiles, Minecraft launches, and the Python process starts independently.

Version warning: HeroBot currently lists Minecraft 1.21.11, while Theobald's published v1.3 practice map is listed for Minecraft 1.21 and requires Carpet PvP. We will verify their exact version compatibility rather than assume that one build supports both environments.

Minecraft Mod
+1






01

Build the observation system
Mod development
First, the mod only reads the game state. No AI control yet.

It will collect:
- Player state: position, velocity, yaw, pitch, health, absorption, armor and equipment.
- Combat state: attack cooldown, recent damage, hurt time, held weapon and relevant status effects.
- Opponent state: relative position, velocity, rotation, health, armor, equipment and recent combat state.
- Movement state: sprinting, sneaking, grounded state, fall distance, water/lava state and movement restrictions.
- World state: nearby collision geometry, obstacles, ground conditions and relevant environmental hazards.
- Timing: server tick, observation sequence, match ID and the time of relevant events.
We will also calculate useful derived features, including opponent distance, relative velocity, angle to target and whether the target is reachable.

We will distinguish information obtained directly from Minecraft from values estimated by our own calculations. This prevents us from inadvertently treating an estimate as authoritative game state.

Completion criterion: We can record a complete structured observation every game tick and verify its values against known in-game states.



02

Build the action controller
The mod will accept an action description and execute it in Minecraft.

Our initial action space will include:
Action category	Controls
Movement	Forward, backward, strafe and stop
Mobility	Jump, sprint and sneak
Aiming	Yaw and pitch adjustments
Combat	Attack and use item
Equipment	Select hotbar slot, with inventory actions added later

The action protocol will represent persistent button states separately from one-time actions. This matters because holding forward for five ticks is different from pressing and releasing forward every tick.

We will implement separate execution adapters for the real client player and HeroBot. Both will accept the same abstract action format.

Completion criterion: A scripted controller can move, aim, attack, jump and equip items reliably without a neural network.



03

Connect the Fabric mod to Python
We will build a local communication bridge, initially using JSON over a WebSocket connection.

Mod responsibilities: publish observations, receive actions, validate them and queue execution on the proper game thread.

Python responsibilities: receive observations, identify the agent and tick, calculate actions, and send them back.

Every message will include a protocol version, agent ID, tick or sequence number, and match ID. The system will reject stale actions and use a safe fallback instead of freezing Minecraft if Python misses an update.

The server tick must never block while waiting for neural-network training.

Completion criterion: Python can control the real player through a scripted policy, and the mod can record the complete observation-action sequence.



04

Train against Theobald's practice bot
First learning milestone
This preserves the approach that has already shown promising results for you.

We will begin with sword PvP and use the structured observations to train the policy against Theobald's bot on its existing practice map.

Learning method: Proximal Policy Optimization (PPO), using a small multilayer perceptron as the initial policy.

Initial reward signals: match outcome, damage dealt, damage received, hit events, and carefully weighted shaping terms for basic combat behavior.

We will record episodes as observation, action, reward and next-observation sequences, then update the model from collected rollouts.

Training will progress from basic aiming and tracking to movement, attack timing and complete duels. These are curriculum stages, not separate permanently hard-coded combat systems.

Completion criterion: The AI can complete repeatable duels, training resumes from saved checkpoints, and we can measure whether successive versions perform better against the same opponents.

This environment will use its compatible Carpet PvP setup. It will not load HeroBot alongside Carpet PvP.



05

Integrate HeroBot and create the self-play arena
Second learning environment
We will build a dedicated, controlled arena in a HeroBot-compatible environment without Carpet or Carpet PvP.

The integration will:
- Spawn and configure the HeroBot fake player.
- Control its movement, aim, attacks and equipment through a dedicated action adapter.
- Collect structured observations for both fighters.
- Reset duels, restore kits, position both fighters and detect match outcomes.
- Ensure that each fighter gets its own policy state and action history.
We will first test the fake player using a scripted opponent. Then we will make both fighters use our neural network.

Completion criterion: Two agents controlled by Python can complete repeatable duels inside one Minecraft world without requiring a second client.

This phase will also establish whether the available HeroBot control hooks are sufficient for low-latency tick-level actions. If they are not, we will implement the required server-side integration rather than depend on a command being dispatched for every individual action.



06

Implement self-play training
This is where the AI begins generating its own competitive training experience.

Initially, the two agents can share the same learned policy weights while maintaining separate histories. We will then train against archived versions of the model.

The self-play system will include:
- Current-policy matches: expose the learner to opponents near its current level.
- Archived opponents: preserve older strategies so that the AI does not learn to beat only the latest version.
- Specialized exploiters: introduce opponents that target predictable aiming, poor movement and ineffective attack timing.
- Side and spawn randomization: reduce dependence on a particular starting position or orientation.
- Independent evaluation: test checkpoints against opponents that were not used for the corresponding training updates.
We will initially use PPO with a configurable opponent sampler. More sophisticated league-training methods can be introduced once the basic system produces reliable results.

Completion criterion: The training process can select opponents, run self-play episodes, update the policy, retain checkpoints and compare their tournament performance automatically.



07

Combine training environments and improve robustness
At this point, we will train across both environments rather than abandon Theobald's bot once self-play works.

The training scheduler will mix matches against the practice bot, current self-play policies, archived checkpoints and specialized opponents.

We will also introduce temporal context through a recurrent policy or transformer only when the simpler model shows a meaningful limitation. This avoids spending computing resources on a larger model before we have established a strong baseline.

Other upgrades include:
- Better reward calibration and fewer reward-hacking opportunities.
- Detection and mitigation of catastrophic forgetting.
- Optional imitation learning from recorded structured state/action trajectories.
- Better opponent modelling and prediction of relative movement.
- Longer training runs and automated recovery after interrupted sessions.
Completion criterion: The agent improves across both environments and retains previously learned skills as new opponents and training objectives are introduced.



08

Benchmark, optimize and expand
Once sword PvP is stable, we will measure performance and optimize the training pipeline.

Key metrics will include:
- Win rate and tournament rating against a fixed opponent pool.
- Damage differential and time to first hit.
- Hit rate, attack timing and target-tracking error.
- Recovery following knockback and the ability to escape bad positioning.
- Inference latency, match throughput and training stability.
Only after these results are reliable will we expand into more complicated PvP modes, such as crystal PvP, which introduces inventory use, explosive placement and substantially different action requirements.

Completion criterion: We have repeatable benchmarks, retained model checkpoints, reproducible training configurations and a clear record of which changes genuinely improve performance.