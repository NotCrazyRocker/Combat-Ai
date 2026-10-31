package com.grim.minecraftai.client;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;

/** Read-only probes used by the Python client verification suite. */
public final class RemoteInputDiagnostics {
	private static volatile long attackClicks;
	private static volatile long useClicks;
	private static volatile long lastAppliedSequence = -1;
	private static volatile long lastAppliedObservationTick = -1;

	private RemoteInputDiagnostics() { }

	public static void recordClick(boolean attack) {
		if (attack) attackClicks++;
		else useClicks++;
	}

	public static void resetAppliedAction() {
		lastAppliedSequence = -1;
		lastAppliedObservationTick = -1;
	}

	public static void markApplied(long sequence, long observationTick) {
		lastAppliedSequence = sequence;
		lastAppliedObservationTick = observationTick;
	}

	static JsonObject snapshot(Minecraft client) {
		JsonObject state = new JsonObject();
		state.addProperty("control_enabled", MinecraftAIClient.isAIControlEnabled());
		if (client.player == null) {
			for (String key : new String[] {"forward", "backward", "left", "right", "jump", "sneak", "sprint", "attack", "use"}) {
				state.addProperty(key, false);
			}
		} else {
			state.addProperty("forward", client.options.keyUp.isDown());
			state.addProperty("backward", client.options.keyDown.isDown());
			state.addProperty("left", client.options.keyLeft.isDown());
			state.addProperty("right", client.options.keyRight.isDown());
			state.addProperty("jump", client.options.keyJump.isDown());
			state.addProperty("sneak", client.options.keyShift.isDown());
			state.addProperty("sprint", client.options.keySprint.isDown());
			state.addProperty("attack", client.options.keyAttack.isDown());
			state.addProperty("use", client.options.keyUse.isDown());
		}
		JsonObject clicks = new JsonObject();
		clicks.addProperty("attack", attackClicks);
		clicks.addProperty("use", useClicks);
		JsonObject result = new JsonObject();
		result.add("action_state", state);
		result.add("consumed_clicks", clicks);
		result.addProperty("last_applied_sequence", lastAppliedSequence);
		result.addProperty("last_applied_observation_tick", lastAppliedObservationTick);
		return result;
	}
}
