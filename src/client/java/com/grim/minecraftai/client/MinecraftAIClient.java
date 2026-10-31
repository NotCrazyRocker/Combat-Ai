package com.grim.minecraftai.client;

import com.grim.minecraftai.MinecraftAI;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.hud.VanillaHudElements;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.resources.Identifier;
import net.minecraft.util.Mth;
import org.lwjgl.glfw.GLFW;

public class MinecraftAIClient implements ClientModInitializer {
	private static final String TOGGLE_KEY_TRANSLATION = "key.minecraft-ai.toggle_ai_control";
	private static final String DEBUG_OVERLAY_KEY_TRANSLATION = "key.minecraft-ai.toggle_debug_overlay";
	private static final boolean DEFAULT_AI_CONTROL_ENABLED = false;
	private static volatile boolean aiControlEnabled = DEFAULT_AI_CONTROL_ENABLED;
	private static volatile boolean debugOverlayEnabled;
	private static final PythonControlServer CONTROL_SERVER = new PythonControlServer(
			Integer.getInteger("minecraftai.controlPort", 8765)
	);

	@Override
	public void onInitializeClient() {
		CONTROL_SERVER.start();

		KeyMapping.Category category = KeyMapping.Category.register(id("main"));
		KeyMapping toggleAIControlKey = KeyBindingHelper.registerKeyBinding(
				new KeyMapping(TOGGLE_KEY_TRANSLATION, InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_K, category)
		);
		KeyMapping toggleDebugOverlayKey = KeyBindingHelper.registerKeyBinding(
				new KeyMapping(DEBUG_OVERLAY_KEY_TRANSLATION, InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F8, category)
		);

		ClientTickEvents.END_CLIENT_TICK.register(client -> {
			ObservationBuilder.beginTick();
			while (toggleAIControlKey.consumeClick()) {
				aiControlEnabled = !aiControlEnabled;
				if (!aiControlEnabled) {
					CONTROL_SERVER.clear();
				}
				CONTROL_SERVER.publishControlState(aiControlEnabled);
				MinecraftAI.LOGGER.info("AI control {}", aiControlEnabled ? "enabled" : "disabled");
			}
			while (toggleDebugOverlayKey.consumeClick()) {
				debugOverlayEnabled = !debugOverlayEnabled;
				MinecraftAI.LOGGER.info("AI diagnostics overlay {}", debugOverlayEnabled ? "enabled" : "disabled");
			}

			ControlCommand command = client.player == null ? null : CONTROL_SERVER.pollAimCommand();
			if (command != null) {
				// Keep yaw continuous for render interpolation of the hand and body.
				// Wrapping +180° directly to -180° makes the pose interpolate across
				// almost a full revolution for one frame.
				float yaw = client.player.getYRot() + command.yawDelta();
				float pitch = Mth.clamp(client.player.getXRot() + command.pitchDelta(), -90.0F, 90.0F);
				client.player.setYRot(yaw);
				client.player.setYHeadRot(yaw);
				// Keep the first-person arm model aligned with the continuous camera yaw.
				client.player.setYBodyRot(yaw);
				client.player.setXRot(pitch);
				RemoteInputDiagnostics.markApplied(command.sequence(), command.observationTick());
			}
			if (client.player != null && client.level != null && (aiControlEnabled || debugOverlayEnabled)) {
				JsonObject observation = ObservationBuilder.build(client, CONTROL_SERVER);
				if (aiControlEnabled) {
					CONTROL_SERVER.publishObservation(observation, ObservationBuilder.currentTick());
				}
			}
		});

		HudElementRegistry.attachElementBefore(
				VanillaHudElements.CHAT,
				id("ai_control_status"),
				MinecraftAIClient::renderAIControlStatus
		);
	}

	public static boolean isAIControlEnabled() {
		return aiControlEnabled;
	}

	public static ControlCommand getCurrentControlCommand() {
		return aiControlEnabled ? CONTROL_SERVER.getCurrentCommand() : null;
	}

	public static void recordAttackAttempt(net.minecraft.world.entity.player.Player attacker,
			net.minecraft.world.entity.player.Player target, boolean critical, boolean knockback,
			boolean sweep) {
		ObservationBuilder.recordAttackAttempt(attacker, target, critical, knockback, sweep);
	}

	public static void recordDamageEvent(int damagedEntityId, int sourceCauseId) {
		ObservationBuilder.recordDamageEvent(damagedEntityId, sourceCauseId);
	}

	public static String getLastHitHudText() {
		return ObservationBuilder.lastHitHudText();
	}

	public static int getCurrentCombo() {
		return ObservationBuilder.currentCombo();
	}

	public static int getCurrentKnockbackCombo() {
		return ObservationBuilder.currentKnockbackCombo();
	}

	private static void renderAIControlStatus(GuiGraphics graphics, DeltaTracker tickCounter) {
		Minecraft client = Minecraft.getInstance();
		String status = "AI Control: " + (aiControlEnabled ? "ON" : "OFF");
		int x = 6;
		int y = 6;

		int width = client.font.width(status) + 8;
		graphics.fill(x, y, x + width, y + 18, 0xA0000000);
		graphics.drawString(client.font, status, x + 4, y + 5, aiControlEnabled ? 0xFF55FF55 : 0xFFFF5555);
		if (aiControlEnabled) {
			AIActionOverlay.render(graphics, client.font, getCurrentControlCommand());
		}
		if (debugOverlayEnabled) {
			ObservationBuilder.renderDebugOverlay(graphics, client.getWindow().getGuiScaledWidth(),
					client.getWindow().getGuiScaledHeight(), aiControlEnabled ? 150 : 0);
		}
	}

	private static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MinecraftAI.MOD_ID, path);
	}
}
