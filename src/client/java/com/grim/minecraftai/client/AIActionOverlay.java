package com.grim.minecraftai.client;

import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphics;

/** Arcade-style view of the latest AI command applied to Minecraft controls. */
final class AIActionOverlay {
	private static final int PANEL_X = 6;
	private static final int PANEL_Y = 30;
	private static final int PANEL_WIDTH = 222;
	private static final int PANEL_HEIGHT = 140;
	private static final long PULSE_NANOS = 350_000_000L;
	private static final int CYAN = 0xFF00E5FF;
	private static final int MAGENTA = 0xFFFF3DE1;
	private static final int PANEL = 0xE908101D;
	private static final int BORDER = 0xFF263750;

	private static long lastSequence = Long.MIN_VALUE;
	private static long attackPulseAt;
	private static long usePulseAt;
	private static long lookPulseAt;
	private static float lastYawDelta;
	private static float lastPitchDelta;

	private AIActionOverlay() { }

	static void render(GuiGraphics graphics, Font font, ControlCommand command) {
		long now = System.nanoTime();
		if (command != null && command.sequence() != lastSequence) {
			lastSequence = command.sequence();
			if (command.attack()) attackPulseAt = now;
			if (command.use()) usePulseAt = now;
			if (command.yawDelta() != 0.0F || command.pitchDelta() != 0.0F) {
				lastYawDelta = command.yawDelta();
				lastPitchDelta = command.pitchDelta();
				lookPulseAt = now;
			}
		}

		int x = PANEL_X;
		int y = PANEL_Y;
		graphics.fill(x, y, x + PANEL_WIDTH, y + PANEL_HEIGHT, PANEL);
		graphics.fill(x, y, x + PANEL_WIDTH, y + 2, CYAN);
		graphics.fill(x, y + PANEL_HEIGHT - 1, x + PANEL_WIDTH, y + PANEL_HEIGHT, BORDER);
		graphics.fill(x, y, x + 1, y + PANEL_HEIGHT, BORDER);
		graphics.fill(x + PANEL_WIDTH - 1, y, x + PANEL_WIDTH, y + PANEL_HEIGHT, BORDER);

		graphics.drawString(font, "AI // INPUT MATRIX", x + 8, y + 6, CYAN, false);
		String state = command == null ? "WAIT" : "LIVE";
		int stateColor = command == null ? 0xFF75839B : MAGENTA;
		graphics.drawString(font, state, x + PANEL_WIDTH - font.width(state) - 8, y + 6, stateColor, false);
		graphics.fill(x + 8, y + 19, x + PANEL_WIDTH - 8, y + 20, 0xFF263750);

		// WASD movement cluster.
		int keyW = 26;
		int keyH = 20;
		int moveX = x + 20;
		int moveY = y + 25;
		drawKey(graphics, font, moveX + keyW, moveY, keyW, keyH, "W", command != null && command.forward());
		drawKey(graphics, font, moveX, moveY + 22, keyW, keyH, "A", command != null && command.left());
		drawKey(graphics, font, moveX + keyW, moveY + 22, keyW, keyH, "S", command != null && command.backward());
		drawKey(graphics, font, moveX + keyW * 2, moveY + 22, keyW, keyH, "D", command != null && command.right());

		// Every non-directional button is shown explicitly as its own keycap.
		int actionX = x + 104;
		int actionY = y + 25;
		drawKey(graphics, font, actionX, actionY, 50, keyH, "SPACE", command != null && command.jump());
		drawKey(graphics, font, actionX + 54, actionY, 56, keyH, "SPRINT", command != null && command.sprint());
		drawKey(graphics, font, actionX, actionY + 22, 50, keyH, "SNEAK", command != null && command.sneak());
		drawKey(graphics, font, actionX + 54, actionY + 22, 56, keyH, "LMB", isPressed(command, true, now));
		drawKey(graphics, font, actionX, actionY + 44, 50, keyH, "RMB", isPressed(command, false, now));
		graphics.drawString(font, "USE ITEM", actionX + 55, actionY + 50, 0xFF93A6BC, false);

		boolean lookActive = command != null && (command.yawDelta() != 0.0F || command.pitchDelta() != 0.0F);
		boolean showLook = lookActive || now - lookPulseAt <= PULSE_NANOS;
		String look = showLook
				? String.format(java.util.Locale.ROOT, "LOOK   YAW %+.2f°   PITCH %+.2f°", lastYawDelta, lastPitchDelta)
				: "LOOK   YAW +0.00°   PITCH +0.00°";
		int knockbackCombo = MinecraftAIClient.getCurrentKnockbackCombo();
		graphics.drawString(font, String.format(java.util.Locale.ROOT, "KNOCKBACK COMBO // %02d", knockbackCombo),
				x + 8, y + 101, knockbackCombo > 0 ? MAGENTA : 0xFF718198, false);
		graphics.drawString(font, look, x + 8, y + 117, showLook ? MAGENTA : 0xFF718198, false);
	}

	private static boolean isPressed(ControlCommand command, boolean attack, long now) {
		if (command != null && (attack ? command.attack() : command.use())) return true;
		long pulseAt = attack ? attackPulseAt : usePulseAt;
		return pulseAt != 0 && now - pulseAt <= PULSE_NANOS;
	}

	private static void drawKey(GuiGraphics graphics, Font font, int x, int y, int width, int height,
			String label, boolean active) {
		int edge = active ? CYAN : BORDER;
		int face = active ? 0xFF073542 : 0xFF111A29;
		int text = active ? 0xFFB9FAFF : 0xFF8291A8;
		graphics.fill(x, y, x + width, y + height, edge);
		graphics.fill(x + 1, y + 1, x + width - 1, y + height - 1, face);
		if (active) graphics.fill(x + 2, y + height - 3, x + width - 2, y + height - 2, MAGENTA);
		graphics.drawString(font, label, x + (width - font.width(label)) / 2,
				y + (height - font.lineHeight) / 2, text, false);
	}
}
