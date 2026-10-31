package com.grim.minecraftai.client;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

import java.util.Comparator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class ObservationBuilder {
	private static final Map<Integer, AttackAttempt> ATTACKS = new HashMap<>();
	private static long lastOutgoingHitTick = -1;
	private static long lastIncomingHitTick = -1;
	private static String lastHitType = "none";
	private static boolean lastHitCritical;
	private static boolean lastHitKnockback;
	private static boolean lastHitSweep;
	private static boolean lastHitNormal;
	private static int hitCombo;
	private static int knockbackCombo;
	private static long tick;
	private static volatile DebugProjection debugProjection = DebugProjection.EMPTY;

	private ObservationBuilder() { }

	static void beginTick() { tick++; }
	static long currentTick() { return tick; }

	static String lastHitHudText() {
		if (lastHitType.equals("none")) return "Last hit: none yet";
		StringBuilder text = new StringBuilder("Last hit: ");
		if (lastHitSweep) text.append("Sweep");
		if (lastHitCritical) appendHitLabel(text, "Critical");
		if (lastHitKnockback) appendHitLabel(text, "Knockback");
		if (lastHitNormal) appendHitLabel(text, "Normal");
		return text.toString();
	}

	private static void appendHitLabel(StringBuilder text, String label) {
		if (text.length() > "Last hit: ".length()) text.append(" + ");
		text.append(label);
	}

	static void recordAttackAttempt(
			Player attacker, Player target, boolean critical, boolean knockback, boolean sweep
	) {
		ATTACKS.put(target.getId(), new AttackAttempt(tick, critical, knockback, sweep));
		ATTACKS.entrySet().removeIf(entry -> tick - entry.getValue().tick() > 3);
	}

	static void recordDamageEvent(int damagedEntityId, int sourceCauseId) {
		Minecraft client = Minecraft.getInstance();
		Player self = client.player;
		if (self == null) return;
		if (damagedEntityId == self.getId()) {
			Player source = client.level == null ? null : client.level.players().stream()
					.filter(player -> player.getId() == sourceCauseId).findFirst().orElse(null);
			if (source != null && source != self) {
				lastIncomingHitTick = tick;
				hitCombo = 0;
				knockbackCombo = 0;
			}
			return;
		}
		AttackAttempt attempt = ATTACKS.remove(damagedEntityId);
		if (sourceCauseId != self.getId() || attempt == null || tick - attempt.tick() > 3) return;
		lastOutgoingHitTick = tick;
		hitCombo++;
		lastHitCritical = attempt.critical();
		lastHitKnockback = attempt.knockback();
		knockbackCombo = lastHitKnockback ? knockbackCombo + 1 : 0;
		lastHitSweep = attempt.sweep();
		lastHitNormal = !lastHitCritical && !lastHitKnockback && !lastHitSweep;
		lastHitType = lastHitSweep ? "sweep" : lastHitCritical ? "critical" : lastHitKnockback ? "knockback" : "normal";
	}

	static JsonObject build(Minecraft client, PythonControlServer server) {
		JsonObject frame = new JsonObject();
		frame.addProperty("type", "observation");
		frame.addProperty("schema", "minecraftai.pvp.v1");
		frame.addProperty("tick", tick);
		JsonObject f = new JsonObject();
		f.add("remote_control_diagnostics", RemoteInputDiagnostics.snapshot(client));
		Player self = client.player;
		if (self == null || client.level == null) {
			f.addProperty("player_valid", false);
			frame.add("features", f);
			frame.add("action_history", actionHistory(server));
			return frame;
		}
		f.addProperty("player_valid", true);
		f.addProperty("player_x", self.getX());
		f.addProperty("player_y", self.getY());
		f.addProperty("player_z", self.getZ());
		f.addProperty("player_y_level", self.getY());
		Vec3 velocity = self.getDeltaMovement();
		f.addProperty("player_velocity_x", velocity.x);
		f.addProperty("player_velocity_y", velocity.y);
		f.addProperty("player_velocity_z", velocity.z);
		f.addProperty("player_horizontal_speed", Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z));
		f.addProperty("camera_yaw", self.getYRot());
		f.addProperty("camera_pitch", self.getXRot());
		f.addProperty("camera_yaw_sin", Math.sin(Math.toRadians(self.getYRot())));
		f.addProperty("camera_yaw_cos", Math.cos(Math.toRadians(self.getYRot())));
		f.addProperty("camera_pitch_sin", Math.sin(Math.toRadians(self.getXRot())));
		f.addProperty("camera_pitch_cos", Math.cos(Math.toRadians(self.getXRot())));
		double verticalFov = Math.toRadians(client.options.fov().get());
		double screenAspect = (double) client.getWindow().getWidth() / Math.max(1, client.getWindow().getHeight());
		f.addProperty("camera_horizontal_fov_degrees",
				Math.toDegrees(2.0 * Math.atan(Math.tan(verticalFov / 2.0) * screenAspect)));
		f.addProperty("player_on_ground", self.onGround());
		f.addProperty("player_sprinting", self.isSprinting());
		f.addProperty("player_sneaking", self.isCrouching());
		f.addProperty("player_fall_distance", self.fallDistance);
		f.addProperty("player_health", self.getHealth());
		f.addProperty("player_absorption", self.getAbsorptionAmount());
		f.addProperty("player_food", self.getFoodData().getFoodLevel());
		f.addProperty("attack_strength", self.getAttackStrengthScale(0.0F));
		f.addProperty("mainhand_is_sword", self.getMainHandItem().is(ItemTags.SWORDS));
		f.addProperty("ticks_since_hit_enemy", elapsed(lastOutgoingHitTick));
		f.addProperty("has_hit_enemy", lastOutgoingHitTick >= 0);
		f.addProperty("ticks_since_hit_by_enemy", elapsed(lastIncomingHitTick));
		f.addProperty("has_been_hit", lastIncomingHitTick >= 0);
		f.addProperty("last_hit_critical", lastHitCritical);
		f.addProperty("last_hit_knockback", lastHitKnockback);
		f.addProperty("last_hit_sweep", lastHitSweep);
		f.addProperty("last_hit_normal", lastHitNormal);
		f.addProperty("last_hit_type", lastHitType);
		f.addProperty("hit_combo", hitCombo);
		f.addProperty("knockback_combo", knockbackCombo);

		Player target = client.level.players().stream()
				.filter(p -> p != self && p.isAlive() && !p.isSpectator() && !self.isAlliedTo(p) && self.hasLineOfSight(p))
				.min(Comparator.comparingDouble(self::distanceToSqr)).orElse(null);
		addTargetFeatures(client, self, target, f);
		frame.add("features", f);
		frame.add("action_history", actionHistory(server));
		return frame;
	}

	private static JsonArray actionHistory(PythonControlServer server) {
		JsonArray history = new JsonArray();
		for (var action : server.actionHistory()) {
			JsonArray row = new JsonArray();
			for (Object value : action) row.add((Number) (value instanceof Boolean ? ((Boolean) value ? 1 : 0) : value));
			history.add(row);
		}
		return history;
	}

	private static void addTargetFeatures(Minecraft client, Player self, Player target, JsonObject f) {
		f.addProperty("target_valid", target != null);
		if (target == null) {
			f.addProperty("target_distance", 0.0);
			f.addProperty("target_horizontal_distance", 0.0);
			f.addProperty("target_height", 0.0);
			f.addProperty("target_health", 0.0);
			f.addProperty("target_hurt_time", 0);
			f.addProperty("target_on_ground", false);
			f.addProperty("target_sprinting", false);
			f.addProperty("target_sneaking", false);
			f.addProperty("target_yaw", 0.0);
			f.addProperty("target_pitch", 0.0);
			f.addProperty("target_relative_velocity_x", 0.0);
			f.addProperty("target_relative_velocity_y", 0.0);
			f.addProperty("target_relative_velocity_z", 0.0);
			f.addProperty("target_relative_horizontal_speed", 0.0);
			f.addProperty("target_on_screen", false);
			f.addProperty("target_bbox_projection_valid", false);
			f.addProperty("crosshair_on_target", false);
			f.addProperty("target_aim_projection_valid", false);
			f.addProperty("target_aim_dx", 0.0);
			f.addProperty("target_aim_dy", 0.0);
			f.addProperty("target_dx", 0.0);
			f.addProperty("target_dy", 0.0);
			f.addProperty("target_bbox_left", 0.0);
			f.addProperty("target_bbox_top", 0.0);
			f.addProperty("target_bbox_right", 0.0);
			f.addProperty("target_bbox_bottom", 0.0);
			f.addProperty("target_bbox_width", 0.0);
			f.addProperty("target_bbox_height", 0.0);
			debugProjection = DebugProjection.EMPTY;
			return;
		}
		Vec3 relative = target.position().subtract(self.position());
		Vec3 relativeVelocity = target.getDeltaMovement().subtract(self.getDeltaMovement());
		f.addProperty("target_distance", self.distanceTo(target));
		f.addProperty("target_height", target.getBbHeight());
		f.addProperty("target_hurt_time", target.hurtTime);
		f.addProperty("target_on_ground", target.onGround());
		f.addProperty("target_sprinting", target.isSprinting());
		f.addProperty("target_sneaking", target.isCrouching());
		f.addProperty("target_health", target.getHealth());
		f.addProperty("target_relative_velocity_x", relativeVelocity.x);
		f.addProperty("target_relative_velocity_y", relativeVelocity.y);
		f.addProperty("target_relative_velocity_z", relativeVelocity.z);
		f.addProperty("target_relative_horizontal_speed", Math.sqrt(relativeVelocity.x * relativeVelocity.x + relativeVelocity.z * relativeVelocity.z));
		double yaw = Math.toRadians(self.getYRot());
		double pitch = Math.toRadians(self.getXRot());
		Vec3 forward = new Vec3(-Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
		// Minecraft yaw 0 looks toward +Z. The camera's screen-right axis therefore
		// points toward -X (the old basis was mirrored horizontally).
		Vec3 right = new Vec3(-Math.cos(yaw), 0.0, -Math.sin(yaw));
		Vec3 up = right.cross(forward).normalize();
		Vec3 eye = self.getEyePosition();
		double vfov = Math.toRadians(client.options.fov().get());
		double aspect = (double) client.getWindow().getWidth() / Math.max(1, client.getWindow().getHeight());
		double tanV = Math.tan(vfov / 2.0);
		double tanH = tanV * aspect;
		AABB box = target.getBoundingBox();
		CameraPoint[] corners = new CameraPoint[8];
		for (int mask = 0; mask < 8; mask++) {
			Vec3 corner = new Vec3((mask & 1) == 0 ? box.minX : box.maxX,
					(mask & 2) == 0 ? box.minY : box.maxY,
					(mask & 4) == 0 ? box.minZ : box.maxZ);
			Vec3 delta = corner.subtract(eye);
			corners[mask] = new CameraPoint(delta.dot(right), delta.dot(up), delta.dot(forward));
		}
		// Clip each of the six AABB faces against the camera frustum before
		// perspective division. Projecting only the original corners fails when
		// an edge crosses the near plane and can produce screen-sized boxes.
		List<ScreenPoint> projected = clipBoxToView(corners, tanH, tanV, 0.05);
		boolean anyInFront = Arrays.stream(corners).anyMatch(point -> point.depth() > 0.05);
		if (eye.x >= box.minX && eye.x <= box.maxX && eye.y >= box.minY && eye.y <= box.maxY
				&& eye.z >= box.minZ && eye.z <= box.maxZ) {
			// When the camera is inside the target's collision box, its clipped
			// surface is behind the near plane; represent that overlap explicitly.
			projected = List.of(new ScreenPoint(0, 0), new ScreenPoint(1, 0),
					new ScreenPoint(1, 1), new ScreenPoint(0, 1));
			anyInFront = true;
		}
		double minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0;
		for (ScreenPoint point : projected) {
			minX = Math.min(minX, point.x()); maxX = Math.max(maxX, point.x());
			minY = Math.min(minY, point.y()); maxY = Math.max(maxY, point.y());
		}
		boolean boxOnScreen = !projected.isEmpty();
		boolean onScreen = boxOnScreen;
		f.addProperty("target_bbox_projection_valid", anyInFront);
		f.addProperty("target_on_screen", onScreen);
		// Keep the target-center observation independent from the clipped box.
		Vec3 centerDelta = new Vec3(target.getX(), target.getY() + target.getBbHeight() * 0.5, target.getZ()).subtract(eye);
		double centerDepth = centerDelta.dot(forward);
		boolean centerProjectionValid = centerDepth > 0.05;
		double centerX = centerProjectionValid ? 0.5 + 0.5 * centerDelta.dot(right) / (centerDepth * tanH) : 0.0;
		double centerY = centerProjectionValid ? 0.5 - 0.5 * centerDelta.dot(up) / (centerDepth * tanV) : 0.0;
		f.addProperty("target_dx", centerProjectionValid ? centerX - 0.5 : 0.0);
		f.addProperty("target_dy", centerProjectionValid ? centerY - 0.5 : 0.0);
		// Project an explicit point near the center of the target's head instead of estimating
		// it from a screen-clipped bounding box. The latter collapses or shifts
		// when only part of the target is inside the viewport.
		Vec3 aimPoint = new Vec3(target.getX(), target.getY() + target.getBbHeight() * 0.90, target.getZ());
		Vec3 aimDelta = aimPoint.subtract(eye);
		double aimDepth = aimDelta.dot(forward);
		boolean aimProjectionValid = aimDepth > 0.05;
		f.addProperty("target_aim_projection_valid", aimProjectionValid);
		double aimDx = aimProjectionValid
				? 0.5 + 0.5 * aimDelta.dot(right) / (aimDepth * tanH)
				: 0.0;
		f.addProperty("target_aim_dx", aimDx);
		f.addProperty("target_aim_dy", aimProjectionValid
				? 0.5 - 0.5 * aimDelta.dot(up) / (aimDepth * tanV)
				: 0.0);
		boolean crosshairOnTarget = client.hitResult instanceof net.minecraft.world.phys.EntityHitResult hit
				&& hit.getEntity() == target;
		debugProjection = new DebugProjection(anyInFront, onScreen,
				boxOnScreen ? clamp01(minX) : 0.0, boxOnScreen ? clamp01(minY) : 0.0,
				boxOnScreen ? clamp01(maxX) : 0.0, boxOnScreen ? clamp01(maxY) : 0.0, aimProjectionValid, aimDx,
				aimProjectionValid ? 0.5 - 0.5 * aimDelta.dot(up) / (aimDepth * tanV) : 0.0,
				centerProjectionValid, centerX, centerY, true, crosshairOnTarget, self.distanceTo(target),
				relativeVelocity.x, relativeVelocity.y, relativeVelocity.z, target.onGround(),
				target.isSprinting(), target.isCrouching());
		f.addProperty("target_bbox_left", boxOnScreen ? clamp01(minX) : 0.0);
		f.addProperty("target_bbox_top", boxOnScreen ? clamp01(minY) : 0.0);
		f.addProperty("target_bbox_right", boxOnScreen ? clamp01(maxX) : 0.0);
		f.addProperty("target_bbox_bottom", boxOnScreen ? clamp01(maxY) : 0.0);
		f.addProperty("target_bbox_width", boxOnScreen ? clamp01(maxX) - clamp01(minX) : 0.0);
		f.addProperty("target_bbox_height", boxOnScreen ? clamp01(maxY) - clamp01(minY) : 0.0);
		f.addProperty("crosshair_on_target", crosshairOnTarget);
		f.addProperty("target_horizontal_distance", Math.sqrt(relative.x * relative.x + relative.z * relative.z));
		f.addProperty("target_yaw", target.getYRot());
		f.addProperty("target_pitch", target.getXRot());
	}

	private static List<ScreenPoint> clipBoxToView(CameraPoint[] corners, double tanH, double tanV, double near) {
		int[][] faces = {{0, 2, 6, 4}, {1, 3, 7, 5}, {0, 1, 5, 4},
				{2, 3, 7, 6}, {0, 1, 3, 2}, {4, 5, 7, 6}};
		List<ScreenPoint> result = new ArrayList<>();
		for (int[] face : faces) {
			List<CameraPoint> polygon = new ArrayList<>(4);
			for (int index : face) polygon.add(corners[index]);
			for (int plane = 0; plane < 5 && !polygon.isEmpty(); plane++) {
				List<CameraPoint> clipped = new ArrayList<>();
				CameraPoint previous = polygon.get(polygon.size() - 1);
				double previousDistance = planeDistance(previous, plane, tanH, tanV, near);
				for (CameraPoint current : polygon) {
					double currentDistance = planeDistance(current, plane, tanH, tanV, near);
					boolean previousInside = previousDistance >= 0.0;
					boolean currentInside = currentDistance >= 0.0;
					if (previousInside != currentInside) {
						double fraction = previousDistance / (previousDistance - currentDistance);
						clipped.add(previous.interpolate(current, fraction));
					}
					if (currentInside) clipped.add(current);
					previous = current;
					previousDistance = currentDistance;
				}
				polygon = clipped;
			}
			for (CameraPoint point : polygon) {
				result.add(new ScreenPoint(0.5 + 0.5 * point.x() / (point.depth() * tanH),
						0.5 - 0.5 * point.y() / (point.depth() * tanV)));
			}
		}
		return result;
	}

	private static double planeDistance(CameraPoint point, int plane, double tanH, double tanV, double near) {
		return switch (plane) {
			case 0 -> point.depth() - near;
			case 1 -> point.x() + point.depth() * tanH;
			case 2 -> point.depth() * tanH - point.x();
			case 3 -> point.y() + point.depth() * tanV;
			default -> point.depth() * tanV - point.y();
		};
	}

	private record CameraPoint(double x, double y, double depth) {
		private CameraPoint interpolate(CameraPoint other, double t) {
			return new CameraPoint(x + (other.x - x) * t, y + (other.y - y) * t,
					depth + (other.depth - depth) * t);
		}
	}
	private record ScreenPoint(double x, double y) { }

	static void renderDebugOverlay(net.minecraft.client.gui.GuiGraphics graphics, int width, int height, int verticalOffset) {
		Minecraft client = Minecraft.getInstance();
		DebugProjection projection = client.player == null ? DebugProjection.EMPTY : debugProjection;
		int textColor = 0xFFFFFFFF;
		int y = 39 + verticalOffset;
		graphics.fill(4, 34 + verticalOffset, Math.max(8, Math.min(width - 4, 760)),
				Math.min(height - 4, 170 + verticalOffset), 0x90000000);
		String targetState = !projection.targetValid ? "Target: none (alive, visible, non-allied player required)"
				: String.format(java.util.Locale.ROOT, "Target: %.2fm | screen=%s | crosshair hit=%s | box=%s",
						projection.targetDistance, projection.onScreen ? "yes" : "no",
						projection.crosshairOnTarget ? "YES" : "no",
						projection.boxValid ? "valid" : "behind camera");
		graphics.drawString(client.font, "MinecraftAI diagnostics — F8 hides this panel", 8, y, 0xFF66FFFF, true); y += 12;
		graphics.drawString(client.font, targetState, 8, y, textColor, true); y += 12;
		String center = projection.centerValid
				? String.format(java.util.Locale.ROOT,
						"Target center screen UV: x=%.3f y=%.3f | reticle error: dx=%+.3f dy=%+.3f",
						projection.centerX, projection.centerY, projection.centerX - 0.5, projection.centerY - 0.5)
				: "Target center screen: invalid / behind camera";
		String head = projection.aimValid
				? String.format(java.util.Locale.ROOT,
						"Head point screen UV:   x=%.3f y=%.3f | reticle error: dx=%+.3f dy=%+.3f",
						projection.aimDx, projection.aimDy, projection.aimDx - 0.5, projection.aimDy - 0.5)
				: "Head point screen:   invalid / behind camera";
		graphics.drawString(client.font, center, 8, y, 0xFF55FFFF, true); y += 12;
		graphics.drawString(client.font, head, 8, y, 0xFFFF66FF, true); y += 12;

		if (projection.targetValid) {
			String relativeVelocity = String.format(java.util.Locale.ROOT,
					"Target relative velocity x/y/z: %.2f / %.2f / %.2f | ground=%s sprint=%s sneak=%s",
					projection.relativeVelocityX, projection.relativeVelocityY, projection.relativeVelocityZ,
					projection.targetOnGround, projection.targetSprinting, projection.targetSneaking);
			graphics.drawString(client.font, relativeVelocity, 8, y, textColor, true); y += 12;
		}

		if (client.player != null) {
			Player self = client.player;
			graphics.drawString(client.font, String.format(java.util.Locale.ROOT,
					"Camera yaw=%.1f pitch=%.1f | player Y=%.2f speed=%.2f | ground=%s sprint=%s sneak=%s",
					self.getYRot(), self.getXRot(), self.getY(), Math.sqrt(self.getDeltaMovement().x * self.getDeltaMovement().x
							+ self.getDeltaMovement().z * self.getDeltaMovement().z), self.onGround(),
					self.isSprinting(), self.isCrouching()), 8, y, textColor, true); y += 12;
			graphics.drawString(client.font, String.format(java.util.Locale.ROOT,
					"Player health=%.1f food=%d attack charge=%.2f sword=%s",
					self.getHealth(), self.getFoodData().getFoodLevel(), self.getAttackStrengthScale(0.0F),
					self.getMainHandItem().is(ItemTags.SWORDS)), 8, y, textColor, true); y += 12;
		}

		String hitText = lastHitHudText();
		String outgoing = lastOutgoingHitTick < 0 ? "never" : elapsed(lastOutgoingHitTick) + " ticks ago";
		String incoming = lastIncomingHitTick < 0 ? "never" : elapsed(lastIncomingHitTick) + " ticks ago";
		graphics.drawString(client.font, "Last outgoing hit: " + hitText.substring("Last hit: ".length())
				+ " (" + outgoing + ") | incoming damage: " + incoming, 8, y,
				lastHitType.equals("none") ? 0xFFAAAAAA : 0xFFFFD45C, true); y += 12;

		JsonObject input = RemoteInputDiagnostics.snapshot(client);
		JsonObject state = input.getAsJsonObject("action_state");
		graphics.drawString(client.font, String.format(java.util.Locale.ROOT,
				"Keys F/B/L/R/J/S/Sneak/Attack/Use: %s/%s/%s/%s/%s/%s/%s/%s/%s",
				state.get("forward").getAsBoolean(), state.get("backward").getAsBoolean(),
				state.get("left").getAsBoolean(), state.get("right").getAsBoolean(),
				state.get("jump").getAsBoolean(), state.get("sprint").getAsBoolean(),
				state.get("sneak").getAsBoolean(), state.get("attack").getAsBoolean(),
				state.get("use").getAsBoolean()), 8, y, textColor, true); y += 12;
		JsonObject clicks = input.getAsJsonObject("consumed_clicks");
		graphics.drawString(client.font, "Remote click counts attack/use: " + clicks.get("attack").getAsLong() + "/"
				+ clicks.get("use").getAsLong() + " | applied command seq: "
				+ input.get("last_applied_sequence").getAsLong(), 8, y, textColor, true);

		if (projection.boxValid && projection.onScreen) {
			int left = (int) (projection.left * width);
			int top = (int) (projection.top * height);
			int right = (int) (projection.right * width);
			int bottom = (int) (projection.bottom * height);
			graphics.fill(left, top, right + 1, top + 2, 0xFF00FF00);
			graphics.fill(left, bottom - 1, right + 1, bottom + 1, 0xFF00FF00);
			graphics.fill(left, top, left + 2, bottom + 1, 0xFF00FF00);
			graphics.fill(right - 1, top, right + 1, bottom + 1, 0xFF00FF00);
		}
		if (projection.centerValid && projection.centerX >= 0.0 && projection.centerX <= 1.0
				&& projection.centerY >= 0.0 && projection.centerY <= 1.0) {
			int centerX = (int) (projection.centerX * width);
			int centerY = (int) (projection.centerY * height);
			graphics.fill(centerX - 4, centerY - 1, centerX + 5, centerY + 2, 0xFF55FFFF);
			graphics.fill(centerX - 1, centerY - 4, centerX + 2, centerY + 5, 0xFF55FFFF);
		}
		if (projection.aimValid && projection.aimDx >= 0.0 && projection.aimDx <= 1.0
				&& projection.aimDy >= 0.0 && projection.aimDy <= 1.0) {
			int headX = (int) (projection.aimDx * width);
			int headY = (int) (projection.aimDy * height);
			graphics.fill(headX - 5, headY - 1, headX + 6, headY + 2, 0xFFFF33FF);
			graphics.fill(headX - 1, headY - 5, headX + 2, headY + 6, 0xFFFF33FF);
		}
	}

	private static int elapsed(long then) { return then < 0 ? -1 : (int) Math.min(1_000_000, tick - then); }
	static int currentCombo() { return hitCombo; }
	static int currentKnockbackCombo() { return knockbackCombo; }
	private static double clamp01(double value) { return Math.max(0.0, Math.min(1.0, value)); }
	private record DebugProjection(boolean boxValid, boolean onScreen, double left, double top, double right,
			double bottom, boolean aimValid, double aimDx, double aimDy, boolean centerValid, double centerX,
			double centerY, boolean targetValid, boolean crosshairOnTarget, double targetDistance,
			double relativeVelocityX, double relativeVelocityY, double relativeVelocityZ,
			boolean targetOnGround, boolean targetSprinting, boolean targetSneaking) {
		private static final DebugProjection EMPTY = new DebugProjection(false, false, 0, 0, 0, 0,
				false, 0, 0, false, 0, 0, false, false, 0, 0, 0, 0, false, false, false);
	}
	private record AttackAttempt(long tick, boolean critical, boolean knockback, boolean sweep) { }
}
