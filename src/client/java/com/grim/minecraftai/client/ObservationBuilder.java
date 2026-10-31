package com.grim.minecraftai.client;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.core.Holder;
import net.minecraft.tags.ItemTags;
import net.minecraft.util.Mth;
import net.minecraft.world.effect.MobEffect;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class ObservationBuilder {
	static final String SCHEMA = "minecraftai.pvp.v2";

	/** Ticks a tracked target may stay out of sight before we switch to a visible one. */
	private static final int TARGET_SWITCH_AFTER_TICKS = 20;
	private static final int VELOCITY_HISTORY = 5;
	private static final double VELOCITY_SMOOTHING = 0.5;
	private static final String[] ITEM_CATEGORIES = {
			"empty", "sword", "axe", "ranged", "trident_or_mace", "ender_pearl",
			"golden_apple", "potion", "shield", "fishing_rod", "block", "other"
	};

	private static final Map<Integer, AttackAttempt> ATTACKS = new HashMap<>();
	private static final Map<String, Object> TARGET_DEFAULTS = targetDefaults();
	private static final ArrayDeque<Vec3> relVelHistory = new ArrayDeque<>();

	private static long tick;
	private static volatile DebugProjection debugProjection = DebugProjection.EMPTY;

	// ---- combat history
	private static long lastOutgoingHitTick = -1;
	private static long lastIncomingHitTick = -1;
	private static String lastHitType = "none";
	private static boolean lastHitCritical;
	private static boolean lastHitKnockback;
	private static boolean lastHitSweep;
	private static boolean lastHitNormal;
	private static int hitCombo;
	private static int knockbackCombo;
	private static String lastAttackOutcome = "none";
	private static long lastAttackOutcomeTick = -1;

	// ---- events: accumulate between ticks, published for the current tick
	private static int pendingHitsDealt;
	private static int pendingHitsTaken;
	private static String pendingOutcome = "none";
	private static int eventHitsDealt;
	private static int eventHitsTaken;
	private static String eventOutcome = "none";
	private static double eventSelfHealthDelta;
	private static double eventTargetHealthDelta;
	private static boolean attackAttemptThisTick;

	// ---- self tracking
	private static Vec3 prevSelfPos;
	private static Vec3 selfPosVelocity = Vec3.ZERO;
	private static Vec3 prevSelfMotion;
	private static boolean prevOnGround;
	private static boolean prevSprinting;
	private static boolean prevSwinging;
	private static int prevSwingTime;
	private static int ticksOnGround;
	private static int ticksInAir;
	private static long lastJumpTick = -1;
	private static long lastSprintToggleTick = -1;
	private static long lastAttackSwingTick = -1;
	private static long lastKnockedBackTick = -1;
	private static float prevSelfHealth = -1.0F;

	// ---- target tracking (sticky target + line-of-sight memory)
	private static int targetId = -1;
	private static Player trackedTarget;
	private static boolean targetVisible;
	private static long lastSeenTick = -1;
	private static Vec3 lastKnownCenter;
	private static Vec3 prevTargetPos;
	private static Vec3 relVelSmoothed = Vec3.ZERO;
	private static Vec3 relAccel = Vec3.ZERO;
	private static double hitboxDistance;
	private static double prevHitboxDistance = -1.0;
	private static double closingSpeed;
	private static float prevTargetHealth = -1.0F;

	private ObservationBuilder() { }

	static long currentTick() { return tick; }

	// =====================================================================
	// Per-tick tracking. Runs every client tick, even when AI control is off,
	// so counters and deltas stay consistent.
	// =====================================================================

	static void beginTick(Minecraft client) {
		tick++;
		eventHitsDealt = pendingHitsDealt;
		eventHitsTaken = pendingHitsTaken;
		eventOutcome = pendingOutcome;
		pendingHitsDealt = 0;
		pendingHitsTaken = 0;
		pendingOutcome = "none";
		expireAttacks();

		Player self = client.player;
		if (self == null || client.level == null) {
			resetSelfTracking();
			resetTarget();
			targetId = -1;
			eventSelfHealthDelta = 0.0;
			attackAttemptThisTick = false;
			return;
		}
		trackSelf(client, self);
		trackTarget(client, self);
		if (!eventOutcome.equals("none")) {
			lastAttackOutcome = eventOutcome;
			lastAttackOutcomeTick = tick;
		}
		attackAttemptThisTick = false;
	}

	private static void setEventOutcome(String outcome) {
		if (eventOutcome.equals("none")) eventOutcome = outcome;
	}

	private static void expireAttacks() {
		Iterator<Map.Entry<Integer, AttackAttempt>> it = ATTACKS.entrySet().iterator();
		while (it.hasNext()) {
			AttackAttempt attempt = it.next().getValue();
			if (tick - attempt.tick() > 3) {
				it.remove();
				// Attack connected with the entity but produced no damage event:
				// shield block, or the target's invulnerability frames.
				setEventOutcome(attempt.targetBlocking() ? "blocked" : "no_effect");
			}
		}
	}

	private static void resetSelfTracking() {
		prevSelfPos = null;
		selfPosVelocity = Vec3.ZERO;
		prevSelfMotion = null;
		prevSelfHealth = -1.0F;
		ticksOnGround = 0;
		ticksInAir = 0;
	}

	private static void trackSelf(Minecraft client, Player self) {
		Vec3 pos = self.position();
		selfPosVelocity = prevSelfPos == null ? Vec3.ZERO : pos.subtract(prevSelfPos);
		prevSelfPos = pos;

		boolean onGround = self.onGround();
		if (onGround) { ticksOnGround++; ticksInAir = 0; } else { ticksInAir++; ticksOnGround = 0; }
		Vec3 motion = self.getDeltaMovement();
		if (prevOnGround && !onGround && motion.y > 0.0) lastJumpTick = tick;
		prevOnGround = onGround;

		boolean sprinting = self.isSprinting();
		if (sprinting != prevSprinting) lastSprintToggleTick = tick;
		prevSprinting = sprinting;

		// Swing start: a new swing begins when the swing flag rises or the swing timer restarts.
		boolean swingStarted = self.swinging && (!prevSwinging || self.swingTime < prevSwingTime);
		prevSwinging = self.swinging;
		prevSwingTime = self.swingTime;
		if (swingStarted && client.options.keyAttack.isDown()) {
			lastAttackSwingTick = tick;
			if (!attackAttemptThisTick) setEventOutcome("miss");
		}

		// Knockback heuristic: a large horizontal velocity jump while recently hurt.
		if (prevSelfMotion != null && self.hurtTime > 5) {
			double dx = motion.x - prevSelfMotion.x;
			double dz = motion.z - prevSelfMotion.z;
			if (Math.sqrt(dx * dx + dz * dz) > 0.3) lastKnockedBackTick = tick;
		}
		prevSelfMotion = motion;

		float health = self.getHealth() + self.getAbsorptionAmount();
		eventSelfHealthDelta = prevSelfHealth < 0.0F ? 0.0 : health - prevSelfHealth;
		prevSelfHealth = health;
	}

	private static boolean eligible(Player self, Player p) {
		return p != self && p.isAlive() && !p.isSpectator() && !self.isAlliedTo(p);
	}

	private static Player selectTarget(Minecraft client, Player self) {
		Player current = null;
		if (targetId >= 0) {
			for (Player p : client.level.players()) {
				if (p.getId() == targetId && eligible(self, p)) { current = p; break; }
			}
		}
		boolean currentVisible = current != null && self.hasLineOfSight(current);
		if (current != null && (currentVisible || tick - lastSeenTick <= TARGET_SWITCH_AFTER_TICKS)) {
			return current;
		}
		Player best = client.level.players().stream()
				.filter(p -> eligible(self, p) && self.hasLineOfSight(p))
				.min(Comparator.comparingDouble(self::distanceToSqr)).orElse(null);
		return best != null ? best : current;
	}

	private static void resetTarget() {
		trackedTarget = null;
		targetVisible = false;
		lastSeenTick = -1;
		lastKnownCenter = null;
		prevTargetPos = null;
		relVelSmoothed = Vec3.ZERO;
		relAccel = Vec3.ZERO;
		relVelHistory.clear();
		hitboxDistance = 0.0;
		prevHitboxDistance = -1.0;
		closingSpeed = 0.0;
		prevTargetHealth = -1.0F;
		eventTargetHealthDelta = 0.0;
	}

	private static void trackTarget(Minecraft client, Player self) {
		eventTargetHealthDelta = 0.0;
		closingSpeed = 0.0;
		Player target = selectTarget(client, self);
		int newId = target == null ? -1 : target.getId();
		if (newId != targetId) {
			resetTarget();
			targetId = newId;
		}
		trackedTarget = target;
		if (target == null) {
			targetVisible = false;
			return;
		}
		targetVisible = self.hasLineOfSight(target);
		if (targetVisible) {
			lastSeenTick = tick;
			lastKnownCenter = center(target);
		}

		// Remote-player velocity is inferred from position deltas, then smoothed and stacked.
		Vec3 pos = target.position();
		if (prevTargetPos != null) {
			Vec3 rel = pos.subtract(prevTargetPos).subtract(selfPosVelocity);
			boolean first = relVelHistory.isEmpty();
			Vec3 previous = relVelSmoothed;
			relVelSmoothed = first ? rel : previous.scale(1.0 - VELOCITY_SMOOTHING).add(rel.scale(VELOCITY_SMOOTHING));
			relAccel = first ? Vec3.ZERO : relVelSmoothed.subtract(previous);
			relVelHistory.addLast(rel);
			while (relVelHistory.size() > VELOCITY_HISTORY) relVelHistory.removeFirst();
		}
		prevTargetPos = pos;

		hitboxDistance = hitboxDistance(self.getEyePosition(), target.getBoundingBox());
		closingSpeed = prevHitboxDistance < 0.0 ? 0.0 : prevHitboxDistance - hitboxDistance;
		prevHitboxDistance = hitboxDistance;

		float health = target.getHealth() + target.getAbsorptionAmount();
		eventTargetHealthDelta = prevTargetHealth < 0.0F ? 0.0 : health - prevTargetHealth;
		prevTargetHealth = health;
	}

	// =====================================================================
	// Events from mixins
	// =====================================================================

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
		ATTACKS.put(target.getId(), new AttackAttempt(tick, critical, knockback, sweep, target.isBlocking()));
		attackAttemptThisTick = true;
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
				pendingHitsTaken++;
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
		pendingHitsDealt++;
		pendingOutcome = "hit";
	}

	// =====================================================================
	// Observation
	// =====================================================================

	static JsonObject build(Minecraft client, PythonControlServer server) {
		JsonObject frame = new JsonObject();
		frame.addProperty("type", "observation");
		frame.addProperty("schema", SCHEMA);
		frame.addProperty("tick", tick);
		frame.addProperty("timestamp_nanos", System.nanoTime());
		frame.addProperty("wall_time_millis", System.currentTimeMillis());
		frame.add("diagnostics", RemoteInputDiagnostics.snapshot(client));
		frame.add("events", events());

		JsonObject f = new JsonObject();
		Player self = client.player;
		if (self == null || client.level == null) {
			f.addProperty("player_valid", false);
			frame.add("features", f);
			frame.add("critic_only", criticOnly(null, null, 0.0));
			frame.add("action_history", actionHistory(server));
			return frame;
		}
		f.addProperty("player_valid", true);
		double yawRad = Math.toRadians(self.getYRot());
		Vec3 eye = self.getEyePosition();

		// ---- self: motion (egocentric)
		Vec3 velocity = self.getDeltaMovement();
		double[] egoVelocity = toEgo(velocity, yawRad);
		f.addProperty("player_velocity_forward", egoVelocity[0]);
		f.addProperty("player_velocity_right", egoVelocity[1]);
		f.addProperty("player_velocity_y", velocity.y);
		f.addProperty("player_horizontal_speed", Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z));
		f.addProperty("player_on_ground", self.onGround());
		f.addProperty("player_fall_distance", self.fallDistance);

		// ---- self: camera (yaw is intentionally unwrapped for continuous control)
		f.addProperty("camera_yaw", self.getYRot());
		f.addProperty("camera_pitch", self.getXRot());
		f.addProperty("camera_yaw_sin", Math.sin(yawRad));
		f.addProperty("camera_yaw_cos", Math.cos(yawRad));
		f.addProperty("camera_pitch_sin", Math.sin(Math.toRadians(self.getXRot())));
		f.addProperty("camera_pitch_cos", Math.cos(Math.toRadians(self.getXRot())));
		double verticalFov = Math.toRadians(client.options.fov().get());
		double screenAspect = (double) client.getWindow().getWidth() / Math.max(1, client.getWindow().getHeight());
		f.addProperty("camera_horizontal_fov_degrees",
				Math.toDegrees(2.0 * Math.atan(Math.tan(verticalFov / 2.0) * screenAspect)));

		// ---- self: stance and status
		f.addProperty("player_sprinting", self.isSprinting());
		f.addProperty("player_sneaking", self.isCrouching());
		f.addProperty("player_hurt_time", self.hurtTime);
		f.addProperty("player_invulnerable_ticks", self.invulnerableTime);
		f.addProperty("player_blocking", self.isBlocking());
		f.addProperty("player_using_item", self.isUsingItem());
		f.addProperty("player_on_fire", self.isOnFire());
		f.addProperty("player_in_water", self.isInWater());
		f.addProperty("player_in_lava", self.isInLava());
		f.addProperty("player_on_climbable", self.onClimbable());

		// ---- self: vitals, loadout, attributes, effects
		f.addProperty("player_health", self.getHealth());
		f.addProperty("player_absorption", self.getAbsorptionAmount());
		f.addProperty("player_food", self.getFoodData().getFoodLevel());
		f.addProperty("player_armor", self.getAttributeValue(Attributes.ARMOR));
		f.addProperty("player_armor_toughness", self.getAttributeValue(Attributes.ARMOR_TOUGHNESS));
		f.addProperty("player_attack_damage", self.getAttributeValue(Attributes.ATTACK_DAMAGE));
		double attackSpeed = self.getAttributeValue(Attributes.ATTACK_SPEED);
		f.addProperty("player_attack_speed", attackSpeed);
		f.addProperty("player_reach", self.entityInteractionRange());
		f.addProperty("mainhand_is_sword", self.getMainHandItem().is(ItemTags.SWORDS));
		f.addProperty("effect_speed", effectLevel(self, MobEffects.SPEED));
		f.addProperty("effect_slowness", effectLevel(self, MobEffects.SLOWNESS));
		f.addProperty("effect_strength", effectLevel(self, MobEffects.STRENGTH));
		f.addProperty("effect_weakness", effectLevel(self, MobEffects.WEAKNESS));
		f.addProperty("effect_resistance", effectLevel(self, MobEffects.RESISTANCE));
		f.addProperty("effect_jump_boost", effectLevel(self, MobEffects.JUMP_BOOST));
		f.addProperty("effect_regeneration", effectLevel(self, MobEffects.REGENERATION));

		// ---- self: attack readiness
		double attackScale = self.getAttackStrengthScale(0.0F);
		double attackDelay = attackSpeed > 0.0 ? 20.0 / attackSpeed : 0.0;
		f.addProperty("attack_strength", attackScale);
		f.addProperty("ticks_until_attack_ready", Math.max(0.0, (1.0 - attackScale) * attackDelay));

		// ---- self: timing counters
		f.addProperty("ticks_on_ground", ticksOnGround);
		f.addProperty("ticks_in_air", ticksInAir);
		f.addProperty("ticks_since_jump", elapsed(lastJumpTick));
		f.addProperty("ticks_since_sprint_toggle", elapsed(lastSprintToggleTick));
		f.addProperty("ticks_since_attack_swing", elapsed(lastAttackSwingTick));
		f.addProperty("ticks_since_knocked_back", elapsed(lastKnockedBackTick));

		// ---- combat history
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
		f.addProperty("last_attack_outcome", lastAttackOutcome);
		f.addProperty("ticks_since_attack_outcome", elapsed(lastAttackOutcomeTick));
		f.addProperty("last_attack_hit", lastAttackOutcome.equals("hit"));
		f.addProperty("last_attack_miss", lastAttackOutcome.equals("miss"));
		f.addProperty("last_attack_blocked", lastAttackOutcome.equals("blocked"));
		f.addProperty("last_attack_no_effect", lastAttackOutcome.equals("no_effect"));

		// ---- target: tracking state and line-of-sight memory
		Player target = trackedTarget;
		boolean tracked = target != null;
		boolean visible = tracked && targetVisible;
		f.addProperty("target_tracked", tracked);
		f.addProperty("target_valid", visible);
		f.addProperty("ticks_since_target_seen", elapsed(lastSeenTick));
		boolean hasMemory = lastKnownCenter != null;
		double[] memory = hasMemory ? toEgo(lastKnownCenter.subtract(eye), yawRad) : new double[3];
		f.addProperty("has_last_known_target", hasMemory);
		f.addProperty("last_known_target_rel_forward", memory[0]);
		f.addProperty("last_known_target_rel_right", memory[1]);
		f.addProperty("last_known_target_rel_up", memory[2]);
		f.addProperty("last_known_target_distance", hasMemory ? lastKnownCenter.distanceTo(eye) : 0.0);
		f.addProperty("last_known_target_yaw_error", hasMemory ? yawErrorTo(self, eye, lastKnownCenter) : 0.0);

		// ---- target: live features (masked to zero unless visible)
		if (visible) {
			addLiveTargetFeatures(client, self, target, f);
		} else {
			debugProjection = DebugProjection.EMPTY;
		}
		for (Map.Entry<String, Object> entry : TARGET_DEFAULTS.entrySet()) {
			if (f.has(entry.getKey())) continue;
			Object value = entry.getValue();
			if (value instanceof Boolean b) f.addProperty(entry.getKey(), b);
			else if (value instanceof Number n) f.addProperty(entry.getKey(), n);
			else f.addProperty(entry.getKey(), String.valueOf(value));
		}
		f.add("target_relative_velocity_history", velocityHistory(yawRad, visible));

		frame.add("features", f);
		frame.add("critic_only", criticOnly(self, target, yawRad));
		frame.add("action_history", actionHistory(server));
		return frame;
	}

	private static JsonObject events() {
		JsonObject e = new JsonObject();
		e.addProperty("damage_dealt", Math.max(0.0, -eventTargetHealthDelta));
		e.addProperty("damage_taken", Math.max(0.0, -eventSelfHealthDelta));
		e.addProperty("target_health_delta", eventTargetHealthDelta);
		e.addProperty("self_health_delta", eventSelfHealthDelta);
		e.addProperty("hits_dealt", eventHitsDealt);
		e.addProperty("hits_taken", eventHitsTaken);
		e.addProperty("attack_outcome", eventOutcome);
		return e;
	}

	/** Privileged information for an asymmetric critic. The policy must not consume this. */
	private static JsonObject criticOnly(Player self, Player target, double yawRad) {
		JsonObject c = new JsonObject();
		boolean tracked = self != null && target != null;
		double[] rel = tracked ? toEgo(center(target).subtract(self.getEyePosition()), yawRad) : new double[3];
		c.addProperty("target_tracked", tracked);
		c.addProperty("target_visible", tracked && targetVisible);
		c.addProperty("target_rel_forward", rel[0]);
		c.addProperty("target_rel_right", rel[1]);
		c.addProperty("target_rel_up", rel[2]);
		c.addProperty("target_hitbox_distance", tracked ? hitboxDistance : 0.0);
		c.addProperty("target_yaw_error", tracked ? yawErrorTo(self, self.getEyePosition(), center(target)) : 0.0);
		c.addProperty("target_health", tracked ? target.getHealth() : 0.0);
		c.addProperty("target_absorption", tracked ? target.getAbsorptionAmount() : 0.0);
		return c;
	}

	private static void addLiveTargetFeatures(Minecraft client, Player self, Player target, JsonObject f) {
		double yaw = Math.toRadians(self.getYRot());
		double pitch = Math.toRadians(self.getXRot());
		Vec3 eye = self.getEyePosition();
		Vec3 center = center(target);
		double[] rel = toEgo(center.subtract(eye), yaw);
		double[] relVel = toEgo(relVelSmoothed, yaw);
		double[] relAcc = toEgo(relAccel, yaw);
		Vec3 relative = target.position().subtract(self.position());

		// State and intent
		f.addProperty("target_health", target.getHealth());
		f.addProperty("target_absorption", target.getAbsorptionAmount());
		f.addProperty("target_armor", target.getArmorValue());
		f.addProperty("target_height", target.getBbHeight());
		f.addProperty("target_hurt_time", target.hurtTime);
		f.addProperty("target_invulnerable_ticks", target.invulnerableTime);
		f.addProperty("target_on_ground", target.onGround());
		f.addProperty("target_sprinting", target.isSprinting());
		f.addProperty("target_sneaking", target.isCrouching());
		f.addProperty("target_swinging", target.swinging);
		f.addProperty("target_swing_progress", target.getAttackAnim(0.0F));
		f.addProperty("target_blocking", target.isBlocking());
		f.addProperty("target_using_item", target.isUsingItem());
		int category = itemCategory(target.getMainHandItem());
		f.addProperty("target_held_item", ITEM_CATEGORIES[category]);
		f.addProperty("target_held_item_id", category);

		// Distance, reach and attack feasibility
		f.addProperty("target_distance", self.distanceTo(target));
		f.addProperty("target_horizontal_distance", Math.sqrt(relative.x * relative.x + relative.z * relative.z));
		f.addProperty("target_hitbox_distance", hitboxDistance);
		f.addProperty("target_in_reach", hitboxDistance <= self.entityInteractionRange());
		f.addProperty("closing_speed", closingSpeed);
		boolean canCrit = self.getAttackStrengthScale(0.0F) > 0.9F
				&& ((PlayerAttackAccess) (Object) self).minecraftai$canCriticalNow(target);
		f.addProperty("can_crit_now", canCrit);

		// Egocentric geometry. Yaw error aims at the target's center; pitch is intentionally not targeted.
		f.addProperty("target_rel_forward", rel[0]);
		f.addProperty("target_rel_right", rel[1]);
		f.addProperty("target_rel_up", rel[2]);
		f.addProperty("target_yaw_error", yawErrorTo(self, eye, center));
		float targetYaw = Mth.wrapDegrees(target.getYRot());
		f.addProperty("target_yaw", targetYaw);
		f.addProperty("target_yaw_sin", Math.sin(Math.toRadians(targetYaw)));
		f.addProperty("target_yaw_cos", Math.cos(Math.toRadians(targetYaw)));
		f.addProperty("target_pitch", target.getXRot());
		Vec3 targetToSelf = self.position().subtract(target.position());
		double facingError = targetToSelf.x * targetToSelf.x + targetToSelf.z * targetToSelf.z < 1.0E-8 ? 0.0
				: Mth.wrapDegrees(Math.toDegrees(Math.atan2(-targetToSelf.x, targetToSelf.z)) - target.getYHeadRot());
		f.addProperty("target_facing_error", facingError);
		f.addProperty("target_facing_cos", Math.cos(Math.toRadians(facingError)));

		// Relative motion in the player's yaw frame
		f.addProperty("target_relative_velocity_forward", relVel[0]);
		f.addProperty("target_relative_velocity_right", relVel[1]);
		f.addProperty("target_relative_velocity_up", relVel[2]);
		f.addProperty("target_relative_horizontal_speed",
				Math.sqrt(relVelSmoothed.x * relVelSmoothed.x + relVelSmoothed.z * relVelSmoothed.z));
		f.addProperty("target_relative_acceleration_forward", relAcc[0]);
		f.addProperty("target_relative_acceleration_right", relAcc[1]);
		f.addProperty("target_relative_acceleration_up", relAcc[2]);

		// Screen projection of the clipped bounding box and the center point
		Vec3 forward = new Vec3(-Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
		// Minecraft yaw 0 looks toward +Z, so screen-right points toward -X.
		Vec3 right = new Vec3(-Math.cos(yaw), 0.0, -Math.sin(yaw));
		Vec3 up = right.cross(forward).normalize();
		double tanV = Math.tan(Math.toRadians(client.options.fov().get()) / 2.0);
		double aspect = (double) client.getWindow().getWidth() / Math.max(1, client.getWindow().getHeight());
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
		List<ScreenPoint> projected = clipBoxToView(corners, tanH, tanV, 0.05);
		boolean anyInFront = Arrays.stream(corners).anyMatch(point -> point.depth() > 0.05);
		if (eye.x >= box.minX && eye.x <= box.maxX && eye.y >= box.minY && eye.y <= box.maxY
				&& eye.z >= box.minZ && eye.z <= box.maxZ) {
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
		f.addProperty("target_bbox_projection_valid", anyInFront);
		f.addProperty("target_on_screen", boxOnScreen);

		Vec3 centerDelta = center.subtract(eye);
		double centerDepth = centerDelta.dot(forward);
		boolean centerValid = centerDepth > 0.05;
		double centerX = centerValid ? 0.5 + 0.5 * centerDelta.dot(right) / (centerDepth * tanH) : 0.0;
		double centerY = centerValid ? 0.5 - 0.5 * centerDelta.dot(up) / (centerDepth * tanV) : 0.0;
		f.addProperty("target_center_projection_valid", centerValid);
		// Offsets from screen center, so 0 means centered and an invalid projection is neutral.
		f.addProperty("target_dx", centerValid ? centerX - 0.5 : 0.0);
		f.addProperty("target_dy", centerValid ? centerY - 0.5 : 0.0);
		f.addProperty("target_bbox_left", boxOnScreen ? clamp01(minX) : 0.0);
		f.addProperty("target_bbox_top", boxOnScreen ? clamp01(minY) : 0.0);
		f.addProperty("target_bbox_right", boxOnScreen ? clamp01(maxX) : 0.0);
		f.addProperty("target_bbox_bottom", boxOnScreen ? clamp01(maxY) : 0.0);
		f.addProperty("target_bbox_width", boxOnScreen ? clamp01(maxX) - clamp01(minX) : 0.0);
		f.addProperty("target_bbox_height", boxOnScreen ? clamp01(maxY) - clamp01(minY) : 0.0);
		boolean crosshairOnTarget = client.hitResult instanceof net.minecraft.world.phys.EntityHitResult hit
				&& hit.getEntity() == target;
		f.addProperty("crosshair_on_target", crosshairOnTarget);

		debugProjection = new DebugProjection(anyInFront, boxOnScreen,
				boxOnScreen ? clamp01(minX) : 0.0, boxOnScreen ? clamp01(minY) : 0.0,
				boxOnScreen ? clamp01(maxX) : 0.0, boxOnScreen ? clamp01(maxY) : 0.0,
				centerValid, centerX, centerY, true, crosshairOnTarget, self.distanceTo(target),
				hitboxDistance, yawErrorTo(self, eye, center), closingSpeed, relVel[0], relVel[1], relVel[2],
				target.onGround(), target.isSprinting(), target.isCrouching());
	}

	/** Keys that must exist in every frame; zero or false when the target is not visible. */
	private static Map<String, Object> targetDefaults() {
		Map<String, Object> d = new LinkedHashMap<>();
		for (String key : new String[] {
				"target_health", "target_absorption", "target_armor", "target_height", "target_hurt_time",
				"target_invulnerable_ticks", "target_swing_progress", "target_held_item_id",
				"target_distance", "target_horizontal_distance", "target_hitbox_distance", "closing_speed",
				"target_rel_forward", "target_rel_right", "target_rel_up",
				"target_yaw_error", "target_yaw", "target_yaw_sin", "target_yaw_cos", "target_pitch",
				"target_facing_error", "target_facing_cos",
				"target_relative_velocity_forward", "target_relative_velocity_right", "target_relative_velocity_up",
				"target_relative_horizontal_speed",
				"target_relative_acceleration_forward", "target_relative_acceleration_right",
				"target_relative_acceleration_up",
				"target_dx", "target_dy", "target_bbox_left", "target_bbox_top", "target_bbox_right",
				"target_bbox_bottom", "target_bbox_width", "target_bbox_height"}) {
			d.put(key, 0.0);
		}
		for (String key : new String[] {
				"target_on_ground", "target_sprinting", "target_sneaking", "target_swinging", "target_blocking",
				"target_using_item", "target_in_reach", "can_crit_now", "target_on_screen",
				"target_bbox_projection_valid", "target_center_projection_valid", "crosshair_on_target"}) {
			d.put(key, false);
		}
		d.put("target_held_item", "none");
		return d;
	}

	private static JsonArray velocityHistory(double yawRad, boolean live) {
		JsonArray history = new JsonArray();
		int pad = VELOCITY_HISTORY - (live ? relVelHistory.size() : 0);
		for (int i = 0; i < pad; i++) history.add(row(0.0, 0.0, 0.0));
		if (live) {
			for (Vec3 v : relVelHistory) {
				double[] e = toEgo(v, yawRad);
				history.add(row(e[0], e[1], e[2]));
			}
		}
		return history;
	}

	private static JsonArray row(double a, double b, double c) {
		JsonArray row = new JsonArray();
		row.add(a);
		row.add(b);
		row.add(c);
		return row;
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

	// =====================================================================
	// Helpers
	// =====================================================================

	private static Vec3 center(Player p) {
		return new Vec3(p.getX(), p.getY() + p.getBbHeight() * 0.5, p.getZ());
	}

	/** Player-yaw frame: forward and right are horizontal, up is world-up. */
	private static double[] toEgo(Vec3 v, double yawRad) {
		double sin = Math.sin(yawRad);
		double cos = Math.cos(yawRad);
		return new double[] {v.x * -sin + v.z * cos, v.x * -cos + v.z * -sin, v.y};
	}

	/** Signed degrees to turn to face {@code to}; positive means turn right. Wrapped to ±180. */
	private static double yawErrorTo(Player self, Vec3 from, Vec3 to) {
		double dx = to.x - from.x;
		double dz = to.z - from.z;
		if (dx * dx + dz * dz < 1.0E-8) return 0.0;
		return Mth.wrapDegrees(Math.toDegrees(Math.atan2(-dx, dz)) - self.getYRot());
	}

	/** Distance from the eye to the nearest point of the hitbox, the vanilla reach metric. */
	private static double hitboxDistance(Vec3 eye, AABB box) {
		double dx = Math.max(Math.max(box.minX - eye.x, 0.0), eye.x - box.maxX);
		double dy = Math.max(Math.max(box.minY - eye.y, 0.0), eye.y - box.maxY);
		double dz = Math.max(Math.max(box.minZ - eye.z, 0.0), eye.z - box.maxZ);
		return Math.sqrt(dx * dx + dy * dy + dz * dz);
	}

	private static int effectLevel(LivingEntity entity, Holder<MobEffect> effect) {
		MobEffectInstance instance = entity.getEffect(effect);
		return instance == null ? 0 : instance.getAmplifier() + 1;
	}

	private static int itemCategory(ItemStack s) {
		if (s.isEmpty()) return 0;
		if (s.is(ItemTags.SWORDS)) return 1;
		if (s.is(ItemTags.AXES)) return 2;
		if (s.is(Items.BOW) || s.is(Items.CROSSBOW)) return 3;
		if (s.is(Items.TRIDENT) || s.is(Items.MACE)) return 4;
		if (s.is(Items.ENDER_PEARL)) return 5;
		if (s.is(Items.GOLDEN_APPLE) || s.is(Items.ENCHANTED_GOLDEN_APPLE)) return 6;
		if (s.is(Items.POTION) || s.is(Items.SPLASH_POTION) || s.is(Items.LINGERING_POTION)) return 7;
		if (s.is(Items.SHIELD)) return 8;
		if (s.is(Items.FISHING_ROD)) return 9;
		if (s.getItem() instanceof BlockItem) return 10;
		return 11;
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

	// =====================================================================
	// Debug overlay
	// =====================================================================

	static void renderDebugOverlay(net.minecraft.client.gui.GuiGraphics graphics, int width, int height, int verticalOffset) {
		Minecraft client = Minecraft.getInstance();
		DebugProjection projection = client.player == null ? DebugProjection.EMPTY : debugProjection;
		int textColor = 0xFFFFFFFF;
		int y = 39 + verticalOffset;
		graphics.fill(4, 34 + verticalOffset, Math.max(8, Math.min(width - 4, 760)),
				Math.min(height - 4, 175 + verticalOffset), 0x90000000);
		String targetState;
		if (projection.targetValid) {
			targetState = String.format(java.util.Locale.ROOT,
					"Target: %.2fm (hitbox %.2fm) | screen=%s | crosshair hit=%s | box=%s",
					projection.targetDistance, projection.hitboxDistance, projection.onScreen ? "yes" : "no",
					projection.crosshairOnTarget ? "YES" : "no", projection.boxValid ? "valid" : "behind camera");
		} else if (trackedTarget != null) {
			targetState = "Target: tracked, not visible (seen " + elapsed(lastSeenTick) + " ticks ago)";
		} else {
			targetState = "Target: none (alive, non-allied player required)";
		}
		graphics.drawString(client.font, "MinecraftAI diagnostics — F8 hides this panel", 8, y, 0xFF66FFFF, true); y += 12;
		graphics.drawString(client.font, targetState, 8, y, textColor, true); y += 12;
		String center = projection.centerValid
				? String.format(java.util.Locale.ROOT,
						"Center screen UV: x=%.3f y=%.3f | yaw error to center: %+.1f° | closing %+.3f/tick",
						projection.centerX, projection.centerY, projection.yawError, projection.closingSpeed)
				: "Center screen: invalid / behind camera";
		graphics.drawString(client.font, center, 8, y, 0xFF55FFFF, true); y += 12;

		if (projection.targetValid) {
			String relativeVelocity = String.format(java.util.Locale.ROOT,
					"Target rel velocity fwd/right/up: %.2f / %.2f / %.2f | ground=%s sprint=%s sneak=%s",
					projection.relVelForward, projection.relVelRight, projection.relVelUp,
					projection.targetOnGround, projection.targetSprinting, projection.targetSneaking);
			graphics.drawString(client.font, relativeVelocity, 8, y, textColor, true); y += 12;
		}

		if (client.player != null) {
			Player self = client.player;
			graphics.drawString(client.font, String.format(java.util.Locale.ROOT,
					"Camera yaw=%.1f pitch=%.1f | speed=%.2f | ground=%s sprint=%s sneak=%s",
					self.getYRot(), self.getXRot(), Math.sqrt(self.getDeltaMovement().x * self.getDeltaMovement().x
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
				+ " (" + outgoing + ") | incoming: " + incoming + " | last swing: " + lastAttackOutcome, 8, y,
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
	}

	private static int elapsed(long then) { return then < 0 ? -1 : (int) Math.min(1_000_000, tick - then); }
	static int currentCombo() { return hitCombo; }
	static int currentKnockbackCombo() { return knockbackCombo; }
	private static double clamp01(double value) { return Math.max(0.0, Math.min(1.0, value)); }

	private record DebugProjection(boolean boxValid, boolean onScreen, double left, double top, double right,
			double bottom, boolean centerValid, double centerX, double centerY, boolean targetValid,
			boolean crosshairOnTarget, double targetDistance, double hitboxDistance, double yawError,
			double closingSpeed, double relVelForward, double relVelRight, double relVelUp,
			boolean targetOnGround, boolean targetSprinting, boolean targetSneaking) {
		private static final DebugProjection EMPTY = new DebugProjection(false, false, 0, 0, 0, 0,
				false, 0, 0, false, false, 0, 0, 0, 0, 0, 0, 0, false, false, false);
	}
	private record AttackAttempt(long tick, boolean critical, boolean knockback, boolean sweep,
			boolean targetBlocking) { }
}