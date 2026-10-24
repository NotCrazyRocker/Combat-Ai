package com.grim.swordpvpai.client.observation;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.VoxelShape;

import java.util.Comparator;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

/** Captures the client-side Minecraft world state once at the end of every client tick. */
public final class ObservationFeature {
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().serializeNulls().create();
    private static final AtomicLong SEQUENCE = new AtomicLong();
    private static volatile long latestIntegratedServerTick = -1L;
    private static final double TARGET_SEARCH_RANGE = 32.0;
    private static final int GEOMETRY_RADIUS = 3;
    private static final int GEOMETRY_Y_BELOW = 1;
    private static final int GEOMETRY_Y_ABOVE = 2;

    private static UUID previousPlayerId;
    private static double previousPlayerTotalHealth;
    private static UUID previousTargetId;
    private static double previousTargetTotalHealth;
    private static Vec3 previousTargetVelocity;

    private ObservationFeature() {}

    public static void register() {
        ClientTickEvents.END_CLIENT_TICK.register(ObservationFeature::captureTick);
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            if (Minecraft.getInstance().getSingleplayerServer() == server) {
                latestIntegratedServerTick = server.getTickCount();
            }
        });
    }

    private static void captureTick(Minecraft client) {
        if (!client.isLocalServer()) {
            latestIntegratedServerTick = -1L;
        }
        if (client.player == null || client.level == null) {
            resetTracking();
            return;
        }

        Player player = client.player;
        Level level = client.level;
        Player opponent = findNearestOpponent(player, client);
        long sequence = SEQUENCE.incrementAndGet();

        double playerTotalHealth = totalHealth(player);
        double playerDamageTaken = player.getUUID().equals(previousPlayerId)
                ? Math.max(0.0, previousPlayerTotalHealth - playerTotalHealth) : 0.0;
        previousPlayerId = player.getUUID();
        previousPlayerTotalHealth = playerTotalHealth;

        double opponentDamage = 0.0;
        Vec3 opponentVelocityChange = Vec3.ZERO;
        if (opponent != null) {
            double total = totalHealth(opponent);
            if (opponent.getUUID().equals(previousTargetId)) {
                opponentDamage = Math.max(0.0, previousTargetTotalHealth - total);
                if (previousTargetVelocity != null) {
                    opponentVelocityChange = opponent.getDeltaMovement().subtract(previousTargetVelocity);
                }
            }
            previousTargetId = opponent.getUUID();
            previousTargetTotalHealth = total;
            previousTargetVelocity = opponent.getDeltaMovement();
        } else {
            previousTargetId = null;
            previousTargetTotalHealth = 0.0;
            previousTargetVelocity = null;
        }

        JsonObject root = new JsonObject();
        root.addProperty("schema_version", "1.0.0");
        root.addProperty("sequence", sequence);
        root.addProperty("client_tick_sequence", sequence);
        long serverTick = client.isLocalServer() ? latestIntegratedServerTick : -1L;
        if (serverTick >= 0L) {
            root.addProperty("server_tick", serverTick);
        } else {
            root.add("server_tick", JsonNull.INSTANCE);
        }
        root.addProperty("server_tick_available", serverTick >= 0L);
        root.addProperty("captured_at_epoch_ms", System.currentTimeMillis());
        root.addProperty("level_game_time", level.getGameTime());
        root.addProperty("tick_source", "fabric_client_end_tick");
        root.addProperty("authority", "client_observed_world_state");
        root.add("provenance", provenance());
        root.add("player", entityState(player, playerDamageTaken));
        root.add("opponent", opponent == null ? JsonNull.INSTANCE : entityState(opponent, opponentDamage));
        root.add("derived", derived(player, opponent, client));
        root.add("events", events(player, opponent, playerDamageTaken, opponentDamage,
                opponentVelocityChange));
        root.add("world", worldState(level, player));
        JsonObject transport = new JsonObject();
        transport.addProperty("queued_observations_dropped", ObservationBridge.droppedCount());
        root.add("transport", transport);

        ObservationBridge.send(GSON.toJson(root));
    }

    private static JsonObject provenance() {
        JsonObject source = new JsonObject();
        source.addProperty("entity_values", "read from Minecraft client entity objects; remote-world values are server-replicated snapshots");
        source.addProperty("level_game_time", "client world game-time field; not guaranteed to equal an authoritative server tick counter");
        source.addProperty("server_tick", "integrated-server tick count captured from Fabric's end-server-tick event; unavailable to remote clients");
        source.addProperty("derived_features", "calculated locally from captured Minecraft values");
        source.addProperty("damage_deltas", "estimated from consecutive observed health plus absorption values; source attribution is not guaranteed");
        source.addProperty("attack_type", "heuristic estimate only; exact critical/sweep/knockback classification needs server-side hit instrumentation");
        source.addProperty("collision_geometry", "collision shapes and fluid state read directly from loaded client-world blocks");
        return source;
    }

    private static Player findNearestOpponent(Player self, Minecraft client) {
        if (client.level == null) return null;
        double maxDistanceSqr = TARGET_SEARCH_RANGE * TARGET_SEARCH_RANGE;
        return client.level.players().stream()
                .filter(candidate -> candidate != self && candidate.isAlive() && !candidate.isSpectator())
                .filter(candidate -> candidate.distanceToSqr(self) <= maxDistanceSqr)
                .min(Comparator.comparingDouble(candidate -> candidate.distanceToSqr(self)))
                .orElse(null);
    }

    private static JsonObject entityState(LivingEntity entity, double damageThisTick) {
        JsonObject out = new JsonObject();
        out.addProperty("uuid", entity.getUUID().toString());
        out.addProperty("name", entity.getName().getString());
        out.addProperty("entity_type", BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());
        out.add("position", vec(entity.position()));
        out.add("velocity", vec(entity.getDeltaMovement()));
        out.addProperty("yaw_degrees", entity.getYRot());
        out.addProperty("pitch_degrees", entity.getXRot());
        out.addProperty("health", round(entity.getHealth()));
        out.addProperty("max_health", round(entity.getMaxHealth()));
        out.addProperty("absorption", round(entity.getAbsorptionAmount()));
        out.addProperty("armor_points", entity.getArmorValue());
        out.addProperty("hurt_time", entity.hurtTime);
        out.addProperty("damage_observed_this_tick", round(damageThisTick));
        out.addProperty("attack_cooldown_scale", entity instanceof Player p ? round(p.getAttackStrengthScale(0.0F)) : -1.0);
        out.add("equipment", equipment(entity));
        out.add("status_effects", effects(entity));

        JsonObject movement = new JsonObject();
        movement.addProperty("sprinting", entity.isSprinting());
        movement.addProperty("sneaking", entity.isCrouching());
        movement.addProperty("on_ground", entity.onGround());
        movement.addProperty("fall_distance", round(entity.fallDistance));
        movement.addProperty("in_water", entity.isInWater());
        movement.addProperty("in_lava", entity.isInLava());
        movement.addProperty("swimming", entity.isSwimming());
        movement.addProperty("climbing", entity.onClimbable());
        movement.addProperty("passenger", entity.isPassenger());
        movement.addProperty("using_item", entity.isUsingItem());
        movement.addProperty("fall_flying", entity.isFallFlying());
        if (entity instanceof Player p) {
            movement.addProperty("spectator", p.isSpectator());
            movement.addProperty("creative", p.isCreative());
            movement.addProperty("may_fly", p.getAbilities().mayfly);
        }
        out.add("movement", movement);
        return out;
    }

    private static JsonObject equipment(LivingEntity entity) {
        JsonObject out = new JsonObject();
        addEquipmentSlot(out, "mainhand", entity.getItemBySlot(EquipmentSlot.MAINHAND));
        addEquipmentSlot(out, "offhand", entity.getItemBySlot(EquipmentSlot.OFFHAND));
        addEquipmentSlot(out, "head", entity.getItemBySlot(EquipmentSlot.HEAD));
        addEquipmentSlot(out, "chest", entity.getItemBySlot(EquipmentSlot.CHEST));
        addEquipmentSlot(out, "legs", entity.getItemBySlot(EquipmentSlot.LEGS));
        addEquipmentSlot(out, "feet", entity.getItemBySlot(EquipmentSlot.FEET));
        return out;
    }

    private static void addEquipmentSlot(JsonObject out, String slot, ItemStack stack) {
        JsonObject item = new JsonObject();
        item.addProperty("item_id", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
        item.addProperty("count", stack.getCount());
        item.addProperty("empty", stack.isEmpty());
        item.addProperty("damage", stack.getDamageValue());
        item.addProperty("max_damage", stack.getMaxDamage());
        out.add(slot, item);
    }

    private static JsonArray effects(LivingEntity entity) {
        JsonArray out = new JsonArray();
        entity.getActiveEffects().forEach(effect -> {
            JsonObject item = new JsonObject();
            item.addProperty("effect_id", BuiltInRegistries.MOB_EFFECT.getKey(effect.getEffect().value()).toString());
            item.addProperty("amplifier", effect.getAmplifier());
            item.addProperty("duration_ticks", effect.getDuration());
            item.addProperty("ambient", effect.isAmbient());
            item.addProperty("visible", effect.isVisible());
            item.addProperty("show_icon", effect.showIcon());
            out.add(item);
        });
        return out;
    }

    private static JsonObject derived(Player player, Player opponent, Minecraft client) {
        JsonObject out = new JsonObject();
        if (opponent == null || client.level == null) {
            out.add("target_relative_position", JsonNull.INSTANCE);
            out.add("target_relative_velocity", JsonNull.INSTANCE);
            out.add("aim", JsonNull.INSTANCE);
            out.add("reachability", JsonNull.INSTANCE);
            return out;
        }

        Vec3 relative = opponent.position().subtract(player.position());
        Vec3 relativeVelocity = opponent.getDeltaMovement().subtract(player.getDeltaMovement());
        Vec3 eyeDelta = opponent.getEyePosition().subtract(player.getEyePosition());
        double distance = relative.length();
        double horizontalDistance = Math.sqrt(relative.x * relative.x + relative.z * relative.z);
        double desiredYaw = Math.toDegrees(Math.atan2(-eyeDelta.x, eyeDelta.z));
        double desiredPitch = -Math.toDegrees(Math.atan2(eyeDelta.y,
                Math.sqrt(eyeDelta.x * eyeDelta.x + eyeDelta.z * eyeDelta.z)));
        double yawError = wrapDegrees(desiredYaw - player.getYRot());
        double pitchError = wrapDegrees(desiredPitch - player.getXRot());
        boolean lineOfSight = client.level.clip(new ClipContext(player.getEyePosition(), opponent.getEyePosition(),
                ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, player)).getType() == HitResult.Type.MISS;

        out.add("target_relative_position", vec(relative));
        out.add("target_relative_velocity", vec(relativeVelocity));
        out.addProperty("distance_blocks", round(distance));
        out.addProperty("horizontal_distance_blocks", round(horizontalDistance));
        out.addProperty("eye_distance_blocks", round(eyeDelta.length()));
        JsonObject aim = new JsonObject();
        aim.addProperty("desired_yaw_degrees", round(desiredYaw));
        aim.addProperty("desired_pitch_degrees", round(desiredPitch));
        aim.addProperty("yaw_error_degrees", round(yawError));
        aim.addProperty("pitch_error_degrees", round(pitchError));
        out.add("aim", aim);
        JsonObject reach = new JsonObject();
        reach.addProperty("line_of_sight", lineOfSight);
        reach.addProperty("within_3_block_estimate", distance <= 3.0);
        reach.addProperty("estimated_reachable", distance <= 3.0 && lineOfSight);
        reach.addProperty("authority", "heuristic; actual attack reach and server validation may differ");
        out.add("reachability", reach);
        return out;
    }

    private static JsonObject events(Player player, Player opponent, double damageTaken,
                                     double opponentDamage, Vec3 targetVelocityChange) {
        JsonObject out = new JsonObject();
        out.addProperty("damage_received_this_tick", round(damageTaken));
        out.addProperty("opponent_damage_observed_this_tick", round(opponentDamage));
        out.addProperty("opponent_damage_is_attributed_to_us", false);
        out.add("observed_hit", JsonNull.INSTANCE);

        if (opponent != null && opponentDamage > 0.0) {
            JsonObject hit = new JsonObject();
            hit.addProperty("detected", true);
            hit.addProperty("source", "opponent_health_or_absorption_decrease");
            hit.addProperty("damage", round(opponentDamage));
            String type = "normal_hit_candidate";
            String reason = "health delta observed; attack attribution and subtype are uncertain";
            boolean possibleCritical = !player.onGround() && player.fallDistance > 0.0F
                    && !player.isSprinting() && !player.isInWater() && !player.onClimbable()
                    && !player.isPassenger();
            boolean sword = BuiltInRegistries.ITEM.getKey(player.getMainHandItem().getItem())
                    .getPath().endsWith("_sword");
            double horizontalSpeed = Math.sqrt(player.getDeltaMovement().x * player.getDeltaMovement().x
                    + player.getDeltaMovement().z * player.getDeltaMovement().z);
            if (possibleCritical) {
                type = "critical_hit_candidate";
                reason = "player movement meets some critical-hit conditions; not server-confirmed";
            } else if (sword && player.onGround() && !player.isSprinting()
                    && player.getAttackStrengthScale(0.0F) >= 0.9F && horizontalSpeed > 0.01) {
                type = "sweep_hit_candidate";
                reason = "movement and cooldown resemble a sweep; not server-confirmed";
            } else if ((targetVelocityChange.x * targetVelocityChange.x + targetVelocityChange.z * targetVelocityChange.z) > 0.04) {
                type = "knockback_hit_candidate";
                reason = "target velocity changed around the observed health delta; attribution uncertain";
            }
            hit.addProperty("type_estimate", type);
            hit.addProperty("confidence", "low");
            hit.addProperty("reason", reason);
            out.add("observed_hit", hit);
        }
        return out;
    }

    private static JsonObject worldState(Level level, Player player) {
        JsonObject out = new JsonObject();
        BlockPos origin = BlockPos.containing(player.getX(), player.getY(), player.getZ());
        BlockPos below = BlockPos.containing(player.getX(), player.getY() - 0.05, player.getZ());
        BlockState ground = level.getBlockState(below);
        JsonObject groundObject = new JsonObject();
        groundObject.addProperty("block_id", BuiltInRegistries.BLOCK.getKey(ground.getBlock()).toString());
        groundObject.addProperty("has_collision", !ground.getCollisionShape(level, below).isEmpty());
        groundObject.addProperty("water", ground.getFluidState().is(FluidTags.WATER));
        groundObject.addProperty("lava", ground.getFluidState().is(FluidTags.LAVA));
        out.add("ground_below_player", groundObject);
        out.add("nearby_collision_geometry", nearbyGeometry(level, origin));
        return out;
    }

    private static JsonArray nearbyGeometry(Level level, BlockPos origin) {
        JsonArray blocks = new JsonArray();
        for (int dx = -GEOMETRY_RADIUS; dx <= GEOMETRY_RADIUS; dx++) {
            for (int dy = -GEOMETRY_Y_BELOW; dy <= GEOMETRY_Y_ABOVE; dy++) {
                for (int dz = -GEOMETRY_RADIUS; dz <= GEOMETRY_RADIUS; dz++) {
                    BlockPos pos = origin.offset(dx, dy, dz);
                    BlockState state = level.getBlockState(pos);
                    VoxelShape shape = state.getCollisionShape(level, pos);
                    boolean water = state.getFluidState().is(FluidTags.WATER);
                    boolean lava = state.getFluidState().is(FluidTags.LAVA);
                    boolean hazard = state.is(net.minecraft.world.level.block.Blocks.FIRE)
                            || state.is(net.minecraft.world.level.block.Blocks.SOUL_FIRE)
                            || state.is(net.minecraft.world.level.block.Blocks.CACTUS)
                            || state.is(net.minecraft.world.level.block.Blocks.MAGMA_BLOCK)
                            || state.is(net.minecraft.world.level.block.Blocks.POWDER_SNOW)
                            || state.is(net.minecraft.world.level.block.Blocks.SWEET_BERRY_BUSH)
                            || state.is(net.minecraft.world.level.block.Blocks.CAMPFIRE)
                            || state.is(net.minecraft.world.level.block.Blocks.SOUL_CAMPFIRE)
                            || state.is(net.minecraft.world.level.block.Blocks.POINTED_DRIPSTONE);
                    if (state.isAir() && shape.isEmpty() && !water && !lava && !hazard) continue;

                    JsonObject block = new JsonObject();
                    block.addProperty("x_offset", dx);
                    block.addProperty("y_offset", dy);
                    block.addProperty("z_offset", dz);
                    block.addProperty("block_id", BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());
                    block.addProperty("has_collision", !shape.isEmpty());
                    block.addProperty("water", water);
                    block.addProperty("lava", lava);
                    block.addProperty("hazard_candidate", hazard || lava);
                    JsonArray boxes = new JsonArray();
                    for (AABB box : shape.toAabbs()) {
                        JsonObject bounds = new JsonObject();
                        // Collision-shape AABBs are expressed in the shape's block-local coordinate space.
                        bounds.addProperty("min_x", round(box.minX));
                        bounds.addProperty("min_y", round(box.minY));
                        bounds.addProperty("min_z", round(box.minZ));
                        bounds.addProperty("max_x", round(box.maxX));
                        bounds.addProperty("max_y", round(box.maxY));
                        bounds.addProperty("max_z", round(box.maxZ));
                        boxes.add(bounds);
                    }
                    block.add("collision_boxes", boxes);
                    blocks.add(block);
                }
            }
        }
        return blocks;
    }

    private static JsonObject vec(Vec3 vector) {
        JsonObject out = new JsonObject();
        out.addProperty("x", round(vector.x));
        out.addProperty("y", round(vector.y));
        out.addProperty("z", round(vector.z));
        return out;
    }

    private static double totalHealth(LivingEntity entity) {
        return entity.getHealth() + entity.getAbsorptionAmount();
    }

    private static double wrapDegrees(double degrees) {
        double wrapped = degrees % 360.0;
        if (wrapped >= 180.0) wrapped -= 360.0;
        if (wrapped < -180.0) wrapped += 360.0;
        return wrapped;
    }

    private static double round(double value) {
        return Math.round(value * 10000.0) / 10000.0;
    }

    private static void resetTracking() {
        previousPlayerId = null;
        previousPlayerTotalHealth = 0.0;
        previousTargetId = null;
        previousTargetTotalHealth = 0.0;
        previousTargetVelocity = null;
    }
}
