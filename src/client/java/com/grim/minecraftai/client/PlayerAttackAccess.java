package com.grim.minecraftai.client;

import net.minecraft.world.entity.Entity;

/** Implemented onto Player by PlayerAttackMixin so observations reuse vanilla's critical-hit check. */
public interface PlayerAttackAccess {
	boolean minecraftai$canCriticalNow(Entity target);
}