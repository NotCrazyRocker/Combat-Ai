package com.grim.minecraftai.client.mixin;

import com.grim.minecraftai.client.MinecraftAIClient;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.enchantment.ItemEnchantments;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.gen.Invoker;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Player.class)
abstract class PlayerAttackMixin {
	@Shadow public abstract float getAttackStrengthScale(float adjustTicks);

	@Invoker("canCriticalAttack")
	protected abstract boolean minecraftai$canCriticalAttack(Entity target);

	@Invoker("isSweepAttack")
	protected abstract boolean minecraftai$isSweepAttack(boolean fullyCharged, boolean critical, boolean sprintKnockback);

	@Inject(method = "attack", at = @At("HEAD"))
	private void minecraftai$recordAttackAttempt(Entity target, CallbackInfo ci) {
		Player attacker = (Player) (Object) this;
		if (attacker != net.minecraft.client.Minecraft.getInstance().player || !(target instanceof Player opponent)) return;
		boolean fullyCharged = getAttackStrengthScale(0.5F) > 0.9F;
		boolean sprintKnockback = attacker.isSprinting() && fullyCharged;
		boolean critical = fullyCharged && minecraftai$canCriticalAttack(target);
		boolean sweep = minecraftai$isSweepAttack(fullyCharged, critical, sprintKnockback);
		boolean enchantedKnockback = hasKnockbackEnchantment(attacker.getMainHandItem().getEnchantments());
		MinecraftAIClient.recordAttackAttempt(
				attacker, opponent, critical, enchantedKnockback || sprintKnockback, sweep
		);
	}

	private static boolean hasKnockbackEnchantment(ItemEnchantments enchantments) {
		return enchantments.keySet().stream().anyMatch(holder -> holder.unwrapKey()
				.map(key -> key.identifier().toString().equals("minecraft:knockback"))
				.orElse(false));
	}
}
