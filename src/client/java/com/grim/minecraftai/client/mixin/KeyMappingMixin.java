package com.grim.minecraftai.client.mixin;

import com.grim.minecraftai.client.ControlCommand;
import com.grim.minecraftai.client.MinecraftAIClient;
import com.grim.minecraftai.client.RemoteInputDiagnostics;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(KeyMapping.class)
abstract class KeyMappingMixin {
	@Unique
	private long minecraftAI$lastAttackClickNanos = Long.MIN_VALUE;
	@Unique
	private long minecraftAI$lastUseClickNanos = Long.MIN_VALUE;

	@Inject(method = "isDown", at = @At("HEAD"), cancellable = true)
	private void minecraftAI$useRemoteControlsWhenEnabled(CallbackInfoReturnable<Boolean> callback) {
		if (!MinecraftAIClient.isAIControlEnabled()) {
			return;
		}

		Minecraft client = Minecraft.getInstance();
		if (client.player == null) {
			return;
		}

		KeyMapping mapping = (KeyMapping) (Object) this;
		ControlCommand command = MinecraftAIClient.getCurrentControlCommand();
		boolean pressed;
		if (mapping == client.options.keyUp) {
			pressed = command != null && command.forward();
		} else if (mapping == client.options.keyDown) {
			pressed = command != null && command.backward();
		} else if (mapping == client.options.keyLeft) {
			pressed = command != null && command.left();
		} else if (mapping == client.options.keyRight) {
			pressed = command != null && command.right();
		} else if (mapping == client.options.keyJump) {
			pressed = command != null && command.jump();
		} else if (mapping == client.options.keyShift) {
			pressed = command != null && command.sneak();
		} else if (mapping == client.options.keySprint) {
			pressed = command != null && command.sprint();
		} else if (mapping == client.options.keyAttack) {
			pressed = command != null && command.attack();
		} else if (mapping == client.options.keyUse) {
			pressed = command != null && command.use();
		} else {
			return;
		}

		callback.setReturnValue(pressed);
	}

	@Inject(method = "consumeClick", at = @At("HEAD"), cancellable = true)
	private void minecraftAI$consumeRemoteClicks(CallbackInfoReturnable<Boolean> callback) {
		if (!MinecraftAIClient.isAIControlEnabled()) {
			return;
		}

		Minecraft client = Minecraft.getInstance();
		if (client.player == null) {
			return;
		}

		KeyMapping mapping = (KeyMapping) (Object) this;
		boolean attack = mapping == client.options.keyAttack;
		boolean use = mapping == client.options.keyUse;
		if (!attack && !use) {
			return;
		}

		ControlCommand command = MinecraftAIClient.getCurrentControlCommand();
		if (command == null) {
			callback.setReturnValue(false);
			return;
		}

		long receivedAtNanos = command.receivedAtNanos();
		if (attack) {
			boolean click = command.attack() && receivedAtNanos != minecraftAI$lastAttackClickNanos;
			minecraftAI$lastAttackClickNanos = receivedAtNanos;
			if (click) RemoteInputDiagnostics.recordClick(true);
			callback.setReturnValue(click);
		} else {
			boolean click = command.use() && receivedAtNanos != minecraftAI$lastUseClickNanos;
			minecraftAI$lastUseClickNanos = receivedAtNanos;
			if (click) RemoteInputDiagnostics.recordClick(false);
			callback.setReturnValue(click);
		}
	}
}
