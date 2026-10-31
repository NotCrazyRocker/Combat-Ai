package com.grim.minecraftai.client.mixin;

import com.grim.minecraftai.client.MinecraftAIClient;
import net.minecraft.client.multiplayer.ClientPacketListener;
import net.minecraft.network.protocol.game.ClientboundDamageEventPacket;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ClientPacketListener.class)
abstract class ClientPacketListenerMixin {
	@Inject(method = "handleDamageEvent", at = @At("HEAD"))
	private void minecraftai$recordDamageEvent(ClientboundDamageEventPacket packet, CallbackInfo ci) {
		MinecraftAIClient.recordDamageEvent(packet.entityId(), packet.sourceCauseId());
	}
}
