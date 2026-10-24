package com.grim.swordpvpai.client.actions;

import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.util.Mth;

public final class ClientPlayerActionAdapter
        implements ActionExecutionAdapter {

    private Minecraft mc() {
        return Minecraft.getInstance();
    }

    @Override
    public void applyButtons(ActionCommand.Buttons b) {
        Minecraft mc = mc();

        if (mc.options == null || b == null) {
            return;
        }

        // Persistent movement state.
        mc.options.keyUp.setDown(b.forward);
        mc.options.keyDown.setDown(b.backward);
        mc.options.keyLeft.setDown(b.left);
        mc.options.keyRight.setDown(b.right);

        mc.options.keyJump.setDown(b.jump);
        mc.options.keySprint.setDown(b.sprint);
        mc.options.keyShift.setDown(b.sneak);

        // Persistent mouse-button state.
        // A new attack click still requires a one-shot event.
        mc.options.keyAttack.setDown(b.attack);
        mc.options.keyUse.setDown(b.use_item);
    }

    @Override
    public void applyAim(ActionCommand.Aim aim) {
        Minecraft mc = mc();
        LocalPlayer player = mc.player;

        if (player == null || aim == null) {
            return;
        }

        if (!Double.isFinite(aim.yaw)
                || !Double.isFinite(aim.pitch)) {
            return;
        }

        float yaw;
        float pitch;

        if ("absolute".equalsIgnoreCase(aim.mode)) {
            yaw = (float) aim.yaw;
            pitch = (float) aim.pitch;
        } else {
            yaw = player.getYRot() + (float) aim.yaw;
            pitch = player.getXRot() + (float) aim.pitch;
        }

        player.setYRot(Mth.wrapDegrees(yaw));
        player.setYHeadRot(Mth.wrapDegrees(yaw));
        player.setXRot(Mth.clamp(pitch, -90.0F, 90.0F));
    }

    @Override
    public void applyOneShot(String action) {
        Minecraft mc = mc();

        if (mc.options == null || action == null) {
            return;
        }

        switch (action.toLowerCase()) {
            case "jump" ->
                // One-tick jump press; subsequent button state decides
                // whether jump remains held.
                mc.options.keyJump.setDown(true);

            case "attack" ->
                // Generate an actual click event for vanilla to consume.
                KeyMapping.click(
                        mc.options.keyAttack.getDefaultKey()
                );

            case "use_item" ->
                KeyMapping.click(
                        mc.options.keyUse.getDefaultKey()
                );

            default -> {
                // Ignore unknown one-shot actions.
            }
        }
    }

    @Override
    public void selectHotbar(int slot) {
        Minecraft mc = mc();

        if (mc.player == null || slot < 0 || slot > 8) {
            return;
        }

        mc.player.getInventory().setSelectedSlot(slot);
    }

    @Override
    public void releaseAll() {
        applyButtons(new ActionCommand.Buttons());
    }
}