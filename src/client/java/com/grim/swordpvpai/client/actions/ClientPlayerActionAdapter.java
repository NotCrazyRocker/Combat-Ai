package com.grim.swordpvpai.client.actions;

import net.minecraft.client.Minecraft;
import net.minecraft.util.Mth;
import net.minecraft.client.player.LocalPlayer;

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

        mc.options.keyUp.setDown(b.forward);
        mc.options.keyDown.setDown(b.backward);
        mc.options.keyLeft.setDown(b.left);
        mc.options.keyRight.setDown(b.right);

        mc.options.keyJump.setDown(b.jump);
        mc.options.keySprint.setDown(b.sprint);
        mc.options.keyShift.setDown(b.sneak);

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

        yaw = Mth.wrapDegrees(yaw);
        pitch = Mth.clamp(pitch, -90.0F, 90.0F);

        player.setYRot(yaw);
        player.setYHeadRot(yaw);
        player.setXRot(pitch);
    }

    @Override
    public void applyOneShot(String action) {
        Minecraft mc = mc();

        if (mc.options == null || action == null) {
            return;
        }

        /*
         * These presses are applied after persistent button states.
         * On the next tick, applyButtons() restores the requested
         * persistent state, so a one-time press is not held forever.
         */
        switch (action.toLowerCase()) {
            case "jump" ->
                mc.options.keyJump.setDown(true);

            case "attack" ->
                mc.options.keyAttack.setDown(true);

            case "use_item" ->
                mc.options.keyUse.setDown(true);

            default -> {
                // Ignore unsupported one-time actions.
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