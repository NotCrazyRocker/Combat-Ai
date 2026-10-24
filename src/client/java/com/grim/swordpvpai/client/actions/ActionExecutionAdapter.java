package com.grim.swordpvpai.client.actions;

/**
 * Backend-independent action interface.
 *
 * The client adapter implements this now.
 * A server-side HeroBot adapter can implement the same
 * interface during the later self-play phase.
 */
public interface ActionExecutionAdapter {
    void applyButtons(ActionCommand.Buttons buttons);

    void applyAim(ActionCommand.Aim aim);

    void applyOneShot(String action);

    void selectHotbar(int slot);

    void releaseAll();
}