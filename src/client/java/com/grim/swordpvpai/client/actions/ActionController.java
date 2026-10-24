package com.grim.swordpvpai.client.actions;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;

import java.util.concurrent.ArrayBlockingQueue;

public final class ActionController {
    private static final int QUEUE_CAPACITY = 128;

    private static final ArrayBlockingQueue<ActionCommand> INCOMING =
            new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private static final ActionExecutionAdapter ADAPTER =
            new ClientPlayerActionAdapter();

    private static ActionCommand.Buttons activeButtons =
            new ActionCommand.Buttons();

    private static long clientTick = 0;
    private static long expiresAtTick = 0;
    private static boolean initialized = false;

    private ActionController() {
    }

    public static synchronized void initialize() {
        if (initialized) {
            return;
        }

        initialized = true;

        ActionSocketServer.start();

        ClientTickEvents.END_CLIENT_TICK.register(
                ActionController::tick
        );

        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            releaseAndClear();
            ActionSocketServer.stop();
        });
    }

    /**
     * Called by the TCP receiver thread.
     * Never perform game-state changes from this method.
     */
    public static void submit(ActionCommand command) {
        if (command == null) {
            return;
        }

        // Keep the queue bounded if Python sends too quickly.
        if (!INCOMING.offer(command)) {
            INCOMING.poll();
            INCOMING.offer(command);
        }
    }

    private static void tick(Minecraft client) {
        clientTick++;

        /*
         * Never let incoming automation control a menu, inventory
         * screen, title screen, or player who is not in a world.
         */
        if (client.player == null
                || client.level == null
                || client.screen != null) {
            releaseAndClear();
            return;
        }

        /*
         * Drain any backlog and apply the newest full desired state.
         * This avoids replaying a long sequence of stale movement
         * commands after a temporary delay.
         */
        ActionCommand newest = null;
        ActionCommand pending;

        while ((pending = INCOMING.poll()) != null) {
            newest = pending;
        }

        if (newest != null) {
            if (newest.buttons == null) {
                newest.buttons = new ActionCommand.Buttons();
            }

            activeButtons = newest.buttons;

            int duration = Math.max(
                    1,
                    Math.min(newest.valid_for_ticks, 20)
            );

            expiresAtTick = clientTick + duration;

            if (newest.aim != null) {
                ADAPTER.applyAim(newest.aim);
            }

            if (newest.hotbar_slot != null) {
                ADAPTER.selectHotbar(newest.hotbar_slot);
            }
        }

        /*
         * Expiration provides a failsafe if Python crashes or stops
         * sending updates. The keys are then automatically released.
         */
        if (clientTick >= expiresAtTick) {
            activeButtons = new ActionCommand.Buttons();
        }

        // Apply persistent buttons every client tick.
        ADAPTER.applyButtons(activeButtons);

        /*
         * Execute one-time presses only for the newest command.
         * Do not replay presses from commands skipped during backlog
         * draining.
         */
        if (newest != null && newest.press != null) {
            for (String action : newest.press) {
                ADAPTER.applyOneShot(action);
            }
        }
    }

    private static void releaseAndClear() {
        INCOMING.clear();

        activeButtons = new ActionCommand.Buttons();
        expiresAtTick = 0;

        ADAPTER.releaseAll();
    }
}