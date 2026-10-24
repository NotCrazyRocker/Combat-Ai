package com.grim.swordpvpai.client.actions;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import com.grim.swordpvpai.client.bridge.WebSocketBridge;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;

public final class ActionController {
    private static final int QUEUE_CAPACITY = 128;

    private static final ArrayBlockingQueue<ActionCommand> INCOMING =
            new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private static final ActionExecutionAdapter ADAPTER =
            new ClientPlayerActionAdapter();

    private static ActionCommand.Buttons activeButtons =
            new ActionCommand.Buttons();

    private static boolean initialized;
    private static boolean automationActive;

    private static long clientTick;
    private static long expiresAtTick;

    private ActionController() {
    }

    public static synchronized void initialize() {
        if (initialized) {
            return;
        }

        initialized = true;

        WebSocketBridge.startBridge();

        ClientTickEvents.END_CLIENT_TICK.register(
                ActionController::tick
        );

        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            deactivateAutomation();
            INCOMING.clear();
            WebSocketBridge.stopBridge();
        });
    }

    /**
     * Called from the WebSocket thread.
     * It does not access or modify Minecraft game state.
     */
    public static void submit(ActionCommand command) {
        if (command == null) {
            return;
        }

        if (!INCOMING.offer(command)) {
            // Bound memory use if the producer gets far ahead.
            INCOMING.poll();
            INCOMING.offer(command);
        }
    }

    private static void tick(Minecraft client) {
        clientTick++;

        // Never control the player while a GUI is open or no world exists.
        if (client.player == null
                || client.level == null
                || client.screen != null) {
            INCOMING.clear();
            deactivateAutomation();
            return;
        }

        /*
         * Drain the batch.
         *
         * Persistent button state comes from the newest command.
         * One-time presses from every received command are preserved.
         */
        List<ActionCommand> batch = new ArrayList<>();
        ActionCommand command;

        while ((command = INCOMING.poll()) != null) {
            batch.add(command);
        }

        ActionCommand newest = batch.isEmpty()
                ? null
                : batch.get(batch.size() - 1);

        if (newest != null) {
            activeButtons = newest.buttons != null
                    ? newest.buttons
                    : new ActionCommand.Buttons();

            int duration = Math.max(
                    1,
                    Math.min(newest.valid_for_ticks, 20)
            );

            automationActive = true;
            expiresAtTick = clientTick + duration;

            if (newest.aim != null) {
                ADAPTER.applyAim(newest.aim);
            }

            if (newest.hotbar_slot != null) {
                ADAPTER.selectHotbar(newest.hotbar_slot);
            }
        }

        /*
         * No new command arrived before the lease expired:
         * release AI-controlled buttons once, then leave native
         * keyboard controls alone.
         */
        if (automationActive && clientTick >= expiresAtTick) {
            deactivateAutomation();
            return;
        }

        /*
         * Critical fix:
         * do not modify key states when automation is inactive.
         * This preserves ordinary manual movement before Python
         * starts and after its commands stop.
         */
        if (automationActive) {
            ADAPTER.applyButtons(activeButtons);
        }

        /*
         * Process one-time events from all commands in this batch.
         * Do not execute only the newest command's press list.
         */
        for (ActionCommand received : batch) {
            if (received.press == null) {
                continue;
            }

            for (String action : received.press) {
                ADAPTER.applyOneShot(action);
            }
        }
    }

    private static void deactivateAutomation() {
        if (automationActive) {
            ADAPTER.releaseAll();
        }

        automationActive = false;
        expiresAtTick = 0;
        activeButtons = new ActionCommand.Buttons();
    }
}