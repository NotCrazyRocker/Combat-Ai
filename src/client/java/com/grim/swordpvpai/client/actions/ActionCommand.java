package com.grim.swordpvpai.client.actions;

import java.util.ArrayList;
import java.util.List;

/**
 * A complete desired action state received from Python.
 *
 * "buttons" are persistent states.
 * "press" contains one-time button presses.
 * "hotbar_slot" is a one-time selection, numbered 0-8.
 */
public final class ActionCommand {
    public int protocol_version = 1;
    public String type = "action";

    public long sequence = 0;

    // How many client ticks this command remains valid.
    // The controller clamps this to 1-20 ticks.
    public int valid_for_ticks = 2;

    public Buttons buttons = new Buttons();
    public Aim aim;

    public List<String> press = new ArrayList<>();

    // Optional, zero-based hotbar slot.
    public Integer hotbar_slot;

    public static final class Buttons {
        public boolean forward;
        public boolean backward;
        public boolean left;
        public boolean right;

        public boolean jump;
        public boolean sprint;
        public boolean sneak;

        public boolean attack;
        public boolean use_item;
    }

    public static final class Aim {
        // "absolute": set yaw/pitch.
        // "delta": add yaw/pitch to current rotation.
        public String mode = "delta";

        // Degrees.
        public double yaw;
        public double pitch;
    }
}