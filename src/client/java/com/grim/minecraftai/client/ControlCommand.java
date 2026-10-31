package com.grim.minecraftai.client;

public record ControlCommand(
		long sequence,
		long observationTick,
		boolean forward,
		boolean backward,
		boolean left,
		boolean right,
		boolean jump,
		boolean sneak,
		boolean sprint,
		boolean attack,
		boolean use,
		float yawDelta,
		float pitchDelta,
		long receivedAtNanos
) {
	static final long TIMEOUT_NANOS = 500_000_000L;

	boolean isFresh(long nowNanos) {
		return nowNanos - receivedAtNanos <= TIMEOUT_NANOS;
	}
}
