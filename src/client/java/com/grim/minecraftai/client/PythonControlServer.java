package com.grim.minecraftai.client;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.grim.minecraftai.MinecraftAI;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicReference;

final class PythonControlServer {
	private static final int MAX_MESSAGE_LENGTH = 4096;
	private static final int MAX_QUEUED_AIM_COMMANDS = 64;

	private final int port;
	private volatile ControlCommand latestCommand;
	private final ConcurrentLinkedQueue<ControlCommand> aimCommands = new ConcurrentLinkedQueue<>();
	private volatile ConnectionSession session;
	private volatile long latestObservationTick;
	private final ArrayDeque<List<Object>> actionHistory = new ArrayDeque<>();

	PythonControlServer(int port) {
		this.port = port;
	}

	void start() {
		Thread thread = new Thread(this::serve, "MinecraftAI Python control server");
		thread.setDaemon(true);
		thread.start();
	}

	synchronized boolean accept(ControlCommand command) {
		if (!MinecraftAIClient.isAIControlEnabled()) {
			return false;
		}

		latestCommand = command;
		actionHistory.addLast(List.of(command.forward(), command.backward(), command.left(), command.right(),
				command.jump(), command.sprint(), command.sneak(), command.attack(), command.use(),
				command.yawDelta(), command.pitchDelta()));
		while (actionHistory.size() > 20) actionHistory.removeFirst();
		while (aimCommands.size() >= MAX_QUEUED_AIM_COMMANDS) {
			aimCommands.poll();
		}
		aimCommands.add(command);
		return true;
	}

	synchronized void clear() {
		latestCommand = null;
		aimCommands.clear();
		actionHistory.clear();
	}

	synchronized List<List<Object>> actionHistory() {
		List<List<Object>> history = new ArrayList<>(20);
		for (int i = actionHistory.size(); i < 20; i++) history.add(List.of(false, false, false, false, false, false, false, false, false, 0.0, 0.0));
		history.addAll(actionHistory);
		return history;
	}

	void publishObservation(JsonObject observation, long tick) {
		latestObservationTick = tick;
		ConnectionSession active = session;
		if (active != null && MinecraftAIClient.isAIControlEnabled()) active.sendObservation(observation.toString());
	}

	synchronized void publishControlState(boolean enabled) {
		ConnectionSession active = session;
		if (active != null) {
			active.sendReliable("{\"type\":\"control_state\",\"ai_control_enabled\":" + enabled + "}");
		}
	}

	ControlCommand getCurrentCommand() {
		ControlCommand command = latestCommand;
		if (!MinecraftAIClient.isAIControlEnabled() || command == null || !command.isFresh(System.nanoTime())) {
			return null;
		}

		return command;
	}

	ControlCommand pollAimCommand() {
		if (!MinecraftAIClient.isAIControlEnabled()) {
			return null;
		}

		ControlCommand command;
		while ((command = aimCommands.poll()) != null) {
			if (command.isFresh(System.nanoTime())) {
				return command;
			}
		}
		return null;
	}

	private void serve() {
		try (ServerSocket server = new ServerSocket(port, 1, InetAddress.getByName("127.0.0.1"))) {
			server.setSoTimeout(1000);
			MinecraftAI.LOGGER.info("Python control listening on 127.0.0.1:{} (TCP)", port);

			while (!Thread.currentThread().isInterrupted()) {
				try (Socket client = server.accept()) {
					handle(client);
				} catch (SocketTimeoutException ignored) {
					// Periodically return to the loop so daemon-thread shutdown remains responsive.
				} catch (IOException exception) {
					MinecraftAI.LOGGER.warn("Python control connection ended: {}", exception.getMessage());
				} finally {
					clear();
				}
			}
		} catch (IOException exception) {
			MinecraftAI.LOGGER.error("Could not start Python control server on 127.0.0.1:{}", port, exception);
		}
	}

	private void handle(Socket client) throws IOException {
		client.setSoTimeout(1000);
		client.setTcpNoDelay(true);
		ConnectionSession active = new ConnectionSession(client);
		try (active;
			 BufferedReader reader = new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8))) {
			RemoteInputDiagnostics.resetAppliedAction();
			synchronized (this) {
				active.sendReliable("{\"type\":\"ready\",\"protocol\":2,\"observation_schema\":\"minecraftai.pvp.v1\",\"ai_control_enabled\":" + MinecraftAIClient.isAIControlEnabled() + "}");
				session = active;
			}
			long lastSequence = -1;
			long lastTick = -1;

			while (true) {
				String message;
				try {
					message = readMessage(reader);
				} catch (SocketTimeoutException exception) {
					continue;
				}

				if (message == null) {
					return;
				}

				long sequence = -1;
				try {
					JsonObject json = JsonParser.parseString(message).getAsJsonObject();
					if (!json.has("type") || !json.get("type").getAsString().equals("control")) {
						throw new IllegalArgumentException("'type' must be 'control'");
					}
					sequence = requiredLong(json, "seq");
					if (sequence <= lastSequence) {
						writeAck(active, sequence, false, "sequence must increase");
						continue;
					}

					ControlCommand command = parseCommand(json, sequence);
					long currentTick = latestObservationTick;
					if (command.observationTick() <= lastTick) {
						writeAck(active, sequence, false, "observation tick was already used");
						continue;
					}
					if (command.observationTick() > currentTick) {
						writeAck(active, sequence, false, "observation tick is in the future");
						continue;
					}
					if (currentTick - command.observationTick() > 2) {
						writeAck(active, sequence, false, "observation tick is stale");
						continue;
					}
					boolean accepted = accept(command);
					if (accepted) {
						lastSequence = sequence;
						lastTick = command.observationTick();
						writeAck(active, sequence, true, null);
					} else {
						writeAck(active, sequence, false, "AI control is disabled; press the in-game toggle key first");
					}
				} catch (RuntimeException exception) {
					writeAck(active, sequence, false, "invalid control message: " + exception.getMessage());
				}
			}
		} finally {
			synchronized (this) {
				if (session == active) session = null;
			}
		}
	}

	private static ControlCommand parseCommand(JsonObject json, long sequence) {
		float yawDelta = requiredFiniteFloat(json, "yaw_delta");
		float pitchDelta = requiredFiniteFloat(json, "pitch_delta");

		return new ControlCommand(
				sequence,
				requiredLong(json, "tick"),
				optionalBoolean(json, "forward"),
				optionalBoolean(json, "backward"),
				optionalBoolean(json, "left"),
				optionalBoolean(json, "right"),
				optionalBoolean(json, "jump"),
				optionalBoolean(json, "sneak"),
				optionalBoolean(json, "sprint"),
				optionalBoolean(json, "attack"),
				optionalBoolean(json, "use"),
				yawDelta,
				pitchDelta,
				System.nanoTime()
		);
	}

	private static boolean optionalBoolean(JsonObject json, String field) {
		if (!json.has(field)) {
			return false;
		}
		if (!json.get(field).isJsonPrimitive() || !json.get(field).getAsJsonPrimitive().isBoolean()) {
			throw new IllegalArgumentException("'" + field + "' must be a boolean");
		}
		return json.get(field).getAsBoolean();
	}

	private static long requiredLong(JsonObject json, String field) {
		if (!json.has(field) || !json.get(field).isJsonPrimitive() || !json.get(field).getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("'" + field + "' must be an integer");
		}
		try {
			return json.get(field).getAsBigDecimal().longValueExact();
		} catch (ArithmeticException exception) {
			throw new IllegalArgumentException("'" + field + "' must be an integer", exception);
		}
	}

	private static float requiredFiniteFloat(JsonObject json, String field) {
		if (!json.has(field) || !json.get(field).isJsonPrimitive() || !json.get(field).getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("'" + field + "' must be a number");
		}
		float value = json.get(field).getAsFloat();
		if (!Float.isFinite(value)) {
			throw new IllegalArgumentException("'" + field + "' must be finite");
		}
		return value;
	}

	private static String readMessage(BufferedReader reader) throws IOException {
		StringBuilder message = new StringBuilder();
		while (true) {
			int character = reader.read();
			if (character == -1) {
				return null;
			}
			if (character == '\n') {
				return message.toString();
			}
			if (character != '\r') {
				if (message.length() >= MAX_MESSAGE_LENGTH) {
					throw new IOException("control message exceeded maximum length");
				}
				message.append((char) character);
			}
		}
	}

	private void writeAck(ConnectionSession session, long sequence, boolean accepted, String error) throws IOException {
		session.sendReliable("{\"type\":\"ack\",\"seq\":" + sequence
				+ ",\"accepted\":" + accepted
				+ ",\"ai_control_enabled\":" + MinecraftAIClient.isAIControlEnabled()
				+ ",\"latest_observation_tick\":" + latestObservationTick
				+ ",\"error\":" + (error == null ? "null" : quoteJson(error)) + "}");
	}

	private static final class ConnectionSession implements AutoCloseable {
		private final Socket socket;
		private final BufferedWriter writer;
		private final ArrayBlockingQueue<String> reliableOutgoing = new ArrayBlockingQueue<>(128);
		private final AtomicReference<String> latestObservation = new AtomicReference<>();
		private final Semaphore available = new Semaphore(0);
		private final Thread sender;
		private volatile boolean closed;

		ConnectionSession(Socket socket) throws IOException {
			this.socket = socket;
			writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
			sender = new Thread(() -> {
				try {
					while (!closed) {
						String line = reliableOutgoing.poll();
						if (line == null) line = latestObservation.getAndSet(null);
						if (line == null) {
							available.acquire();
							continue;
						}
						writer.write(line);
						writer.write('\n');
						writer.flush();
					}
				} catch (Exception ignored) {
					close();
				}
			}, "MinecraftAI Python observation writer");
			sender.setDaemon(true);
			sender.start();
		}

		void sendReliable(String line) {
			if (closed) return;
			if (!reliableOutgoing.offer(line)) {
				close();
				return;
			}
			available.release();
		}

		void sendObservation(String line) {
			if (closed) return;
			if (latestObservation.getAndSet(line) == null) available.release();
		}

		@Override public void close() {
			if (closed) return;
			closed = true;
			sender.interrupt();
			try {
				socket.close();
			} catch (IOException ignored) {
			}
		}
	}

	private static String quoteJson(String value) {
		return new com.google.gson.Gson().toJson(value);
	}
}
