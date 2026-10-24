package com.grim.swordpvpai.client.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import com.grim.swordpvpai.client.actions.ActionCommand;
import com.grim.swordpvpai.client.actions.ActionController;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.UUID;

public final class WebSocketBridge extends WebSocketServer {
    public static final int PROTOCOL_VERSION = 1;
    public static final String AGENT_ID = "client_player";

    private static final Gson GSON = new Gson();
    private static WebSocketBridge instance;

    private final String matchId = UUID.randomUUID().toString();

    private volatile long latestObservationTick = -1;
    private volatile long lastActionSequence = -1;

    private WebSocketBridge(int port) throws Exception {
        super(new InetSocketAddress(
                InetAddress.getByName("127.0.0.1"),
                port
        ));
    }

    public static synchronized void startBridge() {
        if (instance != null) {
            return;
        }

        try {
            int port = Integer.getInteger(
                    "swordpvpai.websocket.port", 8765
            );

            instance = new WebSocketBridge(port);
            instance.setDaemon(true);
            instance.start();

        } catch (Exception e) {
            instance = null;
            System.err.println(
                    "[Combat-AI] WebSocket startup failed: "
                    + e.getMessage()
            );
        }
    }

    public static synchronized void stopBridge() {
        if (instance == null) {
            return;
        }

        WebSocketBridge old = instance;
        instance = null;

        try {
            old.stop(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void onStart() {
        System.out.println(
                "[Combat-AI] WebSocket bridge listening on "
                + "ws://127.0.0.1:" + getPort()
        );
    }

    @Override
    public void onOpen(WebSocket connection, ClientHandshake handshake) {
        JsonObject hello = new JsonObject();

        hello.addProperty("protocol_version", PROTOCOL_VERSION);
        hello.addProperty("type", "hello");
        hello.addProperty("agent_id", AGENT_ID);
        hello.addProperty("match_id", matchId);

        connection.send(GSON.toJson(hello));

        System.out.println("[Combat-AI] Python WebSocket connected.");
    }

    @Override
    public void onClose(
            WebSocket connection,
            int code,
            String reason,
            boolean remote
    ) {
        System.out.println(
                "[Combat-AI] WebSocket disconnected: " + reason
        );
    }

    @Override
    public void onError(WebSocket connection, Exception error) {
        System.err.println(
                "[Combat-AI] WebSocket error: " + error.getMessage()
        );
    }

    @Override
    public void onMessage(WebSocket connection, String message) {
        try {
            JsonObject root = JsonParser.parseString(message)
                    .getAsJsonObject();

            if (root.get("protocol_version").getAsInt()
                    != PROTOCOL_VERSION) {
                reject(connection, "protocol_version");
                return;
            }

            if (!"action".equals(root.get("type").getAsString())) {
                reject(connection, "message_type");
                return;
            }

            if (!AGENT_ID.equals(
                    root.get("agent_id").getAsString())) {
                reject(connection, "agent_id");
                return;
            }

            if (!matchId.equals(
                    root.get("match_id").getAsString())) {
                reject(connection, "match_id");
                return;
            }

            long sequence = root.get("sequence").getAsLong();
            long tick = root.get("tick").getAsLong();

            // Enforce strictly increasing action sequence numbers.
            synchronized (this) {
                if (sequence <= lastActionSequence) {
                    reject(connection, "stale_sequence");
                    return;
                }

                long observedTick = latestObservationTick;

                // No action is accepted before the first observation.
                if (observedTick < 0) {
                    reject(connection, "no_observation_yet");
                    return;
                }

                // Permit small inference/network delays, not old actions.
                if (tick < observedTick - 5 || tick > observedTick + 1) {
                    reject(connection, "stale_or_future_tick");
                    return;
                }

                if (!root.has("action")
                        || !root.get("action").isJsonObject()) {
                    reject(connection, "missing_action");
                    return;
                }

                ActionCommand action = GSON.fromJson(
                        root.getAsJsonObject("action"),
                        ActionCommand.class
                );

                if (action == null) {
                    reject(connection, "invalid_action");
                    return;
                }

                action.protocol_version = PROTOCOL_VERSION;
                action.type = "action";
                action.sequence = sequence;

                if (action.buttons == null) {
                    action.buttons = new ActionCommand.Buttons();
                }

                // Advance only after the message passes validation.
                lastActionSequence = sequence;
                ActionController.submit(action);
            }

        } catch (Exception e) {
            reject(connection, "invalid_json");
        }
    }

    private void reject(WebSocket connection, String reason) {
        JsonObject response = new JsonObject();

        response.addProperty("protocol_version", PROTOCOL_VERSION);
        response.addProperty("type", "rejected");
        response.addProperty("reason", reason);

        if (connection != null && connection.isOpen()) {
            connection.send(GSON.toJson(response));
        }
    }

    /**
     * Call this on the client/game thread after creating an observation.
     *
     * observationJson is the JSON produced by the existing observation
     * collector, not a screenshot or a client-side visual inference.
     */
    public static void publishObservation(
            String observationJson,
            long tick
    ) {
        WebSocketBridge bridge = instance;

        if (bridge == null || observationJson == null) {
            return;
        }

        JsonElement data;

        try {
            data = JsonParser.parseString(observationJson);
        } catch (Exception e) {
            System.err.println(
                    "[Combat-AI] Invalid observation JSON: "
                    + e.getMessage()
            );
            return;
        }

        bridge.latestObservationTick = tick;

        JsonObject envelope = new JsonObject();
        envelope.addProperty("protocol_version", PROTOCOL_VERSION);
        envelope.addProperty("type", "observation");
        envelope.addProperty("agent_id", AGENT_ID);
        envelope.addProperty("tick", tick);
        envelope.addProperty("sequence", tick);
        envelope.addProperty("match_id", bridge.matchId);
        envelope.add("data", data);

        String json = GSON.toJson(envelope);

        // Do not block the game thread on a slow or disconnected client.
        for (WebSocket connection : bridge.getConnections()) {
            if (connection.isOpen()) {
                try {
                    connection.send(json);
                } catch (Exception e) {
                    System.err.println(
                            "[Combat-AI] Observation send failed: "
                            + e.getMessage()
                    );
                }
            }
        }
    }
}