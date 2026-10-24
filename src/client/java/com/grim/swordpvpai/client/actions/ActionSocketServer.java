package com.grim.swordpvpai.client.actions;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public final class ActionSocketServer {
    private static final Gson GSON = new Gson();

    private static ActionSocketServer instance;

    private final ServerSocket serverSocket;
    private volatile boolean running;
    private Thread thread;

    private ActionSocketServer() throws IOException {
        int port = Integer.getInteger(
                "swordpvpai.action.port", 8766
        );

        serverSocket = new ServerSocket();
        serverSocket.setReuseAddress(true);

        // Local machine only. Do not expose the action port to a LAN.
        serverSocket.bind(new InetSocketAddress(
                InetAddress.getByName("127.0.0.1"),
                port
        ));
    }

    public static synchronized void start() {
        if (instance != null) {
            return;
        }

        try {
            instance = new ActionSocketServer();
            instance.running = true;

            instance.thread = new Thread(
                    instance::run,
                    "Combat-AI-Action-Server"
            );

            instance.thread.setDaemon(true);
            instance.thread.start();

            System.out.println(
                    "[Combat-AI] Action server listening on 127.0.0.1:"
                    + instance.serverSocket.getLocalPort()
            );

        } catch (IOException e) {
            System.err.println(
                    "[Combat-AI] Could not start action server: "
                    + e.getMessage()
            );

            instance = null;
        }
    }

    public static synchronized void stop() {
        if (instance == null) {
            return;
        }

        instance.running = false;

        try {
            instance.serverSocket.close();
        } catch (IOException ignored) {
        }

        instance = null;
    }

    private void run() {
        while (running) {
            try (Socket socket = serverSocket.accept()) {
                socket.setTcpNoDelay(true);

                System.out.println(
                        "[Combat-AI] Python action client connected."
                );

                readCommands(socket);

            } catch (IOException e) {
                if (running) {
                    System.err.println(
                            "[Combat-AI] Action connection error: "
                            + e.getMessage()
                    );
                }
            }
        }
    }

    private void readCommands(Socket socket) throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(
                        socket.getInputStream(),
                        StandardCharsets.UTF_8
                ))) {

            String line;

            while (running && (line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                try {
                    ActionCommand command =
                            GSON.fromJson(line, ActionCommand.class);

                    if (command == null
                            || command.protocol_version != 1
                            || !"action".equals(command.type)) {
                        System.err.println(
                                "[Combat-AI] Rejected invalid action."
                        );
                        continue;
                    }

                    ActionController.submit(command);

                } catch (JsonParseException | IllegalStateException e) {
                    System.err.println(
                            "[Combat-AI] Invalid action JSON: "
                            + e.getMessage()
                    );
                }
            }
        }

        System.out.println(
                "[Combat-AI] Python action client disconnected."
        );
    }
}