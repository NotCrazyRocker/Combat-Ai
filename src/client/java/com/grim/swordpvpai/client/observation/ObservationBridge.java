package com.grim.swordpvpai.client.observation;

import com.grim.swordpvpai.SwordPvPAI;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Non-blocking-to-the-game-thread JSON-lines transport to the local Python process. */
public final class ObservationBridge {
    private static final String HOST = System.getProperty("swordpvpai.observation.host", "127.0.0.1");
    private static final int PORT = Integer.getInteger("swordpvpai.observation.port", 8765);
    private static final int QUEUE_CAPACITY = 512;
    private static final ArrayBlockingQueue<String> QUEUE = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static final AtomicBoolean STARTED = new AtomicBoolean(false);
    private static final AtomicLong DROPPED = new AtomicLong();

    private ObservationBridge() {}

    public static void start() {
        if (!STARTED.compareAndSet(false, true)) return;
        Thread thread = new Thread(ObservationBridge::runSender, "SwordPvPAI-observation-bridge");
        thread.setDaemon(true);
        thread.start();
        SwordPvPAI.LOGGER.info("Observation bridge started; Python receiver should listen on {}:{}", HOST, PORT);
    }

    public static void send(String jsonLine) {
        if (!STARTED.get()) return;
        if (!QUEUE.offer(jsonLine)) {
            QUEUE.poll();
            DROPPED.incrementAndGet();
            QUEUE.offer(jsonLine);
        }
    }

    public static long droppedCount() {
        return DROPPED.get();
    }

    private static void runSender() {
        while (!Thread.currentThread().isInterrupted()) {
            try (Socket socket = new Socket()) {
                socket.connect(new InetSocketAddress(HOST, PORT), 500);
                socket.setTcpNoDelay(true);
                try (BufferedWriter writer = new BufferedWriter(
                        new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8))) {
                    SwordPvPAI.LOGGER.info("Connected to Python observation receiver at {}:{}", HOST, PORT);
                    while (!socket.isClosed()) {
                        String line = QUEUE.poll(500, TimeUnit.MILLISECONDS);
                        if (line == null) continue;
                        writer.write(line);
                        writer.newLine();
                        writer.flush();
                    }
                }
            } catch (IOException exception) {
                // Python may be started after Minecraft. Retry without blocking the client tick.
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            } catch (RuntimeException exception) {
                SwordPvPAI.LOGGER.warn("Observation bridge recovered from a sender error", exception);
            }
        }
    }
}
