package com.vivian.glass;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.json.JSONException;
import org.json.JSONObject;

import java.net.URI;
import java.net.URISyntaxException;

/**
 * Manages the WebSocket connection to the Pi's glass_server.py.
 * Auto-reconnects with exponential backoff (1s -> 2s -> 4s -> ... -> 15s max).
 * Parses incoming state JSON and updates VivianState.
 */
public class WebSocketManager {
    private static final String TAG = "VivianWS";

    // Pi AP gateway address -- standard hostapd default
    private static final String SERVER_URI = "ws://192.168.4.1:9100";

    private static final int INITIAL_RECONNECT_DELAY = 1000;
    private static final int MAX_RECONNECT_DELAY = 15000;

    // If no message received for this long, assume connection is dead
    // Server sends every 0.5s with no delta detection, so 5s means genuinely dead
    private static final int STALE_TIMEOUT = 5000;
    private static final int STALE_CHECK_INTERVAL = 3000;

    private final VivianState state;
    private final Handler handler;
    private WebSocketClient client;
    private int reconnectDelay = INITIAL_RECONNECT_DELAY;
    private boolean shouldConnect = true;
    private volatile long lastMessageTime = 0;

    public WebSocketManager(VivianState state) {
        this.state = state;
        this.handler = new Handler(Looper.getMainLooper());
    }

    /**
     * Start connecting to the Pi. Will auto-reconnect on failure/disconnect.
     */
    public void connect() {
        if (!shouldConnect) return;

        URI uri;
        try {
            uri = new URI(SERVER_URI);
        } catch (URISyntaxException e) {
            Log.e(TAG, "Invalid server URI", e);
            return;
        }

        client = new WebSocketClient(uri) {
            @Override
            public void onOpen(ServerHandshake handshake) {
                Log.i(TAG, "Connected to VIVIAN");
                state.connected = true;
                reconnectDelay = INITIAL_RECONNECT_DELAY;
                lastMessageTime = System.currentTimeMillis();
                startStaleChecker();
            }

            @Override
            public void onMessage(String message) {
                lastMessageTime = System.currentTimeMillis();
                try {
                    JSONObject json = new JSONObject(message);
                    String type = json.optString("type", "");
                    if ("state".equals(type)) {
                        state.updateFromJson(json);
                    }
                } catch (JSONException e) {
                    Log.d(TAG, "Invalid JSON: " + message);
                }
            }

            @Override
            public void onClose(int code, String reason, boolean remote) {
                Log.i(TAG, "Disconnected: " + reason);
                state.connected = false;
                scheduleReconnect();
            }

            @Override
            public void onError(Exception ex) {
                Log.e(TAG, "WebSocket error", ex);
                state.connected = false;
            }
        };

        client.setConnectionLostTimeout(0);  // disable — server broadcasts are implicit keepalive
        client.connect();
    }

    /**
     * Send a command to the Pi (e.g., music control).
     * Silent no-op if not connected.
     */
    public void sendCommand(String action) {
        if (client == null || !client.isOpen()) return;

        try {
            JSONObject cmd = new JSONObject();
            cmd.put("type", "cmd");
            cmd.put("action", action);
            client.send(cmd.toString());
        } catch (JSONException e) {
            Log.e(TAG, "Error building command JSON", e);
        }
    }

    /**
     * Disconnect and stop auto-reconnect.
     */
    public void disconnect() {
        shouldConnect = false;
        handler.removeCallbacksAndMessages(null);
        if (client != null) {
            client.close();
        }
        state.connected = false;
    }

    private void scheduleReconnect() {
        if (!shouldConnect) return;

        Log.d(TAG, "Reconnecting in " + reconnectDelay + "ms");
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                connect();
            }
        }, reconnectDelay);

        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    }

    /**
     * Periodically checks if the connection has gone stale (no messages received).
     * If the server restarts or the connection silently drops, onClose may never fire
     * because connectionLostTimeout is disabled. This catches that case.
     */
    private void startStaleChecker() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (!shouldConnect) return;

                if (client != null && client.isOpen()
                        && System.currentTimeMillis() - lastMessageTime > STALE_TIMEOUT) {
                    Log.w(TAG, "Connection stale — no data for " + STALE_TIMEOUT + "ms, forcing reconnect");
                    state.connected = false;
                    // Close on a background thread to avoid blocking the UI
                    final WebSocketClient staleClient = client;
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                staleClient.closeBlocking();
                            } catch (Exception ignored) {}
                        }
                    }).start();
                    scheduleReconnect();
                    return;
                }

                // Keep checking while connected
                if (client != null && client.isOpen()) {
                    handler.postDelayed(this, STALE_CHECK_INTERVAL);
                }
            }
        }, STALE_CHECK_INTERVAL);
    }
}
