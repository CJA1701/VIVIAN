package com.vivian.glass;

import android.graphics.Bitmap;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Thread-safe state holder for all HUD data.
 * Updated by WebSocketManager on the network thread.
 * Read by VivianLiveCardRenderer on the rendering thread.
 *
 * Uses volatile fields for lock-free thread safety -- each field is
 * independently atomic (JVM guarantees for primitives and references).
 */
public class VivianState {
    private static final String TAG = "VivianState";

    // GPS
    public volatile float speedMph = 0f;
    public volatile float track = 0f;
    public volatile boolean hasFix = false;
    public volatile int satsUsed = 0;
    public volatile double lat = 0.0;
    public volatile double lon = 0.0;
    public volatile float alt = 0f;
    public volatile int speedLimit = -1;  // -1 = unknown

    // Music
    public volatile boolean isPlaying = false;
    public volatile String trackName = "";
    public volatile String artistName = "";
    public volatile long progressMs = 0;
    public volatile long durationMs = 0;
    public volatile long progressReceivedAt = 0;  // system time when progressMs was last set

    // Assistant
    public volatile String assistantState = "idle";
    public volatile String transcript = "";

    // Weather
    public volatile int tempF = 0;
    public volatile String condition = "";
    public volatile boolean hasWeather = false;

    // Sentry
    public volatile boolean sentryActive = false;

    // Map
    public volatile Bitmap mapBitmap = null;
    public volatile long mapFetchTs = 0;

    // Connection
    public volatile boolean connected = false;
    public volatile long lastUpdateTs = 0;

    /**
     * Get interpolated playback progress (smooth between server updates).
     */
    public long getInterpolatedProgressMs() {
        if (!isPlaying || progressReceivedAt == 0) return progressMs;
        long elapsed = System.currentTimeMillis() - progressReceivedAt;
        long interpolated = progressMs + elapsed;
        return Math.min(interpolated, durationMs);
    }

    /**
     * Parse a state broadcast from the Pi and update all fields.
     */
    public void updateFromJson(JSONObject json) {
        try {
            // GPS
            JSONObject gps = json.optJSONObject("gps");
            if (gps != null) {
                hasFix = gps.optBoolean("has_fix", false);
                satsUsed = gps.optInt("sats_used", 0);
                if (hasFix) {
                    speedMph = (float) gps.optDouble("speed_mph", 0.0);
                    track = (float) gps.optDouble("track", 0.0);
                    lat = gps.optDouble("lat", 0.0);
                    lon = gps.optDouble("lon", 0.0);
                    alt = (float) gps.optDouble("alt", 0.0);
                } else {
                    speedMph = 0f;
                }
            }

            // Music
            JSONObject music = json.optJSONObject("music");
            if (music != null) {
                isPlaying = music.optBoolean("is_playing", false);
                trackName = music.optString("track", "");
                artistName = music.optString("artist", "");
                long newProgress = music.optLong("progress_ms", 0);
                durationMs = music.optLong("duration_ms", 0);
                // Only update progress baseline when it actually changes
                if (Math.abs(newProgress - progressMs) > 1500 || !isPlaying) {
                    progressMs = newProgress;
                    progressReceivedAt = System.currentTimeMillis();
                }
            }

            // Assistant
            JSONObject assistant = json.optJSONObject("assistant");
            if (assistant != null) {
                assistantState = assistant.optString("state", "idle");
                transcript = assistant.optString("transcript", "");
            }

            // Weather
            JSONObject weather = json.optJSONObject("weather");
            if (weather != null) {
                tempF = weather.optInt("temp_f", 0);
                condition = weather.optString("condition", "");
                hasWeather = true;
            }

            // Speed limit
            if (json.isNull("speed_limit")) {
                speedLimit = -1;
            } else {
                speedLimit = json.optInt("speed_limit", -1);
            }

            // Sentry
            sentryActive = json.optBoolean("sentry_active", false);

            lastUpdateTs = System.currentTimeMillis();

        } catch (Exception e) {
            Log.e(TAG, "Error parsing state JSON", e);
        }
    }
}
