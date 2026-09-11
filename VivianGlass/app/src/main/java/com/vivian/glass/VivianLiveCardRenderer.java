package com.vivian.glass;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.Log;
import android.view.SurfaceHolder;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * DirectRenderingCallback that draws the VIVIAN HUD on Glass at ~4 FPS.
 *
 * All drawing is white-on-black. On the Glass prism, black pixels are
 * transparent, so only lit content is visible -- ideal for a driving HUD.
 *
 * Layout (640x360):
 * +------------------------------------------------------------------+
 * | [GPS] [SENTRY] [ASSISTANT]                  72F Clear      12:45 |
 * |------------------------------------------------------------------|
 * |         45              N                                         |
 * |        MPH         W --+-- E         Hotel California             |
 * |                         S            Eagles                       |
 * |                        182           > 1:23 / 6:31               |
 * +------------------------------------------------------------------+
 */
public class VivianLiveCardRenderer implements SurfaceHolder.Callback {
    private static final String TAG = "VivianRenderer";
    private static final int FPS = 4;
    private static final long FRAME_INTERVAL_MS = 1000 / FPS;

    // Canvas dimensions (Glass native)
    private static final int W = 640;
    private static final int H = 360;

    // Layout zones
    private static final int STATUS_BAR_HEIGHT = 30;
    private static final float SPEED_CENTER_X = 110f;
    private static final float SPEED_CENTER_Y = 195f;
    private static final float NAV_LEFT = 230f;
    private static final float NAV_TOP = 55f;
    private static final float MUSIC_LEFT = 440f;
    private static final float MUSIC_TOP = 80f;

    // Map tile
    private static final String MAP_URL = "http://192.168.4.1:9101/map.jpg";
    private static final long MAP_FETCH_INTERVAL_MS = 2000;
    private volatile long lastMapFetch = 0;

    private final VivianState state;
    private final Context context;
    private SurfaceHolder holder;
    private volatile boolean running = false;
    private Thread renderThread;

    // Pre-allocated paints (avoid GC during draw)
    private final Paint speedPaint;
    private final Paint unitPaint;
    private final Paint labelPaint;
    private final Paint valuePaint;
    private final Paint smallPaint;
    private final Paint outlinePaint;
    private final Paint fillPaint;
    private final Paint dimPaint;

    // Reusable objects
    private final SimpleDateFormat timeFormat;
    private final RectF progressRect = new RectF();


    // Animation state
    private int dotFrame = 0;

    public VivianLiveCardRenderer(Context context, VivianState state) {
        this.context = context;
        this.state = state;
        this.timeFormat = new SimpleDateFormat("h:mm", Locale.US);

        // Speed: large bold white
        speedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        speedPaint.setColor(Color.WHITE);
        speedPaint.setTypeface(Typeface.DEFAULT_BOLD);
        speedPaint.setTextSize(80f);
        speedPaint.setTextAlign(Paint.Align.CENTER);

        // MPH label
        unitPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        unitPaint.setColor(Color.WHITE);
        unitPaint.setTypeface(Typeface.DEFAULT_BOLD);
        unitPaint.setTextSize(22f);
        unitPaint.setTextAlign(Paint.Align.CENTER);

        // General labels
        labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        labelPaint.setColor(Color.WHITE);
        labelPaint.setTextSize(20f);

        // Value text (music track, etc.)
        valuePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        valuePaint.setColor(Color.WHITE);
        valuePaint.setTypeface(Typeface.DEFAULT_BOLD);
        valuePaint.setTextSize(22f);

        // Small text (status bar)
        smallPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        smallPaint.setColor(Color.WHITE);
        smallPaint.setTextSize(16f);

        // Circle/line outlines
        outlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        outlinePaint.setColor(Color.WHITE);
        outlinePaint.setStyle(Paint.Style.STROKE);
        outlinePaint.setStrokeWidth(1.5f);

        // Filled shapes
        fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        fillPaint.setColor(Color.WHITE);
        fillPaint.setStyle(Paint.Style.FILL);

        // Dimmed text (50% white)
        dimPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        dimPaint.setColor(Color.argb(128, 255, 255, 255));
        dimPaint.setTextSize(16f);

        // Heading degrees text

    }

    @Override
    public void surfaceCreated(SurfaceHolder h) {
        this.holder = h;
        running = true;
        renderThread = new Thread(new Runnable() {
            @Override
            public void run() {
                renderLoop();
            }
        }, "hud-render");
        renderThread.start();
    }

    @Override
    public void surfaceChanged(SurfaceHolder h, int format, int width, int height) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder h) {
        stop();
    }

    public void stop() {
        running = false;
        if (renderThread != null) {
            try {
                renderThread.join(500);
            } catch (InterruptedException ignored) {
            }
        }
    }

    private void fetchMapIfNeeded() {
        long now = System.currentTimeMillis();
        if (now - lastMapFetch < MAP_FETCH_INTERVAL_MS) return;
        if (!state.hasFix || !state.connected) return;
        lastMapFetch = now;

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    HttpURLConnection conn =
                            (HttpURLConnection) new URL(MAP_URL).openConnection();
                    conn.setConnectTimeout(2000);
                    conn.setReadTimeout(2000);
                    try {
                        InputStream in = conn.getInputStream();
                        Bitmap bmp = BitmapFactory.decodeStream(in);
                        if (bmp != null) {
                            state.mapBitmap = bmp;
                            state.mapFetchTs = System.currentTimeMillis();
                        }
                    } finally {
                        conn.disconnect();
                    }
                } catch (Exception e) {
                    // Silent -- map just won't update this cycle
                }
            }
        }, "map-fetch").start();
    }

    private void renderLoop() {
        while (running) {
            long start = System.currentTimeMillis();
            fetchMapIfNeeded();
            Canvas canvas = null;
            try {
                canvas = holder.lockCanvas();
                if (canvas != null) {
                    drawHud(canvas);
                }
            } catch (Exception e) {
                Log.e(TAG, "Render error", e);
            } finally {
                if (canvas != null) {
                    try {
                        holder.unlockCanvasAndPost(canvas);
                    } catch (Exception ignored) {
                    }
                }
            }
            long elapsed = System.currentTimeMillis() - start;
            long sleep = FRAME_INTERVAL_MS - elapsed;
            if (sleep > 0) {
                try {
                    Thread.sleep(sleep);
                } catch (InterruptedException ignored) {
                }
            }
        }
    }

    private void drawHud(Canvas canvas) {
        canvas.drawColor(Color.BLACK);
        dotFrame++;

        if (!state.connected) {
            // If we had data recently, keep showing the HUD with a reconnecting indicator
            long staleSec = (System.currentTimeMillis() - state.lastUpdateTs) / 1000;
            if (state.lastUpdateTs > 0 && staleSec < 15) {
                drawStatusBar(canvas);
                drawSpeed(canvas);
                drawNav(canvas);
                drawMusic(canvas);
                // "Reconnecting..." in status bar area
                canvas.drawText("RECONNECTING", W / 2f, H - 8f, smallPaint);
                return;
            }
            drawDisconnected(canvas);
            return;
        }

        drawStatusBar(canvas);
        drawSpeed(canvas);
        drawNav(canvas);
        drawMusic(canvas);
    }

    // -------------------------------------------------------------------------
    // Status Bar (top 30px)
    // -------------------------------------------------------------------------

    private void drawStatusBar(Canvas canvas) {
        float y = 20f;

        // GPS fix indicator (filled dot = fix, outline = no fix)
        if (state.hasFix) {
            canvas.drawCircle(15f, y - 4f, 5f, fillPaint);
        } else {
            canvas.drawCircle(15f, y - 4f, 5f, outlinePaint);
        }
        String gpsLabel = state.hasFix ? (state.satsUsed + " SAT") : "NO GPS";
        canvas.drawText(gpsLabel, 26f, y, smallPaint);

        // Sentry badge
        if (state.sentryActive) {
            canvas.drawText("SENTRY", 100f, y, smallPaint);
        }

        // Weather + time (right-aligned)
        String timeStr = timeFormat.format(new Date());
        float timeWidth = smallPaint.measureText(timeStr);
        canvas.drawText(timeStr, W - timeWidth - 10f, y, smallPaint);

        if (state.hasWeather) {
            String wx = state.tempF + "\u00B0F " + state.condition;
            float wxWidth = smallPaint.measureText(wx);
            canvas.drawText(wx, W - timeWidth - wxWidth - 25f, y, smallPaint);
        }

        // Divider line
        canvas.drawLine(0, STATUS_BAR_HEIGHT, W, STATUS_BAR_HEIGHT, outlinePaint);
    }

    // -------------------------------------------------------------------------
    // Speed (left zone)
    // -------------------------------------------------------------------------

    private void drawSpeed(Canvas canvas) {
        String speedStr;
        if (state.hasFix) {
            speedStr = String.valueOf(Math.round(state.speedMph));
        } else {
            speedStr = "--";
        }

        canvas.drawText(speedStr, SPEED_CENTER_X, SPEED_CENTER_Y, speedPaint);
        canvas.drawText("MPH", SPEED_CENTER_X, SPEED_CENTER_Y + 30f, unitPaint);

        // Speed limit
        if (state.speedLimit > 0) {
            String limitStr = "LIMIT " + state.speedLimit;
            float lw = dimPaint.measureText(limitStr);
            canvas.drawText(limitStr, SPEED_CENTER_X - lw / 2f, SPEED_CENTER_Y + 58f, dimPaint);
        }
    }

    // -------------------------------------------------------------------------
    // Navigation (center zone)
    // -------------------------------------------------------------------------

    private void drawNav(Canvas canvas) {
        float x = NAV_LEFT;
        float y = NAV_TOP;

        if (!state.hasFix) {
            canvas.drawText("No GPS Fix", x, y + 60f, valuePaint);
            canvas.drawText("Waiting for signal...", x, y + 88f, smallPaint);
            return;
        }

        // Map tile (fetched from Pi HTTP endpoint)
        Bitmap map = state.mapBitmap;
        if (map != null && System.currentTimeMillis() - state.mapFetchTs < 60000) {
            canvas.drawBitmap(map, x, y, null);
        } else {
            canvas.drawText("Loading map...", x, y + 80f, smallPaint);
        }
    }

    // -------------------------------------------------------------------------
    // Music (right zone)
    // -------------------------------------------------------------------------

    private void drawMusic(Canvas canvas) {
        if (!state.isPlaying && (state.trackName == null || state.trackName.isEmpty())) {
            return;
        }

        float x = MUSIC_LEFT;
        float y = MUSIC_TOP;

        // Track name (truncate with ellipsis)
        String track = truncate(state.trackName, 18);
        canvas.drawText(track, x, y, valuePaint);

        // Artist
        String artist = truncate(state.artistName, 22);
        canvas.drawText(artist, x, y + 28f, labelPaint);

        // Play/pause indicator + time
        String icon = state.isPlaying ? "\u25B6" : "\u2759\u2759";
        long currentProgress = state.getInterpolatedProgressMs();
        String timeStr = formatMs(currentProgress) + " / " + formatMs(state.durationMs);
        canvas.drawText(icon + "  " + timeStr, x, y + 58f, smallPaint);

        // Progress bar
        float barX = x;
        float barY = y + 70f;
        float barW = 175f;
        float barH = 4f;

        // Background (dim)
        progressRect.set(barX, barY, barX + barW, barY + barH);
        canvas.drawRect(progressRect, outlinePaint);

        // Fill
        if (state.durationMs > 0) {
            float pct = (float) currentProgress / state.durationMs;
            progressRect.set(barX, barY, barX + barW * pct, barY + barH);
            canvas.drawRect(progressRect, fillPaint);
        }
    }


    // -------------------------------------------------------------------------
    // Disconnected Screen
    // -------------------------------------------------------------------------

    private void drawDisconnected(Canvas canvas) {
        // "VIVIAN" centered large
        canvas.drawText("VIVIAN", W / 2f, H / 2f - 20f, speedPaint);

        // "Connecting..." with animated dots
        int dots = (dotFrame / 2) % 4;
        StringBuilder sb = new StringBuilder("Connecting");
        for (int i = 0; i < dots; i++) sb.append('.');
        Paint connectPaint = new Paint(labelPaint);
        connectPaint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText(sb.toString(), W / 2f, H / 2f + 25f, connectPaint);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static String truncate(String s, int maxLen) {
        if (s == null) return "";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 1) + "\u2026";
    }

    private static String formatMs(long ms) {
        long totalSec = ms / 1000;
        long min = totalSec / 60;
        long sec = totalSec % 60;
        return min + ":" + String.format(Locale.US, "%02d", sec);
    }
}
