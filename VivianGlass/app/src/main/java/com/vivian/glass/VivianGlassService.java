package com.vivian.glass;

import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import com.google.android.glass.timeline.LiveCard;

/**
 * GDK Live Card service -- the entry point for the VIVIAN HUD.
 *
 * Launched via voice trigger ("OK Glass, show VIVIAN") or programmatically.
 * Publishes a Live Card with DirectRenderingCallback for the HUD display.
 * START_STICKY ensures the service restarts if killed by the system.
 */
public class VivianGlassService extends Service {
    private static final String TAG = "VivianService";
    private static final String LIVE_CARD_TAG = "vivian_hud";

    private static VivianGlassService instance;

    private LiveCard liveCard;
    private VivianState state;
    private WebSocketManager wsManager;
    private VivianLiveCardRenderer renderer;
    private PowerManager.WakeLock wakeLock;

    /**
     * Get the running service instance (for MenuActivity to access WebSocketManager).
     */
    public static VivianGlassService getInstance() {
        return instance;
    }

    public WebSocketManager getWebSocketManager() {
        return wsManager;
    }

    public VivianState getState() {
        return state;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        state = new VivianState();
        wsManager = new WebSocketManager(state);
        Log.i(TAG, "VivianGlassService created");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (liveCard == null || !liveCard.isPublished()) {
            liveCard = new LiveCard(this, LIVE_CARD_TAG);

            // Set up direct rendering (Canvas-based HUD)
            renderer = new VivianLiveCardRenderer(this, state);
            liveCard.setDirectRenderingEnabled(true);
            liveCard.getSurfaceHolder().addCallback(renderer);

            // Tap on Live Card opens the menu
            Intent menuIntent = new Intent(this, MenuActivity.class);
            liveCard.setAction(PendingIntent.getActivity(
                    this, 0, menuIntent, 0));

            liveCard.attach(this);
            liveCard.publish(LiveCard.PublishMode.REVEAL);

            // Start WebSocket connection to Pi
            wsManager.connect();

            // Keep Glass awake while HUD is active
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            wakeLock = pm.newWakeLock(
                    PowerManager.SCREEN_BRIGHT_WAKE_LOCK
                    | PowerManager.ACQUIRE_CAUSES_WAKEUP
                    | PowerManager.ON_AFTER_RELEASE,
                    "vivian:hud");
            wakeLock.acquire();

            Log.i(TAG, "Live Card published, WebSocket connecting");
        } else {
            liveCard.navigate();
        }

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (liveCard != null && liveCard.isPublished()) {
            liveCard.unpublish();
            liveCard = null;
        }
        if (wsManager != null) {
            wsManager.disconnect();
        }
        if (renderer != null) {
            renderer.stop();
        }
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        instance = null;
        Log.i(TAG, "VivianGlassService destroyed");
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
