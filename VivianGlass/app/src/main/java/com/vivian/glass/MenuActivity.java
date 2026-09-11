package com.vivian.glass;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;

/**
 * Touchpad menu launched when the user taps the VIVIAN Live Card.
 *
 * Glass menu gestures:
 *   - Swipe forward/back to navigate items
 *   - Tap to select
 *   - Swipe down to dismiss
 */
public class MenuActivity extends Activity {

    private static final int MENU_PLAY_PAUSE = 1;
    private static final int MENU_NEXT = 2;
    private static final int MENU_PREV = 3;
    private static final int MENU_STOP = 4;
    private static final int MENU_SENTRY = 5;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        openOptionsMenu();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, MENU_PLAY_PAUSE, 0, "Play / Pause");
        menu.add(0, MENU_NEXT, 1, "Next Track");
        menu.add(0, MENU_PREV, 2, "Previous Track");
        menu.add(0, MENU_STOP, 3, "Stop HUD");

        // Dynamic sentry label based on current state
        VivianGlassService service = VivianGlassService.getInstance();
        boolean sentryOn = (service != null && service.getState() != null
                && service.getState().sentryActive);
        menu.add(0, MENU_SENTRY, 4, sentryOn ? "Disable Sentry" : "Enable Sentry");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        VivianGlassService service = VivianGlassService.getInstance();
        WebSocketManager ws = (service != null) ? service.getWebSocketManager() : null;

        switch (item.getItemId()) {
            case MENU_PLAY_PAUSE:
                if (ws != null) ws.sendCommand("music_toggle");
                break;
            case MENU_NEXT:
                if (ws != null) ws.sendCommand("music_next");
                break;
            case MENU_PREV:
                if (ws != null) ws.sendCommand("music_prev");
                break;
            case MENU_STOP:
                stopService(new Intent(this, VivianGlassService.class));
                break;
            case MENU_SENTRY:
                if (ws != null) ws.sendCommand("sentry_toggle");
                break;
        }

        finish();
        return true;
    }

    @Override
    public void onOptionsMenuClosed(Menu menu) {
        super.onOptionsMenuClosed(menu);
        finish();
    }
}
