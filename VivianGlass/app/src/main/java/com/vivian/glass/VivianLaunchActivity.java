package com.vivian.glass;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;

/**
 * Thin launch activity that Glass's voice/touch launcher targets.
 * Starts the LiveCard service and immediately finishes.
 */
public class VivianLaunchActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        startService(new Intent(this, VivianGlassService.class));
        finish();
    }
}
