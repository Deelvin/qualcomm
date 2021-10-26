package com.deelvin.openclrunner;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    static {
       System.loadLibrary("openclrunner");
    }

    private AssetManager mgr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mgr = getResources().getAssets();

        final Button runButton = findViewById(R.id.runButton);
        runButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                final TextView timeText = findViewById(R.id.timeView);
                timeText.setText(runOpenCL(mgr));
            }
        });
    }

    public native String runOpenCL(AssetManager mgr);
}