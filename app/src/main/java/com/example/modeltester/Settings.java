package com.example.modeltester;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Log;
import android.view.View;
import android.webkit.MimeTypeMap;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.InputStream;

import static android.content.ContentValues.TAG;


public class Settings extends AppCompatActivity {
    // choose model button
    private Button choose_model;
    // choose label button
    private Button choose_label;
    // model task
    private Spinner task_select;
    // selected tsak
    private String selected_task;
    // save button
    private Button save;
    // cancel button
    private Button cancel;
    // intent model
    public static final int status1 = 1;
    // intent labels
    public static final int status2 = 2;
    // model path textview
    private TextView textview_graph;
    // labels path to textview
    private TextView textview_labels;
    // model lib image view
    private ImageView model_lib_img;
    // if label file selected
    private Boolean IF_IMAGE = false;
    // task type spinner options
    protected String[] task_type_arr;
    // model uri
    public static Uri model_uri;
    // labels uri
    public static Uri labels_uri;
    // detect model lib
    private String model_lib;
    // for modellib
    public Drawable d;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);
        Toolbar toolbar = findViewById(R.id.mytoolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        // choose model file
        choose_model = findViewById(R.id.md_choose);
        choose_model.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent md_intent = new Intent();
                md_intent.setType("*/*");
                md_intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(md_intent, status1);
            }
        });

        // choose labels file
        textview_graph = findViewById(R.id.model_name_textview);
        choose_label = findViewById(R.id.lb_choose);
        textview_labels = findViewById(R.id.labels_name_textview);
        choose_label.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent lb_intent = new Intent();
                lb_intent.setType("*/*");
                lb_intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(lb_intent, status2);
            }
        });

        // choose task type
        task_type_arr = getResources().getStringArray(R.array.task_type);
        task_select = findViewById(R.id.select_task_type);
        task_select.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                selected_task = task_type_arr[i];
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        // given inputs

        // model type selected

        // call method for saving selection to json(save button)
        save = findViewById(R.id.save);
        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {

                    // tensorflow models activity select
                    if (selected_task.equals("Classification") && model_lib.equals("tflite")) {
                        Intent i = new Intent(Settings.this, TfClassifier.class);
                        startActivity(i);
                        Toast.makeText(getApplicationContext(), "Settings Saved", Toast.LENGTH_SHORT).show();
                    } else if (selected_task.equals("Object Detection") && model_lib.equals("tflite")) {
                        Intent i = new Intent(Settings.this, TfDetector.class);
                        startActivity(i);
                        Toast.makeText(getApplicationContext(), "Settings Saved", Toast.LENGTH_SHORT).show();
                    }

                    // pytorch models activity select
                    else if (selected_task.equals("Classification") && model_lib.equals("pt")) {
                        Intent i = new Intent(Settings.this, PyClassifier.class);
                        startActivity(i);
                        Toast.makeText(getApplicationContext(), "Settings Saved", Toast.LENGTH_SHORT).show();
                    } else if (selected_task.equals("Object Detection") && model_lib.equals("pt")) {
                        Intent i = new Intent(Settings.this, TfDetector.class);
                        startActivity(i);
                        Toast.makeText(getApplicationContext(), "Settings Saved", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(getApplicationContext(), "Invalid Model selected", Toast.LENGTH_SHORT).show();
                    }
                }
                else{
                    Toast.makeText(getApplicationContext(), "Label FIle Not Selected", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // cancel button
        cancel = findViewById(R.id.cancel);
        cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });


    } // oncreate close

    // get model uri file path
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent result_data) {
        super.onActivityResult(requestCode,requestCode,result_data);
        if (requestCode == status1 && resultCode == Activity.RESULT_OK) {

            try {
                if (result_data != null) {
                    model_uri = result_data.getData();
                    // log file name
                    model_lib = return_model_lib(model_uri);
                    String path = model_uri.getPath();
                    Log.d(TAG, "Model File Path: " + path);
                    textview_graph.setText(path);
                    display_model_libimg();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        // get label file path
        if (requestCode == status2 && resultCode == Activity.RESULT_OK) {

            try {
                if (result_data != null) {
                    labels_uri = result_data.getData();
                    // log file name
                    String path = labels_uri.getPath();
                    Log.d(TAG, "Label File Path: " + path);
                    textview_labels.setText(path);
                    IF_IMAGE = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }


    // returns model uri
    public static Uri return_model() {
        return model_uri;
    }

    // return labels uri
    public static Uri return_labels() {
        return labels_uri;

    }

    // get model lib by returning file extension
    public String return_model_lib(Uri uri){
        // eg /document/0CFE-0E04:model.tflite to tflite
        return MimeTypeMap.getFileExtensionFromUrl(uri.toString());
    }

        // close settings
        @Override
        public void onBackPressed () {
            finish();
        }

   // pick modelib image from asset
    public Drawable loadDrawableFromAssets(String path, Context context)
    {
        InputStream stream = null;
        try
        {
            stream = context.getAssets().open(path);
            return Drawable.createFromStream(stream, null);
        }
        catch (Exception ignored) {} finally
        {
            try
            {
                if(stream != null)
                {
                    stream.close();
                }
            } catch (Exception ignored) {}
        }
        return null;
    }

    // display the image on correct model selection
    public void display_model_libimg() {
        // set image of model lib
        model_lib_img = findViewById(R.id.imageview);
        if (model_lib.equals("tflite")) {
            d = loadDrawableFromAssets("tf.png", getApplicationContext());
            model_lib_img.setImageDrawable(d);
        }
        if (model_lib.equals("pt")) {
            d = loadDrawableFromAssets("pyt.png", getApplicationContext());
            model_lib_img.setImageDrawable(d);
        }
    }

}
