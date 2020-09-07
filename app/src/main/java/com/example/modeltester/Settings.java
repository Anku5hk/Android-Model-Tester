package com.example.modeltester;

import android.app.Activity;
import android.app.AlertDialog;
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
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.modeltester.Pytorch.PyClassifier;
import com.example.modeltester.Tensorflow.TfClassifier;
import com.example.modeltester.Tensorflow.TfDetector;

import java.io.InputStream;

import static android.content.ContentValues.TAG;

public class Settings extends AppCompatActivity {

    // selected task
    private String selected_task;
    // intent model
    public static final int status1 = 1;
    // intent labels
    public static final int status2 = 2;
    // if label file selected
    private Boolean IF_LABELS = false;
    // task type spinner options
    protected String[] task_type_arr;
    // model type array
    protected String[] model_type_arr;
    // model type
    public static String model_type;
    // mixed precision option
    protected String[] mixed_precision_options;
    // mixed precision option
    protected String[] avail_models_list;
    // mixed precision option selected
    public static String mixed_precision;
    // selected model
    public static String selected_model;
    // model uri
    public static Uri model_uri;
    // labels uri
    public static Uri labels_uri;
    // detect model lib
    private String model_lib;
    // for modellib
    public Drawable d;
    public static float[] img_norms;
    public static int post_mean;
    public static float post_std;
    public static int img_size = 224;

    // activity elements
    // model path textview
    private TextView textview_graph;
    // labels path to textview
    private TextView textview_labels;

    private EditText edit_img_mean;
    private EditText edit_img_std;
    private EditText edit_post_mean;
    private EditText edit_post_std;
    private EditText edit_img_size;

    private LinearLayout ln1;
    private LinearLayout ln2;
    private LinearLayout ln3;
    private LinearLayout ln4;
    private LinearLayout ln5;
    private LinearLayout ln6;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);
        Toolbar toolbar = findViewById(R.id.mytoolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        // choose model file
        Button choose_model = findViewById(R.id.md_choose);
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
        Button choose_label = findViewById(R.id.lb_choose);
        edit_img_mean = findViewById(R.id.edit_img_mean);
        edit_img_std = findViewById(R.id.edit_img_std);
        edit_post_mean = findViewById(R.id.edit_post_mean);
        edit_post_std = findViewById(R.id.edit_post_std);
        edit_img_size = findViewById(R.id.edit_img_size);
        img_norms = new float[2];

        // on start disable some views
        ln1 = findViewById(R.id.linearlayout42);
        ln1.setVisibility(LinearLayout.INVISIBLE);
        ln2 = findViewById(R.id.linearlayout7);
        ln2.setVisibility(LinearLayout.INVISIBLE);
        ln3 = findViewById(R.id.linearlayout81);
        ln3.setVisibility(LinearLayout.INVISIBLE);
        ln4 = findViewById(R.id.linearlayout82);
        ln4.setVisibility(LinearLayout.INVISIBLE);
        ln5 = findViewById(R.id.linearlayout62);
        ln5.setVisibility(LinearLayout.INVISIBLE);
        ln6 = findViewById(R.id.linearlayout08);
        ln6.setVisibility(LinearLayout.INVISIBLE);


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

        // choose task type(classification/object detection)
        task_type_arr = getResources().getStringArray(R.array.task_type);
        Spinner task_select = findViewById(R.id.select_task_type);
        task_select.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                selected_task = task_type_arr[i];

                if (selected_task.equals("Object Detection")){
                    ln1.setVisibility(LinearLayout.VISIBLE);
                    ln2.setVisibility(LinearLayout.INVISIBLE);
                    ln3.setVisibility(LinearLayout.VISIBLE);
                    ln4.setVisibility(LinearLayout.INVISIBLE);
                }
                else {
                    ln1.setVisibility(LinearLayout.INVISIBLE);
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });

        // choose model type(quant/float)
        model_type_arr = getResources().getStringArray(R.array.model_type);
        Spinner model_type_spinner = findViewById(R.id.spinner1);
        model_type_spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                model_type = model_type_arr[i];

                // set hint of edittext
                if (model_type.equals("Float")){
                    edit_img_mean.setHint("127.5f");
                    edit_img_std.setHint("127.5f");
                    edit_post_mean.setHint("Not Required");
                    edit_post_std.setHint("Not Required");

                    img_norms[0] = 127.5f;
                    img_norms[1] = 127.5f;
                }
                else{
                    edit_img_mean.setHint("Not Required");
                    edit_img_std.setHint("Not Required");
                    edit_post_mean.setHint("0");
                    edit_post_std.setHint("255");

                    post_mean = 0;
                    post_std = 255;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });

        // mixed precision
        mixed_precision_options = getResources().getStringArray(R.array.mixed_precision);
        Spinner mixed_precision_spinner = findViewById(R.id.spinner2);
        mixed_precision_spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                mixed_precision = mixed_precision_options[i];
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });

        // select model
        avail_models_list = getResources().getStringArray(R.array.model);
        Spinner select_model = findViewById(R.id.select_model);
        select_model.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                selected_model = avail_models_list[i];
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });

        // save button to launch activity
        Button save = findViewById(R.id.save);
        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_LABELS) {
                    if (selected_task.equals("Classification")){ selected_model = " ";}
                    try {
                        // get norms from edittext
                        if (model_type.equals("Float")) {
                            img_norms[0] = Float.valueOf(edit_img_mean.getText().toString());
                            img_norms[1] = Float.valueOf(edit_img_std.getText().toString());

                        } else {
                            post_mean = Integer.valueOf(edit_post_mean.getText().toString());
                            post_std = Float.parseFloat(edit_post_std.getText().toString());
                        }
                        img_size = Integer.parseInt(edit_img_size.getText().toString());
                    } catch (Exception e) { e.printStackTrace(); }

                    Intent i;
                    switch (selected_task + model_lib + selected_model) {
                        case "Classification" + "tflite" + " ":
                            i = new Intent(Settings.this, TfClassifier.class);
                            startActivity(i);
                            break;
                        case "Object Detection" + "tflite" + "Faster-RCNN":
                            i = new Intent(Settings.this, TfDetector.class);
                            startActivity(i);
                            break;
//                        case ("Object Detection" + "pt"):
//                            i = new Intent(Settings.this, .class);
//                            startActivity(i);
//                            break;
                        case ("Classification" + "pt" + " "):
                            i = new Intent(Settings.this, PyClassifier.class);
                            startActivity(i);
                            break;
                        default:
                            Toast.makeText(getApplicationContext(), "Invalid Model selected", Toast.LENGTH_SHORT).show();
                    }
                }
                else{
                    Toast.makeText(getApplicationContext(), "Label FIle Not Selected", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // cancel button
        Button cancel = findViewById(R.id.cancel);
        cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

        // help info button
        ImageButton help_button = findViewById(R.id.mixp_help);
        help_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                new AlertDialog.Builder(Settings.this)
                        .setTitle("Mixed Precision")
                        .setMessage("For FLOPS calculations use Float16 for multiplications and" +
                                " Float32 for addition. Currently supported using Tensorflow-2.1.0 above only.").create().show();
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
                    Log.d(TAG, "Model File Path: " + model_uri.getPath());
                    String[] pt = model_uri.getPath().split("/");
                    textview_graph.setText(pt[pt.length - 1]);
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
                    Log.d(TAG, "Label File Path: " +  labels_uri.getPath());
                    String[] pt = labels_uri.getPath().split("/");
                    textview_labels.setText(pt[pt.length - 1]);
                    IF_LABELS = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // returns model uri
    public static Uri return_model() { return model_uri; }
    // return labels uri
    public static Uri return_labels() { return labels_uri; }

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
        ImageView model_lib_img = findViewById(R.id.imageview);
        if (model_lib.equals("tflite")) {
            d = loadDrawableFromAssets("tf.png", getApplicationContext());
            model_lib_img.setImageDrawable(d);
            ln6.setVisibility(View.VISIBLE);
            ln2.setVisibility(View.VISIBLE);
            ln3.setVisibility(View.VISIBLE);
            ln4.setVisibility(View.VISIBLE);
            ln5.setVisibility(View.INVISIBLE);
        }
        if (model_lib.equals("pt")) {
            d = loadDrawableFromAssets("pyt.png", getApplicationContext());
            model_lib_img.setImageDrawable(d);
            ln2.setVisibility(View.INVISIBLE);
            ln3.setVisibility(View.INVISIBLE);
            ln4.setVisibility(View.INVISIBLE);
            ln5.setVisibility(View.VISIBLE);
        }
    }

}
