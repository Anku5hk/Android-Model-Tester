package com.example.modeltester.Tensorflow;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import static android.content.ContentValues.TAG;

import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.modeltester.Help;
import com.example.modeltester.R;
import com.example.modeltester.Settings;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;


public class TfClassifier extends AppCompatActivity {

    // holds all the possible classes for graph
    private List<String> classList;
    // array that holds the classes with the highest probabilities
    private String[] topClasses;
    // array that holds the highest probabilities
    private String[] topProb;
    // selected processor
    private String[] processor_select_arr = null;
    // tflite graph
    private Interpreter tflite = null;
    // floating action button status
    public static final int PICK_IMAGE = 1;
    // spinner for device type
    private boolean spinner_pick = false;
    // bool for imageview
    public boolean IF_IMAGE = false;
    // device
    public static String device = "GPU";
    // image dimensions for the graph
    private int SIZE_X;
    private int SIZE_Y;
    // normalize input image according to model(float only)
    private final float IMAGE_MEAN = Settings.img_norms[0];
    private final float IMAGE_STD = Settings.img_norms[1];
    // dequantization in the post-processing values(quant only)
    private final int POST_MEAN = Settings.post_mean;
    private final float POST_STD = Settings.post_std;

    // activity elements
    private ImageView display_imageview;
    private TextView[] class_views = new TextView[3];
    private TextView[] prob_views = new TextView[3];
    private TextView run_time_view;

    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();
    private String mixed_precision = Settings.mixed_precision;
    private String model_type = Settings.model_type; // quant or float

    // priority queue to hold top results
    private PriorityQueue<Map.Entry<String, Float>> sorted_classes_queue =
            new PriorityQueue<>(
                    3,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tf_classifier);
        Toolbar toolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        // textviews
        class_views[0] = findViewById(R.id.class1);
        class_views[1] = findViewById(R.id.class2);
        class_views[2] = findViewById(R.id.class3);
        prob_views[0] = findViewById(R.id.prob1);
        prob_views[1] = findViewById(R.id.prob2);
        prob_views[2] = findViewById(R.id.prob3);

        // initialize imageView that displays selected image to the user
        display_imageview = findViewById(R.id.selected_image);
        // model runtime display
        run_time_view = findViewById(R.id.time_elapsed);

        // initialize array to hold top labels
        topClasses = new String[3];
        // initialize array to hold top probabilities
        topProb = new String[3];

        // classify current displayed image
        Button classify_button = findViewById(R.id.classify);
        classify_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {
                    // get current bitmap from imageView
                    Bitmap bitmap_orig = ((BitmapDrawable) display_imageview.getDrawable()).getBitmap();
                    if (model_type.equals("Float")){

                        // preprocess for float
                        ImageProcessor imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                                        .build();

                        TensorImage tImage = new TensorImage(DataType.FLOAT32);
                        tImage.load(bitmap_orig);
                        tImage = imageProcessor.process(tImage);
                        TensorBuffer probabilityBuffer =
                                TensorBuffer.createFixedSize(new int[]{1,
                                        tflite.getOutputTensor(0).shape()[1]}, DataType.FLOAT32);

                        long tStart = System.currentTimeMillis();
                        tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
                        long tEnd = System.currentTimeMillis();
                        float total_t = tEnd - tStart;
                        run_time_view.setText(("Run Time: "+ total_t/1000 + " sec"));
                        show_results(probabilityBuffer);
                    }
                    else{

                        ImageProcessor imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                        .build();

                        TensorImage tImage = new TensorImage(DataType.UINT8);
                        tImage.load(bitmap_orig);
                        tImage = imageProcessor.process(tImage);
                        TensorBuffer probabilityBuffer =
                                TensorBuffer.createFixedSize(new int[]{1,
                                        tflite.getOutputTensor(0).shape()[1]}, DataType.UINT8);

                        long tStart = System.currentTimeMillis();
                        tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
                        long tEnd = System.currentTimeMillis();
                        float total_t = tEnd - tStart;
                        run_time_view.setText(("Run Time: "+ total_t/1000 + " sec"));
                        show_results(probabilityBuffer);
                    }

                } else {
                    Toast.makeText(getApplicationContext(), "Please select a image first", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // device selector spinner
        processor_select_arr = new String[2];
        processor_select_arr = getResources().getStringArray(R.array.processor_select);
        Spinner process_type = findViewById(R.id.select_processor_type);
        process_type.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                device = processor_select_arr[i];
                if (spinner_pick){
                    try {
                        if (tflite != null){
                            tflite.close();
                        }
                        load_tflite();
                    }
                    catch (Exception e){
                        e.printStackTrace();
                    }
                }
                spinner_pick = true;
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) { }
        });

        // floating button action to pick image
        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, PICK_IMAGE);
            }
        });

        // floating button action to launch camera
        FloatingActionButton fab2 = findViewById(R.id.fab2);
        fab2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // ask for permission
                if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) {
                    Intent intent = new Intent(TfClassifier.this, TfClassifierCamera.class);
                    release_resources();
                    startActivity(intent);
                }
                else {
                  //  Toast.makeText(getApplicationContext(), "Please give camera permission", Toast.LENGTH_SHORT).show();
                    ActivityCompat.requestPermissions(TfClassifier.this,
                            new String[] {Manifest.permission.CAMERA}, 0);
                }
            }
        });
    }

    protected void onResume() {
        super.onResume();
        if (tflite == null) {
            //initialize graph and labels
            try {
                load_tflite();
                classList = load_labels();
                // getting input array size from model
                Tensor ten = tflite.getInputTensor(0);
                int[] inp_arr = ten.shape();

                SIZE_X = inp_arr[1];
                SIZE_Y = inp_arr[2];

                make_logs(); // check if something goes wrong
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        MenuItem mv = menu.findItem(R.id.Settings_option);
        mv.setIcon(R.drawable.setting_tf);
        return true;
    }

    // display results on textviews
    private void show_results(TensorBuffer ten_buf) {

        TensorProcessor probabilityProcessor;
        if (model_type.equals("Float")) {
            probabilityProcessor =
                    new TensorProcessor.Builder().build();
        } else {
            probabilityProcessor =
                    new TensorProcessor.Builder().add(new NormalizeOp(POST_MEAN, POST_STD)).build();
        }
        if (null != classList) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(classList,
                    probabilityProcessor.process(ten_buf));

            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();

            for (String k : floatMap.keySet()) {
                sorted_classes_queue.add(
                        new AbstractMap.SimpleEntry<>(k, floatMap.get(k)));
                if (sorted_classes_queue.size() > 3) {
                    sorted_classes_queue.poll();
                }
            }

            // get top results from priority queue
            final int size = sorted_classes_queue.size();
            for (int a = 0; a < size; ++a) {
                Map.Entry<String, Float> label = sorted_classes_queue.poll();
                topClasses[a] = label.getKey();
                topProb[a] = String.format(Locale.ENGLISH, "%.0f%%", label.getValue() * 100);
            }

            for (int b = 0, j = 2; b < 3; b++, j--) {
                class_views[b].setText((b + 1 + ". " + topClasses[j]));
                prob_views[b].setText(topProb[j]);
            }
        }
    }

    // loads model
    private void load_tflite() {
                try {
                    switch (device) {
                        case "GPU":
                            GpuDelegate delegategpu = new GpuDelegate();
                            Interpreter.Options gpuoptions = (new Interpreter.Options()).addDelegate(delegategpu);
                            if (mixed_precision.equals("On")) {
                                gpuoptions.setAllowFp16PrecisionForFp32(true);
                            }
                            gpuoptions.setNumThreads(4);
                            tflite = new Interpreter(get_model_file(), gpuoptions);
                            delegategpu.close();
                            Log.d(TAG, "Running with " + device);
                            break;

                        case "NPU":
                            NnApiDelegate delegate = new NnApiDelegate();
                            Interpreter.Options npuoptions = (new Interpreter.Options()).addDelegate(delegate);
                            if (mixed_precision.equals("On")) {
                                npuoptions.setAllowFp16PrecisionForFp32(true);
                            }
                            npuoptions.setNumThreads(4);
                            npuoptions.setUseNNAPI(true);
                            tflite = new Interpreter(get_model_file(), npuoptions);
                            delegate.close();
                            Log.d(TAG, "Running with " + device);
                            break;

                        case "CPU":
                            tflite = new Interpreter(get_model_file(), new Interpreter.Options());
                            Log.d(TAG, "Running with " + device);
                            break;
                    }
                } catch (Exception ex) {
                    Log.d(TAG, "Error: " + ex);
                    Toast.makeText(getApplicationContext(), device + " is not supported on this device",
                            Toast.LENGTH_SHORT).show();
                }
            }

    // load model from /downloads
    private MappedByteBuffer get_model_file() throws IOException {

            FileInputStream inputStream = new FileInputStream(getApplicationContext()
                    .getContentResolver().openFileDescriptor(model_uri, "r").getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long start_position = fileChannel.position();
            long size = fileChannel.size();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, start_position, size);
    }

    // loads Label from labels file
    private List<String> load_labels() throws IOException {
        List<String> labelList = new ArrayList<String>();
        FileInputStream fp = new FileInputStream(getApplicationContext()
                .getContentResolver().openFileDescriptor(labels_uri, "r").getFileDescriptor());
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(fp));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // release resources clear views
    private void release_resources(){
        display_imageview.setImageDrawable(null);
        IF_IMAGE = false;
        for (int i=0,j=2; i<3;i++,j--){
            class_views[i].setText(null);
            prob_views[i].setText(null);
        }
        run_time_view.setText(null);
    }

    // check if something goes wrong
    private void make_logs(){
        Log.d(TAG, "Model num_of_classes "+ tflite.getOutputTensor(0).shape()[1]+
                " provided_classes_num " + classList.size());
        Log.d(TAG, "Model input image size "+SIZE_X+" " +SIZE_Y);
        Log.d(TAG, "Model input dtype "+tflite.getInputTensor(0).dataType());
        Log.d(TAG, "Model output dtype "+tflite.getOutputTensor(0).dataType());
    }

    // back button exit or not
    @Override
    public void onBackPressed() {
        new AlertDialog.Builder(this)
                .setTitle("Exit")
                .setMessage("Are you sure you want to exit?")
                .setNegativeButton(android.R.string.no, null)
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        TfClassifier.super.onBackPressed();
                        finishAffinity();
                    }
                }).create().show();
    }

    // actions for toolbar
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.Settings_option:
                Intent se = new Intent(TfClassifier.this, Settings.class);
                startActivity(se);
                return true;

            case R.id.Help_option:
                Intent hp = new Intent(TfClassifier.this, Help.class);
                startActivity(hp);
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    // set image on the imageview
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE) {
            IF_IMAGE = true;
            Uri uri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                display_imageview.setImageBitmap(bitmap);
                display_imageview.setRotation(display_imageview.getRotation());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
