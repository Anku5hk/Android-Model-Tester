package com.example.modeltester;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;

import java.util.Map;
import java.util.PriorityQueue;

public class PyClassifier extends AppCompatActivity {
    // floating action button status
    public static final int PICK_IMAGE = 1;
    // top 3 results to show
    public static final int MAX_RESULTS_TO_SHOW = 3;
    // max labels in total
    public static final int MAX_LABEL_SIZE = 200;
    // image size to crop to
    private int IMAGE_SIZE = 224;
    // top 3 labels
    private String[] top_classes = null;
    // top 3 confidence
    private String[] top_prob = null;
    // scores from predict
    private float[] scores = null;
    // bool for imageview
    public boolean IF_IMAGE = false;
    // holds all classes names
    private ArrayList<String> all_classes;
    // bitmap image
    Bitmap bitmap = null;
    // module
    private Module module = null;
    //model uri
    private Uri model_uri = Settings.return_model();
    // labels uri
    private Uri labels_uri = Settings.return_labels();
      // image uri
    private Uri img_uri = null;
    // read model file
    File filef = null;
    // fab image show
    private ImageView display_imageview;
    // button classify
    private Button classify;

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sorted_classes_queue =
            new PriorityQueue<>(
                    MAX_LABEL_SIZE,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_py_classifier);
        Toolbar toolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        display_imageview = findViewById(R.id.selected_image);
        classify = findViewById(R.id.classify);

        top_classes = new String[MAX_LABEL_SIZE];
        // initialize array to hold top probabilities
        top_prob = new String[MAX_LABEL_SIZE];

        classify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {
                    // preparing input tensor
                    Bitmap img_bitmap = null;
                    try {
                        // load image to bitmap
                        FileInputStream file = new FileInputStream(getApplicationContext().getContentResolver()
                                .openFileDescriptor(img_uri, "r").getFileDescriptor());
                        img_bitmap = BitmapFactory.decodeStream(file);
                        // load labels from selected labels
                        all_classes = load_labels();
                    } catch (IOException e) {
                        Log.e("model error", "" + e);
                    }
                    img_bitmap = Bitmap.createScaledBitmap(img_bitmap, IMAGE_SIZE, IMAGE_SIZE, false);
                    // pre-process img and convert to tensor
                    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(img_bitmap,
                            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                    // count time elapsed(start time)
                    long tStart = System.currentTimeMillis();
                    // running the model
                    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                    // getting tensor content as java array of floats
                    scores = outputTensor.getDataAsFloatArray();
                    // end time
                    long tEnd = System.currentTimeMillis();
                    // time total
                    double tRes = tEnd - tStart;
                    printTopKLabels();

                    // showing className on UI
                    TextView class_view1 = findViewById(R.id.class1);
                    class_view1.setText(top_classes[2]);
                    TextView class_view2 = findViewById(R.id.class2);
                    class_view2.setText(top_classes[1]);
                    TextView class_view3 = findViewById(R.id.class3);
                    class_view3.setText(top_classes[0]);
                    // showing probability in percentage
                    TextView prob_view1 = findViewById(R.id.prob1);
                    TextView prob_view2 = findViewById(R.id.prob2);
                    TextView prob_view3 = findViewById(R.id.prob3);
                    prob_view1.setText(top_prob[2]);
                    prob_view2.setText(top_prob[1]);
                    prob_view3.setText(top_prob[0]);
                    // print runtime
                    TextView elapsed_time = findViewById(R.id.time_elapsed);
                    elapsed_time.setText(("Run Time: "+ tRes/1000));


                } else {
                    Toast.makeText(getApplicationContext(), "Please select a image first", Toast.LENGTH_SHORT).show();
                }
            }
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
                        PyClassifier.super.onBackPressed();
                        finishAffinity();
                    }
                }).create().show();
    }

    protected void onResume() {
        super.onResume();
        try {
            try {
                // uri to file for absolute path using FileUtil.class
                filef = FileUtil.from(PyClassifier.this, model_uri);
                Log.d("file", "File...:::: uti - " + filef.getPath() + " file -" + filef + " : " + filef.exists());

            } catch (IOException e) {
                e.printStackTrace();
                Log.e("graph error: ", "" + e);
            }
            // getting the path for model
            module = Module.load(filef.getAbsolutePath());
            //    labelList = load_labels();
            Log.d("Graph loaded", "Okay...........");
        } catch (Exception e) {
            Log.d("Graph not loaded", "Error reading assets", e);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        MenuItem mv = menu.findItem(R.id.Settings_option);
        mv.setIcon(R.drawable.setting_pt);
        return true;
    }


    // loads classes file
    private ArrayList<String> load_labels() throws IOException {
        ArrayList<String> arr_list = new ArrayList<>();
        FileInputStream fp = new FileInputStream(getApplicationContext()
                .getContentResolver()
                .openFileDescriptor(labels_uri, "r").getFileDescriptor());
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(fp));
        String line;
        while ((line = reader.readLine()) != null) {
            arr_list.add(line);
        }
        reader.close();
        return arr_list;
    }

    // print the top labels and respective confidences
    private void printTopKLabels() {
        // add all results to priority queue
        for (int i = 0; i < scores.length; ++i) {
            sorted_classes_queue.add(
                    new AbstractMap.SimpleEntry<>(all_classes.get(i), scores[i]));
            if (sorted_classes_queue.size() > MAX_RESULTS_TO_SHOW) {
                sorted_classes_queue.poll();
            }
        }
        Log.d( "SIZe..... ", "" + sorted_classes_queue.size());
        // add top 3 results in top_classes and confidence in top_prob
        for (int i = 0; i < 3; i++) {
            Map.Entry<String, Float> label = sorted_classes_queue.poll();
            top_classes[i] = label.getKey();
            top_prob[i] = String.format("%.0f%%", label.getValue());
        }
    }

    // actions for toolbar
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.Settings_option:
                Intent se = new Intent(PyClassifier.this, Settings.class);
                startActivity(se);
                return true;

            case R.id.Help_option:
                Intent hp = new Intent(PyClassifier.this, Help.class);
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
            img_uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), img_uri);
                display_imageview.setImageBitmap(bitmap);
                display_imageview.setRotation(display_imageview.getRotation());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}