package com.example.modeltester.Pytorch;

import android.net.Uri;
import android.os.Bundle;

import com.example.modeltester.FileUtil;
import com.example.modeltester.Settings;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import com.example.modeltester.R;
import com.google.common.util.concurrent.ListenableFuture;

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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static android.content.ContentValues.TAG;

public class PyClassifierCamera extends AppCompatActivity {

    // top 3 results to show
    public static final int MAX_RESULTS_TO_SHOW = 3;
    // max labels in total
    public static final int MAX_LABEL_SIZE = 200;
    // image size to crop to
    private int IMAGE_SIZE = 224;
    // top 3 labels
    private String[] top_classes;
    // top 3 confidence
    private String[] top_prob;
    // scores from predict
    private float[] scores;
    // holds all classes names
    private ArrayList<String> all_classes;
    // module
    private Module module;
    //model uri
    private Uri model_uri = Settings.return_model();
    // labels uri
    private Uri labels_uri = Settings.return_labels();
    // read model file
    File filef;
    private PreviewView viewFinder;
    private TextView[] class_views = new TextView[3];
    private TextView[] prob_views = new TextView[3];
    boolean doubleBackToExitPressedOnce = false;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private final ExecutorService executor = Executors.newFixedThreadPool(1);

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
        setContentView(R.layout.activity_py_classifier_camera);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);
        viewFinder = findViewById(R.id.previewView);

        // initialize camera
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    cameraProvider.unbindAll();
                    bindPreview(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    Log.e(TAG, "Error: " + e);
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }
        }), ContextCompat.getMainExecutor(this));

        top_classes = new String[MAX_LABEL_SIZE];
        // initialize array to hold top probabilities
        top_prob = new String[MAX_LABEL_SIZE];

        // textviews
        class_views[0] = findViewById(R.id.label1);
        class_views[1] = findViewById(R.id.label2);
        class_views[2] = findViewById(R.id.label3);
        prob_views[0] = findViewById(R.id.prob1);
        prob_views[1] = findViewById(R.id.prob2);
        prob_views[2] = findViewById(R.id.prob3);

    }

    protected void onResume() {
        super.onResume();
        try {
            try {
                // uri to file for absolute path using FileUtil.class
                filef = FileUtil.from(PyClassifierCamera.this, model_uri);
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

    // back button exit or not
    @Override
    public void onBackPressed() {
        if (doubleBackToExitPressedOnce) {
            PyClassifierCamera.super.onBackPressed();
            finish();
        }
        this.doubleBackToExitPressedOnce = true;
        Toast.makeText(this, "Press BACK again to exit", Toast.LENGTH_SHORT).show();
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                doubleBackToExitPressedOnce=false;
            }
        }, 2000);
    }
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        // set preview for back camera
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(viewFinder.createSurfaceProvider());
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(IMAGE_SIZE, IMAGE_SIZE))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {

               int rotation =  image.getImageInfo().getRotationDegrees();
                float[] normMeanRGB = new float[] {0.485f, 0.456f, 0.406f};
                float[] normStdRGB =  new float[] {0.229f, 0.224f, 0.225f};

              Tensor inputTensor = TensorImageUtils
                      .imageYUV420CenterCropToFloat32Tensor(image.getImage(),
                      rotation, image.getWidth(),image.getHeight(), normMeanRGB,normStdRGB);
              image.close();
              runOnUiThread(new Runnable() {
                  @Override
                  public void run() {
                      try {
                          all_classes = load_labels();
                      }
                      catch (IOException e){
                          e.printStackTrace();
                      }
                      // running the model
                      final Tensor outputTensor = module.forward(IValue.from(inputTensor))
                              .toTensor();
                      scores = outputTensor.getDataAsFloatArray();
                      printTopKLabels();

                      // showing className on UI
                      class_views[0].setText(top_classes[2]);
                      class_views[1].setText(top_classes[1]);
                      class_views[2].setText(top_classes[0]);
                      // showing probability in percentage
                      prob_views[0].setText(top_prob[2]);
                      prob_views[1].setText(top_prob[1]);
                      prob_views[2].setText(top_prob[0]);
                  }
              });
            }
        });
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
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

}
