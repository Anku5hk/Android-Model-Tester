package com.example.modeltester.Tensorflow;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;

import com.example.modeltester.YuvToRgbConverter;
import com.example.modeltester.Settings;

import static android.content.ContentValues.TAG;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import com.example.modeltester.R;

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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class TfClassifierCamera extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView viewFinder;
    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();
    private String model_type = Settings.model_type;
    private String mixed_precision = Settings.mixed_precision;
    public String device = TfClassifier.device; // default gpu
    // image dimensions for the graph
    private int SIZE_X;
    private int SIZE_Y;
    // tflite graph
    private Interpreter tflite;
    // holds all the possible classes for graph
    private List<String> classList;
    private String[] topClasses;
    // array that holds the highest probabilities
    private String[] topProb = null;
    private TextView[] class_views = new TextView[3];
    private TextView[] prob_views = new TextView[3];
    boolean doubleBackToExitPressedOnce = false;

    // normalize input image according to model(float only)
    private final float IMAGE_MEAN = Settings.img_norms[0];
    private final float IMAGE_STD = Settings.img_norms[1];
    // dequantization in the post-processing values(quant only)
    private final int POST_MEAN = Settings.post_mean;
    private final float POST_STD = Settings.post_std;


    private final ExecutorService executor = Executors.newFixedThreadPool(1);

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
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tf_classifier_camera);
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

        // initialize array to hold top labels
        topClasses = new String[3];
        // initialize array to hold top probabilities
        topProb = new String[3];

        // textviews
        class_views[0] = findViewById(R.id.label1);
        class_views[1] = findViewById(R.id.label2);
        class_views[2] = findViewById(R.id.label3);
        prob_views[0] = findViewById(R.id.prob1);
        prob_views[1] = findViewById(R.id.prob2);
        prob_views[2] = findViewById(R.id.prob3);
    }

    @Override
    protected void onResume() {
        super.onResume();

        //initialize graph and labels
        try {
            load_tflite();
            classList = load_labels();
        } catch (Exception e) {
            e.printStackTrace();
            TfClassifierCamera.super.onBackPressed();
            finish();
        }
        // getting input array size from model
        Tensor ten = tflite.getInputTensor(0);
        int[] inp_arr = ten.shape();
        SIZE_X = inp_arr[1];
        SIZE_Y = inp_arr[2];
    }

    // back button exit or not
    @Override
    public void onBackPressed() {
        if (doubleBackToExitPressedOnce) {
            TfClassifierCamera.super.onBackPressed();
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

    // camera function
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        // set preview for back camera
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(viewFinder.createSurfaceProvider());

        // imageAnalysis for inference
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {

                YuvToRgbConverter yuvtorgb = new YuvToRgbConverter(getApplicationContext());

                // create empty bitmap
                Bitmap bitmap = Bitmap.createBitmap(image.getWidth(),
                        image.getHeight(), Bitmap.Config.ARGB_8888 );
                // yuv to rgb image bitmap
                yuvtorgb.yuvToRgb(image.getImage(), bitmap);
                image.close();

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (model_type.equals("Float")){
                            // for float input, also normalize the input

                            ImageProcessor imageProcessor =
                                    new ImageProcessor.Builder()
                                            .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                            .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                                            .build();

                            TensorImage tImage = new TensorImage(DataType.FLOAT32);
                            tImage.load(bitmap);
                            tImage = imageProcessor.process(tImage);
                            TensorBuffer OutputBuffer =
                                    TensorBuffer.createFixedSize(new int[]{1,
                                            tflite.getOutputTensor(0).shape()[1]}, DataType.FLOAT32);

                            tflite.run(tImage.getBuffer(), OutputBuffer.getBuffer());
                            show_results(OutputBuffer);

                        }
                        else{
                            ImageProcessor imageProcessor =
                                    new ImageProcessor.Builder()
                                            .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                            .build();

                            TensorImage tImage = new TensorImage(DataType.UINT8);
                            tImage.load(bitmap);
                            tImage = imageProcessor.process(tImage);
                            TensorBuffer OutputBuffer =
                                    TensorBuffer.createFixedSize(new int[]{1,
                                            tflite.getOutputTensor(0).shape()[1]}, DataType.UINT8);
                            tflite.run(tImage.getBuffer(), OutputBuffer.getBuffer());
                            show_results(OutputBuffer);
                        }
                    }
                });
            }
        });
        // create camera lifecycle
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }


    // get model
    private MappedByteBuffer get_model_file() throws IOException {

        FileInputStream inputStream = new FileInputStream(getApplicationContext()
                .getContentResolver().openFileDescriptor(model_uri, "r").getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long start_position = fileChannel.position();
        long size = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, start_position, size);
    }

    // loads Label from location
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

    // show top results on the textviews
    private void show_results(TensorBuffer ten_buf) {

        TensorProcessor probabilityProcessor;
        if (model_type.equals("Float")) {
            probabilityProcessor =
                    new TensorProcessor.Builder().build();
        } else {
            // post process for quant models
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

    // load model with options
    private void load_tflite() {
        try {
            switch (device) {
                case "NPU":
                    NnApiDelegate npudelegate = new NnApiDelegate();
                    Interpreter.Options npuoptions = (new Interpreter.Options()).addDelegate(npudelegate);
                    npuoptions.setNumThreads(4);
                    if (mixed_precision.equals("On")) {
                        npuoptions.setAllowFp16PrecisionForFp32(true);
                    }
                    tflite = new Interpreter(get_model_file(), npuoptions);
                    Log.d(TAG, "Running with: " + device);
                    break;
                case "GPU":
                    GpuDelegate delegate = new GpuDelegate();
                    Interpreter.Options gpuoptions = (new Interpreter.Options()).addDelegate(delegate);
                    if (mixed_precision.equals("On")) {
                        gpuoptions.setAllowFp16PrecisionForFp32(true);
                    }
                    gpuoptions.setNumThreads(4);
                    tflite = new Interpreter(get_model_file(), gpuoptions);
                    Log.d(TAG, "Running with: " + device);
                    break;
                case "CPU":
                    tflite = new Interpreter(get_model_file(), new Interpreter.Options());
                    Log.d(TAG, "Running with: " + device);
                    break;
            }
        } catch (IOException ex) {
            Log.d(TAG, "Error: " + ex);
            Toast.makeText(getApplicationContext(), device + " is not supported on this device, Try something else.",
                    Toast.LENGTH_SHORT).show();
        }
    }
}


