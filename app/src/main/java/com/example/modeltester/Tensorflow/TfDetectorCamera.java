package com.example.modeltester.Tensorflow;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;

import com.example.modeltester.Settings;
import com.example.modeltester.YuvToRgbConverter;

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

import android.util.Log;
import android.util.Size;
import android.widget.ImageView;

import com.example.modeltester.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static android.content.ContentValues.TAG;

public class TfDetectorCamera extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView viewFinder;
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    // image size
    private int SIZE_X;
    private int SIZE_Y;
    private int NUM_OF_BOXES = 1;
    private float THRESHOLD = 0.055f;
    // offset value
    private int labelOffset = 1;
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // output classes array[batch size, number of classes]
    private float[][] outputClasses;
    // output boxes array[batch size,number of detection, numofboxes]
    private float[][][] outputBoxes;
    // output prediction probability[batchsize, num of detection]
    private float[][] outputScores;
    // boxes location
    private int[] boxes;
    // map to hold all detection values
    private Map<Integer, Object> outputMap;
    // rect box
    private RectF[] detection;
    // get model and label uri
    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();

    // activity elements
    private ImageView imageView;

    private int[] boxes_colors = new int[3];
    private Paint paint = new Paint();
    private Canvas canvas;
    YuvToRgbConverter yuvtorgb;
    private TensorImage tImage;
    float[] boxes_xs;
    float[] boxes_ys;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tf_detector_camera);
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

        //initialize graph and labels
        try {
            GpuDelegate delegate = new GpuDelegate();
            Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
            tflite = new Interpreter(get_model_file(), options);
            labelList = load_labels();
            // getting input array size from model
            Tensor ten = tflite.getInputTensor(0);
            int[] inp_arr = ten.shape();
            SIZE_X = inp_arr[1];
            SIZE_Y = inp_arr[2];
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // boxes colors
        boxes_colors[0] = Color.RED;
        boxes_colors[1] = Color.GREEN;
        boxes_colors[2] = Color.BLUE;

        imageView = findViewById(R.id.imageview);

        int output_tensor_size = tflite.getOutputTensor(0).shape()[1];

        // initialize output arrays
        outputBoxes = new float[1][output_tensor_size][4];
        outputClasses = new float[1][output_tensor_size];
        outputScores = new float[1][output_tensor_size];
        float[] numDetection = new float[1];
        detection = new RectF[3];

        // map for containing all output arrays
        outputMap = new HashMap<>();
        outputMap.put(0, outputBoxes);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetection);
        // boxes
        boxes = new int[4];
        boxes_xs = new float[output_tensor_size];
        boxes_ys = new float[output_tensor_size];
        tImage = new TensorImage(DataType.UINT8);
    }

    // back button exit or not
    @Override
    public void onBackPressed() {
        TfDetectorCamera.super.onBackPressed();
        finish();
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

                yuvtorgb = new YuvToRgbConverter(getApplicationContext());
                // create empty bitmap
                Bitmap bitmap = Bitmap.createBitmap(image.getWidth(),
                        image.getHeight(), Bitmap.Config.ARGB_8888);
                // yuv to rgb image bitmap
                yuvtorgb.yuvToRgb(image.getImage(), bitmap);
                image.close();

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {

                        ImageProcessor imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                        .build();

                        tImage.load(bitmap);
                        tImage = imageProcessor.process(tImage);

                        // create input object
                        Object[] inputArray = {tImage.getBuffer()};
                        tflite.runForMultipleInputsOutputs(inputArray, outputMap);

                        // loop around top 3 boxes
                        for (int i = 0; i < NUM_OF_BOXES; i++) {
                            // fill RectF with boxes location
                            boxes = boxes_location(outputBoxes[0][i][0], outputBoxes[0][i][1],
                                    outputBoxes[0][i][2], outputBoxes[0][i][3]);

                            /// for putting texts
                            boxes_xs[i] = boxes[1];
                            boxes_ys[i] = boxes[0];

                            detection[i] = new RectF(
                                    boxes[1], // left
                                    boxes[0], // top
                                    boxes[3], // right
                                    boxes[2]  // bottom
                            );
                        }
                        if (outputBoxes[0] != null) {
                            Bitmap bt = Bitmap.createBitmap(tImage.getWidth(), tImage.getHeight(), Bitmap.Config.ARGB_8888);
                            draw_boxes(bt);
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

    // get boxes locations in 0-255 range(because model outputs float values from 0 to 1)
    private int[] boxes_location(float OldValue1, float OldValue2, float OldValue3, float OldValue4) {
        int[] arr = new int[4];

        int OldMin = 0;
        int OldMax = 1;
        int NewMin = 0;
        int NewMax = SIZE_X;

        int OldRange = (OldMax - OldMin);
        int NewRange = (NewMax - NewMin);
        arr[0] = (int) ((((OldValue1 - OldMin) * NewRange) / OldRange) + NewMin);  //top
        arr[1] = (int) ((((OldValue2 - OldMin) * NewRange) / OldRange) + NewMin);  // left
        arr[2] = (int) ((((OldValue3 - OldMin) * NewRange) / OldRange) + NewMin); // bottom
        arr[3] = (int) ((((OldValue4 - OldMin) * NewRange) / OldRange) + NewMin); // right

        return arr;
    }

    // takes bitmap and draws boxes on them
    public void draw_boxes(Bitmap image){

        canvas = new Canvas(image); // new canvas
        paint.setStyle(Paint.Style.STROKE);
        // show boxes result
        for (int i=0; i < NUM_OF_BOXES; i++){
            if (detection[i] != null && outputScores[0][i] > THRESHOLD) {
                // display draw boxes and add texts
                String text = labelList.get((int) outputClasses[0][i] + labelOffset)+
                        " "+ outputScores[0][i];
                canvas.drawText(text, boxes_xs[i], boxes_ys[i],paint);
                paint.setColor(boxes_colors[i]);
                canvas.drawRect(detection[i], paint);
            }
        }
        imageView.setImageBitmap(image);
    }

}
