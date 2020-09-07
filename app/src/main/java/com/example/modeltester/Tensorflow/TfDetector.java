package com.example.modeltester.Tensorflow;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
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

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.example.modeltester.Help;
import com.example.modeltester.R;
import com.example.modeltester.Settings;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
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

import static android.content.ContentValues.TAG;

public class TfDetector extends AppCompatActivity {
    // floating action button status
    public static final int IF_IMAGE_PICK = 1;
    // if image boolean
    private Boolean IF_IMAGE = false;
    // image size
    private int SIZE_X;
    private int SIZE_Y;
    private int CUR_NUM_OF_BOXES = 1;
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
    // image bitmap main
    private Bitmap selected_bitmap;
    // tensorimage
    private TensorImage tImage;
    private ImageProcessor imageProcessor;
    // get model and label uri
    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();
    private String model_input_type = Settings.model_type; // quant or float
    private final float IMAGE_MEAN = Settings.img_norms[0];
    private final float IMAGE_STD = Settings.img_norms[1];

    // activity elements
    private ImageView displayed_image;
    private TextView[] labels_view= new TextView[3];
    private TextView[] probs_view= new TextView[3];
    private Paint paint;
    private TextView cur_res;
    private TextView time_elapsed;

    private int[] boxes_colors = new int[3];

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tf_detector);
        Toolbar toolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        //initialize graph and labels
        try {
            GpuDelegate delegate = new GpuDelegate();
            Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
            tflite = new Interpreter(load_model_file(), options);
            labelList = load_labels();
            // getting input array size from model
            Tensor ten = tflite.getInputTensor(0);
            int[] inp_arr = ten.shape();
            SIZE_X = inp_arr[1];
            SIZE_Y = inp_arr[2];
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        displayed_image = findViewById(R.id.selected_image);
        // boxes colors
        boxes_colors[0] = Color.RED;
        boxes_colors[1] = Color.GREEN;
        boxes_colors[2] = Color.BLUE;
        // initialize elements
        Button add_num_of_boxes = findViewById(R.id.add_res);
        Button rem_num_of_boxes = findViewById(R.id.rem_res);

        // labels that hold top three results of CNN
        labels_view[0] = findViewById(R.id.class1);
        labels_view[1] = findViewById(R.id.class2);
        labels_view[2] = findViewById(R.id.class3);
        // displays the probabilities of top labels
        probs_view[0] = findViewById(R.id.prob1);
        probs_view[1] = findViewById(R.id.prob2);
        probs_view[2] = findViewById(R.id.prob3);

        // model runtime display
        time_elapsed = findViewById(R.id.time_elapsed);
        cur_res = findViewById(R.id.res_text);

        int output_tensor_size = tflite.getOutputTensor(0).shape()[1];
        // initialize output arrays
        outputBoxes = new float[1][output_tensor_size][4];
        outputClasses = new float[1][output_tensor_size];
        outputScores = new float[1][output_tensor_size];
        float[] numDetection = new float[1];
        detection = new RectF[3];

        // map for containing all output arrays
        outputMap =  new HashMap<>();
        outputMap.put(0, outputBoxes);
        outputMap.put(1,outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetection);
        // boxes
        boxes = new int[4];

        paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);

        make_logs(); // to check inputs
        // detect button
        Button detect = findViewById(R.id.detect);
        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {
                    // Image processor to resize input image(for float32 input)
                    if (model_input_type.equals("Float")){
                         imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                                        .build();
                         tImage = new TensorImage(DataType.FLOAT32);
                    }
                    else {
                        // for quant use uint8
                         imageProcessor =
                                new ImageProcessor.Builder()
                                        .add(new ResizeOp(SIZE_X, SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
                                        .build();
                        tImage = new TensorImage(DataType.UINT8); }

                    tImage.load(selected_bitmap);
                    tImage = imageProcessor.process(tImage);

                    // create input object
                    Object[] inputArray = {tImage.getBuffer()};
                    // count time elapsed(start time)
                    long tStart = System.currentTimeMillis();
                    // run the model
                    tflite.runForMultipleInputsOutputs(inputArray, outputMap);
                    // end time
                    long tEnd = System.currentTimeMillis();
                    // time total
                    double tRes = tEnd - tStart;

                    // loop around outputs
                        for (int i = 0; i < CUR_NUM_OF_BOXES; i++) {
                            // fill RectF with boxes location
                            boxes = boxes_location(outputBoxes[0][i][0], outputBoxes[0][i][1],
                                    outputBoxes[0][i][2], outputBoxes[0][i][3]);
                            detection[i] = new RectF(
                                    boxes[1], // left
                                    boxes[0], // top
                                    boxes[3], // right
                                    boxes[2]  // bottom
                            );
                        }
                    if (outputBoxes[0] != null) {
                        draw_boxes(tImage.getBitmap());
                    }
                    else{
                        Toast.makeText(getApplicationContext(), "No Boxes found!!!", Toast.LENGTH_SHORT).show();
                    }
                    // set values
                    time_elapsed.setText(("Run Time: "+ tRes/1000 + " sec"));
                }
                else
                {
                    Toast.makeText(getApplicationContext(), "Please select a image first", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // plus button to show more results
        add_num_of_boxes.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if (CUR_NUM_OF_BOXES != 3) {
                    CUR_NUM_OF_BOXES += 1;
                    cur_res.setText(String.valueOf(CUR_NUM_OF_BOXES));
                }
            }
        });

        // show less results
        rem_num_of_boxes.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if (CUR_NUM_OF_BOXES !=1) {
                    CUR_NUM_OF_BOXES -= 1;
                    cur_res.setText(String.valueOf(CUR_NUM_OF_BOXES));
                }
            }
        });

        // floating button action to pick image
        FloatingActionButton fab1 = findViewById(R.id.fab1);
        fab1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, IF_IMAGE_PICK);
            }
        });

        // floating button action to launch camera for detection
        FloatingActionButton fab2 = findViewById(R.id.fab2);
        fab2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(TfDetector.this, TfDetectorCamera.class);
                startActivity(intent);
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        MenuItem mv = menu.findItem(R.id.Settings_option);
        mv.setIcon(R.drawable.setting_tf);
        return true;
    }

    // takes bitmap and draws boxes on them
    public void draw_boxes(Bitmap image){
        // clear canvas, textboxes
        for (int i=0; i < 3; i++) {
            labels_view[i].setText(null);
            probs_view[i].setText(null);
        }
        Bitmap bt = image.copy(Bitmap.Config.ARGB_8888, true); // set to mutable
        Canvas canvas = new Canvas(bt); // new canvas
        Log.d("Number of  detected Boxes: ", ""+detection.length);

        // show boxes result
        for (int i=0; i < CUR_NUM_OF_BOXES; i++){
            if (detection[i] != null) {
                labels_view[i].setText(labelList.get((int) outputClasses[0][i] + labelOffset));
                probs_view[i].setText(String.valueOf(outputScores[0][i]));
                paint.setColor(boxes_colors[i]);
                canvas.drawRect(detection[i], paint);
            }
            else   {
                break;
            }
        }
        displayed_image.setImageBitmap(bt);
    }


    // load model from uri
    private MappedByteBuffer load_model_file() throws IOException {

        FileInputStream inputStream = new FileInputStream(getApplicationContext()
                .getContentResolver().openFileDescriptor(model_uri, "r").getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long start_postion = fileChannel.position();
        long size = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, start_postion, size);
    }

    // loads Label from custom labels file
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

    // get boxes locations from model(because model outputs float values from 0 to 1)
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

    // check if something goes wrong
    private void make_logs(){
        Log.d(TAG, "Model num_of_classes "+ tflite.getOutputTensor(0).shape()[1]+
                " provided_classes_num " + outputClasses[0].length);
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
                        TfDetector.super.onBackPressed();
                        finishAffinity();
                    }
                }).create().show();
    }


    // actions for toolbar
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.Settings_option:
                Intent se = new Intent(TfDetector.this, Settings.class);
                startActivity(se);
                return true;

            case R.id.Help_option:
                Intent hp = new Intent(TfDetector.this, Help.class);
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
        if (requestCode ==IF_IMAGE_PICK) {
            IF_IMAGE = true;
            Uri uri = data.getData();
            try {
                selected_bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                displayed_image.setImageBitmap(selected_bitmap);
                displayed_image.setRotation(displayed_image.getRotation());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}



