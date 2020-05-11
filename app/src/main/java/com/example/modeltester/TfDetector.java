package com.example.modeltester;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
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

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TfDetector extends AppCompatActivity {
    // floating action button status
    public static final int IF_IMAGE_PICK = 1;
    // if image boolean
    private Boolean IF_IMAGE = false;
    // image size
    private int SIZE_X;
    private int SIZE_Y;
    // image pixel/ channels
    private int NUM_OF_CHANNELS = 3;
    // number of detection
    private int NUM_OF_DETECTION = 10;
    private int CUR_NUM_OF_BOXES = 1;
    // offset value
    private int labelOffset = 1;
    // tflite options
    private Interpreter.Options options = null;
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // int array to hold image data
    private int[] intValues;
    // image buffer data
    private ByteBuffer imgData = null;
    // output classes array[batch size, number of classes]
    private float[][] outputClasses;
    // output boxes array[batch size,number of detection, numofboxes]
    private float[][][] outputBoxes;
    // output prediction probability[batchsize, num of detection]
    private float[][] outputScores;
    // number of detections
    private float[] numDetection;
    // boxes location
    private int[] boxes;
    // map to hold all detection values
    private Map<Integer, Object> outputMap;
    // rect box
    private RectF[] detection;
    // current result threshold
    // image bitmap main
    private Bitmap selected_bitmap;
    // get model and label uri
    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();

    // activity elements
    private ImageView displayed_image;
    private Button detect;
    private Button add_res;
    private Button rem_res;
    private TextView cur_res;
    private TextView time_elapsed;
    private TextView label1;
    private TextView label2;
    private TextView label3;
    private TextView Confidence1;
    private TextView Confidence2;
    private TextView Confidence3;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tf_detector);
        Toolbar toolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayShowTitleEnabled(false);
        // imageview
        displayed_image = findViewById(R.id.selected_image);

        //initialize graph and labels
        try {
            GpuDelegate delegate = new GpuDelegate();
            options = (new Interpreter.Options()).addDelegate(delegate);
            tflite = new Interpreter(load_model(), options);
            labelList = load_labels();
            // getting input array size from model
            Tensor ten = tflite.getInputTensor(0);
            int[] inp_arr = ten.shape();
            SIZE_X = inp_arr[1];
            SIZE_Y = inp_arr[2];
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        // initialize elements
        add_res = findViewById(R.id.add_res);
        rem_res = findViewById(R.id.rem_res);

        // labels that hold top three results of CNN
        label1 = findViewById(R.id.class1);
        label2 = findViewById(R.id.class2);
        label3 = findViewById(R.id.class3);
        // displays the probabilities of top labels
        Confidence1 = findViewById(R.id.prob1);
        Confidence2 = findViewById(R.id.prob2);
        Confidence3 = findViewById(R.id.prob3);

        // model runtime display
        time_elapsed = findViewById(R.id.time_elapsed);
        cur_res = findViewById(R.id.res_text);

        // initialize array to store image data
        intValues = new int[SIZE_X * SIZE_Y];

        //image buffer initialization
        imgData = ByteBuffer.allocateDirect(SIZE_X * SIZE_Y * NUM_OF_CHANNELS);
        imgData.order(ByteOrder.nativeOrder());

        // initialize output arrays
        outputBoxes = new float[1][NUM_OF_DETECTION][4];
        outputClasses = new float[1][NUM_OF_DETECTION];
        outputScores = new float[1][NUM_OF_DETECTION];
        numDetection = new float[1];
        detection = new RectF[3];

        // map for containing all output arrays
        outputMap =  new HashMap<>();
        outputMap.put(0, outputBoxes);
        outputMap.put(1,outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetection);
        // boxes
        boxes = new int[4];

        // detect button
        detect = findViewById(R.id.detect);
        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {
                    // convert the bitmap to bytebufer for input to model
                    convertBitmapToByteBuffer(selected_bitmap);
                    // create input object
                    Object[] inputArray = {imgData};
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
                        boxes =  boxes_location(outputBoxes[0][i][0], outputBoxes[0][i][1],
                                outputBoxes[0][i][2], outputBoxes[0][i][3]);
                         detection[i] = new RectF(
                                 boxes[1], // left
                                 boxes[0], // top
                                 boxes[3], // right
                                 boxes[2]  // bottom
                        );
                    }
                    // clear canvas
                    Canvas canvas = new Canvas(selected_bitmap);
                    Paint paint = new Paint();
                    paint.setStyle(Paint.Style.STROKE);
                    if (CUR_NUM_OF_BOXES == 1) {
                        label1.setText(labelList.get((int) outputClasses[0][0] + labelOffset));
                        Confidence1.setText(String.valueOf(outputScores[0][0]));
                        label2.setText(null);
                        Confidence2.setText(null);
                        label3.setText(null);
                        Confidence3.setText(null);

                        paint.setColor(Color.BLUE);
                        canvas.drawRect(detection[0], paint);
                    }
                    if (CUR_NUM_OF_BOXES == 2) {
                        label1.setText(labelList.get((int) outputClasses[0][0] + labelOffset));
                        Confidence1.setText(String.valueOf(outputScores[0][0]));
                        label2.setText(labelList.get((int) outputClasses[0][1] + labelOffset));
                        Confidence2.setText(String.valueOf(outputScores[0][1]));
                        label3.setText(null);
                        Confidence3.setText(null);

                        paint.setColor(Color.BLUE);
                        canvas.drawRect(detection[0], paint);
                        paint.setColor(Color.GREEN);
                        canvas.drawRect(detection[1], paint);
                    }
                    if (CUR_NUM_OF_BOXES == 3) {
                        label1.setText(labelList.get((int) outputClasses[0][0] + labelOffset));
                        Confidence1.setText(String.valueOf(outputScores[0][0]));
                        label2.setText(labelList.get((int) outputClasses[0][1] + labelOffset));
                        Confidence2.setText(String.valueOf(outputScores[0][1]));
                        label3.setText(labelList.get((int) outputClasses[0][2] + labelOffset));
                        Confidence3.setText(String.valueOf(outputScores[0][2]));

                        paint.setColor(Color.BLUE);
                        canvas.drawRect(detection[0], paint);
                        paint.setColor(Color.GREEN);
                        canvas.drawRect(detection[1], paint);
                        paint.setColor(Color.RED);
                        canvas.drawRect(detection[2], paint);
                    }

                    displayed_image.setImageBitmap(selected_bitmap);
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
        add_res.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if (CUR_NUM_OF_BOXES != 3) {
                    CUR_NUM_OF_BOXES += 1;
                    cur_res.setText(String.valueOf(CUR_NUM_OF_BOXES));
                }
            }
        });

        // show less results
        rem_res.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if (CUR_NUM_OF_BOXES !=1) {
                    CUR_NUM_OF_BOXES -= 1;
                    cur_res.setText(String.valueOf(CUR_NUM_OF_BOXES));
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
                startActivityForResult(intent, IF_IMAGE_PICK);
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


    // load model from uri
    private MappedByteBuffer load_model() throws IOException {

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
                // resize the bitmap
                selected_bitmap = getResizedBitmap(selected_bitmap, SIZE_X, SIZE_Y);
                displayed_image.setImageBitmap(selected_bitmap);
                displayed_image.setRotation(displayed_image.getRotation());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer (Bitmap bitmap){
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < SIZE_X; ++i) {
            for (int j = 0; j < SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
    }



    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }

}



