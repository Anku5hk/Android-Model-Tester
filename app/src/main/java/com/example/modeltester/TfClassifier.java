package com.example.modeltester;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;

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
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

public class TfClassifier extends AppCompatActivity {

    // presets for rgb conversion
    private static final int topResultsToShow = 3;
    // holds all the possible classes for graph
    private List<String> classList;
    // holds image data as bytes
    private ByteBuffer imgBytes = null;
    // holds the probabilities of each class
    private byte[][] classProbArray = null;
    // array that holds the classes with the highest probabilities
    private String[] topClasses = null;
    // array that holds the highest probabilities
    private String[] topProb = null;
    // int array to hold image data
    private int[] intValues;
    // tflite graph
    private Interpreter tflite;
    // tflite options
    private Interpreter.Options options = null;

    // floating action button status
    public static final int PICK_IMAGE = 1;
    // bool for imageview
    public boolean IF_IMAGE = false;

    // image dimensions for the graph
    private int SIZE_X;
    private int SIZE_Y;
    private int INPUT_CHANNELS = 3;

    // activity elements
    private ImageView display_imageview;
    private Button classify_button;
    private TextView class_view1;
    private TextView class_view2;
    private TextView class_view3;
    private TextView prob_view1;
    private TextView prob_view2;
    private TextView prob_view3;
    private TextView run_time_view;

    private Uri model_uri = Settings.return_model();
    private Uri labels_uri = Settings.return_labels();

    // priority queue to hold top results
    private PriorityQueue<Map.Entry<String, Float>> sorted_classes_queue =
            new PriorityQueue<>(
                    topResultsToShow,
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
        //initialize graph and labels
        try {
            GpuDelegate delegate = new GpuDelegate();
            options = (new Interpreter.Options()).addDelegate(delegate);
            tflite = new Interpreter(load_model(), options);
            classList = load_labels();
            // getting input array size from model
            Tensor ten = tflite.getInputTensor(0);
            int[] inp_arr = ten.shape();
            SIZE_X = inp_arr[1];
            SIZE_Y = inp_arr[2];

        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // initialize array that holds image data
        intValues = new int[SIZE_X * SIZE_Y];

        // initialize byte array
        imgBytes =
                ByteBuffer.allocateDirect(
                        SIZE_X * SIZE_Y * INPUT_CHANNELS);

        imgBytes.order(ByteOrder.nativeOrder());

        // initialize probabilities array
        classProbArray = new byte[1][classList.size()];

        // labels that hold top three results of CNN
        class_view1 = findViewById(R.id.class1);
        class_view2 = findViewById(R.id.class2);
        class_view3 = findViewById(R.id.class3);
        // displays the probabilities of top labels
        prob_view1 = findViewById(R.id.prob1);
        prob_view2 = findViewById(R.id.prob2);
        prob_view3 = findViewById(R.id.prob3);
        // initialize imageView that displays selected image to the user
        display_imageview = findViewById(R.id.selected_image);
        // model runtime display
        run_time_view = findViewById(R.id.time_elapsed);

        // initialize array to hold top labels
        topClasses = new String[topResultsToShow];
        // initialize array to hold top probabilities
        topProb = new String[topResultsToShow];

        // classify current displayed image
        classify_button = findViewById(R.id.classify);
        classify_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (IF_IMAGE) {
                    // get current bitmap from imageView
                    Bitmap bitmap_orig = ((BitmapDrawable) display_imageview.getDrawable()).getBitmap();
                    // resize the bitmap for graph
                    Bitmap bitmap = getResizedBitmap(bitmap_orig, SIZE_X, SIZE_Y);
                    // convert bitmap to byte array
                    BitmapToByteBuffer(bitmap);
                    // count time elapsed(start time)
                    long tStart = System.currentTimeMillis();
                    // pass byte data to the graph
                    tflite.run(imgBytes, classProbArray);
                    // end time
                    long tEnd = System.currentTimeMillis();
                    // time total
                    double tRes = tEnd - tStart;
                    // display the results
                    printTopKLabels();
                    run_time_view.setText(("Run Time: "+ tRes/1000 + " sec"));
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

    // coming back from another activity
    protected void onResume() {
        super.onResume();

        //initialize graph and labels
            try {
                tflite = new Interpreter(load_model(), options);
                classList = load_labels();
            } catch (Exception ex) {
                ex.printStackTrace();
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

    // load model from /downloads
    private MappedByteBuffer load_model() throws IOException {

            FileInputStream inputStream = new FileInputStream(getApplicationContext()
                    .getContentResolver().openFileDescriptor(model_uri, "r").getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long start_postion = fileChannel.position();
            long size = fileChannel.size();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, start_postion, size);
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


        // converts bitmap to byte array which is passed in the tflite graph
        private void BitmapToByteBuffer (Bitmap bitmap){
            if (imgBytes == null) {
                return;
            }
            imgBytes.rewind();
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            // loop through all pixels
            int pixel = 0;
            for (int i = 0; i < SIZE_X; ++i) {
                for (int j = 0; j < SIZE_Y; ++j) {
                    final int val = intValues[pixel++];
                    // get rgb values from intValues where each int holds the rgb values for a pixel.
                    // if quantized, convert each rgb value to a byte, otherwise to a float
                    imgBytes.put((byte) ((val >> 16) & 0xFF));
                    imgBytes.put((byte) ((val >> 8) & 0xFF));
                    imgBytes.put((byte) (val & 0xFF));
                }
            }
        }

    // print the top labels and respective confidences
    private void printTopKLabels() {
        // add all results to priority queue
        for (int i = 0; i < classList.size(); ++i) {
            sorted_classes_queue.add(
                    new AbstractMap.SimpleEntry<>(classList.get(i), (classProbArray[0][i] & 0xff) / 255.0f));
            if (sorted_classes_queue.size() > topResultsToShow) {
                sorted_classes_queue.poll();
            }
        }

        // get top results from priority queue
        final int size = sorted_classes_queue.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sorted_classes_queue.poll();
            topClasses[i] = label.getKey();
            topProb[i] = String.format(Locale.ENGLISH,"%.0f%%",label.getValue()*100);
        }

        // set the corresponding textviews with the results
        class_view1.setText(("1. "+topClasses[2]));
        class_view2.setText(("2. "+topClasses[1]));
        class_view3.setText(("3. "+topClasses[0]));
        prob_view1.setText(topProb[2]);
        prob_view2.setText(topProb[1]);
        prob_view3.setText(topProb[0]);
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
