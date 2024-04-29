package com.example.handkeypoint;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import android.Manifest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {

    private static final String TAG = "MainActivity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;
    private Mat mGray;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 101;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        mOpenCvCameraView = findViewById(R.id.camera_view);

        if(OpenCVLoader.initDebug()){
            Log.d("OPENCV:APP", "OpenCV loaded");
            mOpenCvCameraView.enableView();
        }



        mOpenCvCameraView.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            @Override
            public void onCameraViewStarted(int width, int height) {
                mGray = new Mat(height, width, CvType.CV_8UC1);
                mRgba = new Mat(height, width, CvType.CV_8UC4);
            }

            @Override
            public void onCameraViewStopped() {
                Log.d("cam", "cam stopped");
                mGray.release();
                mRgba.release();

            }

            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                Mat rgba = inputFrame.rgba();
                Mat gray = inputFrame.gray();

                // Apply Gaussian blur
                Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

                // Apply thresholding
                Imgproc.threshold(gray, gray, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

                // Find contours
                List<MatOfPoint> contours = new ArrayList<>();
                Mat hierarchy = new Mat();
                Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

                // Find hand contour (largest contour)
                MatOfPoint handContour = null;
                double maxArea = 0;
                for (MatOfPoint contour : contours) {
                    double area = Imgproc.contourArea(contour);
                    if (area > maxArea) {
                        maxArea = area;
                        handContour = contour;
                    }
                }

                // Draw hand contour
                Imgproc.drawContours(rgba, contours, -1, new Scalar(255, 0, 0), 3);

                // Find convex hull of hand contour
                MatOfInt hull = new MatOfInt();
                Imgproc.convexHull(handContour, hull);

                // Find convexity defects
                MatOfInt4 defects = new MatOfInt4();
                Imgproc.convexityDefects(handContour, hull, defects);

                // Count fingers based on defects
                int numFingers = 0;
                for (int i = 0; i < defects.rows(); i++) {
                    double[] defect = defects.get(i, 0);
                    int startIdx = (int) defect[0];
                    int endIdx = (int) defect[1];
                    int farIdx = (int) defect[2];
                    double depth = defect[3];

                    if (depth > 50) {
                        numFingers++;
                        Imgproc.circle(rgba, handContour.toList().get(startIdx), 10, new Scalar(255, 0, 0), -1);
                        Imgproc.circle(rgba, handContour.toList().get(endIdx), 10, new Scalar(255, 0, 0), -1);
                        Imgproc.circle(rgba, handContour.toList().get(farIdx), 10, new Scalar(0, 255, 0), 5);
                    }
                }

                Log.d(TAG, "Number of fingers: " + numFingers);

                return rgba;
            }


        });

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed");
        } else {
            Log.d(TAG, "OpenCV initialization succeeded");
            mOpenCvCameraView.enableView();
        }




        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initDebug();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(grantResults.length>0&&grantResults[0]!=PackageManager.PERMISSION_GRANTED){
            getPermission();
        }
    }
    private void getPermission() {
        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 101);
        }

    }
}
