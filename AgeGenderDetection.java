package ai.certifai.solution.facial_recognition.GenderAndAgeDetector;


import ai.certifai.solution.facial_recognition.identification.Prediction;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Point;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class AgeGenderDetection {
    static final Logger logger = LoggerFactory.getLogger(AgeGenderDetection.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Age and Gender Detection - DL4J";
    private static File CNNAgeModel = new File(System.getProperty("user.dir"), "generated-models/AgeDetection.zip");
    private static File CNNGenderModel = new File(System.getProperty("user.dir"), "generated-models/GenderDetection.zip");
    private static final String[] AGES = new String[]{"0-4", "6-13", "23-35", "37-45", "47-55", "14-21", "60-"};
    private static MultiLayerNetwork AgeModel;
    private static MultiLayerNetwork GenderModel;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;

    private static List<Prediction> predictions;


    private FrameGrabber frameGrabber;
    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private volatile boolean running = false;

    private HaarFaceDetector faceDetector = new HaarFaceDetector();

    private AgeModel ageModel = new AgeModel();

    private JFrame window;
    private JPanel videoPanel;

    public static void main(String[] args) throws Exception {

        if (new File(CNNAgeModel.toString()).exists()) {
            logger.info("Load model...");
            AgeModel = ModelSerializer.restoreMultiLayerNetwork(CNNAgeModel);
            logger.info("Model found.");
        } else {
            logger.info("Model not found.");
        }

        if (new File(CNNGenderModel.toString()).exists()) {
            logger.info("Load model...");
            GenderModel = ModelSerializer.restoreMultiLayerNetwork(CNNGenderModel);
            logger.info("Model found.");
        } else {
            logger.info("Model not found.");
        }

        AgeGenderDetection ageGenderDetection = new AgeGenderDetection();

        logger.info("Starting AgeGenderDetection");
        new Thread(ageGenderDetection::start).start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Stopping AgeGenderDetection");
            ageGenderDetection.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException ignored) { }
    }

    public AgeGenderDetection() {
        window = new JFrame();
        videoPanel = new JPanel();

        window.setLayout(new BorderLayout());
        window.setSize(new Dimension(1280, 720));
        window.add(videoPanel, BorderLayout.CENTER);
        window.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stop();
            }
        });
    }
    private void process() {
        running = true;
        while (running) {
            try {
                // Here we grab frames from our camera
                final org.bytedeco.javacv.Frame frame = frameGrabber.grab();

                Map<Rect, Mat> detectedFaces = faceDetector.detect(frame);
                Mat mat = toMatConverter.convert(frame);

                detectedFaces.entrySet().forEach(rectMatEntry -> {
                    String age = predictAge(rectMatEntry.getValue(), frame);
                    Gender gender = predictGender(rectMatEntry.getValue(), frame);
                    String caption = String.format("%s:[%s]",gender, age);
                    logger.debug("Face's caption : {}", caption);

                    rectangle(mat, new org.bytedeco.opencv.opencv_core.Point(rectMatEntry.getKey().x(), rectMatEntry.getKey().y()),
                            new org.bytedeco.opencv.opencv_core.Point(rectMatEntry.getKey().width() + rectMatEntry.getKey().x(), rectMatEntry.getKey().height() + rectMatEntry.getKey().y()),
                            Scalar.RED, 2, CV_AA, 0);

                    int posX = Math.max(rectMatEntry.getKey().x() - 10, 0);
                    int posY = Math.max(rectMatEntry.getKey().y() - 10, 0);
                    putText(mat, caption, new Point(posX, posY), CV_FONT_HERSHEY_PLAIN, 1.0,
                            new Scalar(255, 255, 255, 2.0));
                });

                // Show the processed mat in UI
                org.bytedeco.javacv.Frame processedFrame = toMatConverter.convert(mat);

                Graphics graphics = videoPanel.getGraphics();
                BufferedImage resizedImage = ImageUtils.getResizedBufferedImage(processedFrame, videoPanel);
                SwingUtilities.invokeLater(() -> {
                    graphics.drawImage(resizedImage, 0, 0, videoPanel);
                });
            } catch (FrameGrabber.Exception e) {
                logger.error("Error when grabbing the frame", e);
            } catch (Exception e) {
                logger.error("Unexpected error occurred while grabbing and processing a frame", e);
            }
        }
    }
    public void start() {
        frameGrabber = new OpenCVFrameGrabber(0);

        //frameGrabber.setFormat("mp4");
        frameGrabber.setImageWidth(1280);
        frameGrabber.setImageHeight(720);

        logger.debug("Starting frame grabber");
        try {
            frameGrabber.start();
            logger.debug("Started frame grabber with image width-height : {}-{}", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        } catch (FrameGrabber.Exception e) {
            logger.error("Error when initializing the frame grabber", e);
            throw new RuntimeException("Unable to start the FrameGrabber", e);
        }

        SwingUtilities.invokeLater(() -> {
            window.setVisible(true);
        });

        process();

        logger.debug("Stopped frame grabbing.");
    }
    public void stop() {
        running = false;
        try {
            logger.debug("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            logger.error("Error occurred when stopping the FrameGrabber", e);
        }

        window.dispose();
    }

    public String predictAge(Mat face, Frame frame) {
        try {
            resize(face, face, new Size(224, 224));

            NativeImageLoader loader = new NativeImageLoader();

            INDArray ds = null;
            try {
                ds = loader.asMatrix(face);
            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(ds);
            System.out.println(java.util.Arrays.toString(ds.shape()));
            INDArray results = AgeModel.output(ds);
            INDArray t = Nd4j.argMax(results,1);
            int c = t.getInt(0);


            return AGES[c];
        } catch (Exception e) {
            logger.error("Error when processing age", e);
        }
        return null;
    }

    public AgeGenderDetection.Gender predictGender(Mat face, Frame frame) {
        try {
            resize(face, face, new Size(224, 224));

            NativeImageLoader loader = new NativeImageLoader();

            INDArray ds = null;
            try {
                ds = loader.asMatrix(face);
            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(ds);
            System.out.println(Arrays.toString(ds.shape()));
            INDArray results = GenderModel.output(ds);
            double a = results.getDouble(0,0);
            double b = results.getDouble(0,1);
            logger.debug("CNN results {},{}", results.getScalar(0,0), results.getScalar(0,1));

            if (a < b) {
                logger.debug("Male detected");
                return AgeGenderDetection.Gender.MALE;
            } else {
                logger.debug("Female detected");
                return AgeGenderDetection.Gender.FEMALE;
            }
        } catch (Exception e) {
            logger.error("Error when processing gender", e);
        }
        return AgeGenderDetection.Gender.NOT_RECOGNIZED;
    }

    public enum Gender {
        FEMALE,
        MALE,
        NOT_RECOGNIZED
    }

}
