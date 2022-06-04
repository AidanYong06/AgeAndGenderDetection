package ai.certifai.solution.facial_recognition.GenderAndAgeDetector;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static ai.certifai.solution.facial_recognition.GenderAndAgeDetector.AgeGenderDetection.logger;

public class GenderModel {

    static final Logger log = org.slf4j.LoggerFactory.getLogger(GenderModel.class);
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 2;
    private static int batchSize = 64;
    private static int seed = 123;
    private static final Random randNumGen = new Random(seed);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int trainPerc = 80;
    private static int epochs = 25;
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/GenderDetection.zip");


    public static void main(String[] Args) throws Exception
    {

        //image augmentation
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        ImageTransform showImage = new ShowImageTransform("Image", 1000);
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage, 0.3)
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        GenderIterator.setup(batchSize, 80, transform);

        //create iterators
        DataSetIterator trainIter = GenderIterator.trainIterator();
        DataSetIterator testIter = GenderIterator.testIterator();

        //model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(0.0005)
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .kernelSize(7,7)
                        .stride(4,4)
                        .nIn(channels)
                        .nOut(96)
                        .activation(Activation.RELU)
                        .build())
                .layer(1,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .build())
                .layer(2, new LocalResponseNormalization.Builder().build())
                .layer(3,new ConvolutionLayer.Builder()
                        .kernelSize(5,5)
                        .padding(2,2)
                        .nIn(96)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(4,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6,new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .padding(1,1)
                        .nIn(256)
                        .nOut(384)
                        .activation(Activation.RELU)
                        .build())
                .layer(7,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .build())
                .layer(8,new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512)
                        .dropOut(0.5)
                        .build())
                .layer(9,new DenseLayer.Builder()
                        .nIn(512)
                        .dropOut(0.5)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(10,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(512)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();


        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //train model and eval model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        model.fit(trainIter, epochs);
        ModelSerializer.writeModel(model, modelFilename, true);
        logger.info("Model completed");

    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        if (training && transform != null){
            recordReader.initialize(split,transform);
        }else{
            recordReader.initialize(split);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, false);
    }

}
