package ai.certifai.solution.facial_recognition.GenderAndAgeDetector;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
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
import org.nd4j.common.io.ClassPathResource;
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
import java.util.*;

public class AgeModel {

    private static final Logger logger = org.slf4j.LoggerFactory.getLogger(AgeModel.class);
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 7;
    private static int batchSize = 32;
    private static int seed = 123;
    private static final Random rng = new Random(seed);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int trainPerc = 80;
    private static int epochs = 25;
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/AgeDetection.zip");




    public static void main(String[] Args) throws Exception
    {

        // image augmentation
        File dir = new ClassPathResource("age-classification").getFile();
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(rng, 15);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3));
        transform = new PipelineImageTransform(pipeline,shuffle);
        FileSplit filesInDir = new FileSplit(dir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];

        //create iterators
        DataSetIterator trainIter = trainIterator();
        DataSetIterator testIter = testIterator();

        //model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(0.0005)
                .list()
                .layer(0,new BatchNormalization())
                .layer(1,new ConvolutionLayer.Builder()
                        .kernelSize(5,5)
                        .stride(2,2)
                        .nIn(channels)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(3,new ConvolutionLayer.Builder()
                        .kernelSize(5,5)
                        .stride(2,2)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(4,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build()).layer(6,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(numClasses)
                        .build())
                .setInputType(InputType.convolutional(height,width,channels))
                .backpropType(BackpropType.Standard)
                .build();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //train model and eval model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(50));

        model.fit(trainIter,epochs);

        {
            Evaluation evaluation = model.evaluate(testIter);
            System.out.println(evaluation.stats());
        }
        {
            Evaluation evaluation = model.evaluate(trainIter);
            System.out.println(evaluation.stats());
        }

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
