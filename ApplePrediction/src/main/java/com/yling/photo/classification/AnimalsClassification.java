package com.yling.photo.classification;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * 用LeNet(一种卷积神经网络）对4类动物的图像进行分类
 * 该示例用较为简单的卷积网络模型LeNet和较低的分辨率(60*60*3)，训练得到的模型准确率较低
 * 可以尝试讲模型修改为较为复杂的网络模型和使用更高的分辨率以获得更高的准确率
 */

public class AnimalsClassification {
    protected static final Logger log = LoggerFactory.getLogger(AnimalsClassification.class);
    protected static int height = 60;
    protected static int width = 60;
    protected static int channels = 3;
    protected static int numExamples = 51;
    protected static int numLabels = 4;
    protected static int batchSize = 15;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);

    protected static int iterations = 1;
    protected static int epochs = 200;
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;


    public void run(String[] args) throws Exception {

        log.info("Load data....");
        /**cd
         *
         * 下面的代码从文件夹中读取图片作为输入数据
         * 将每种类别的图片分别放在不同的文件夹下，并将这些文件夹放在同一个根目录下
         * DL4J会为不同文件夹下的图片分配不同个label，为相同文件夹下的图片分配相同的label
         **/
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("src/main/resources/animals");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        //将数据分为训练数据和测试数据
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        //利用一些图像变换来生成一些训练数据
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});


        //归一化
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");

        MultiLayerNetwork network = lenetModel();
        network.init();
        network.setListeners(new ScoreIterationListener(10));

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        // 用原始图像来训练
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainIter);

        // 用变换的图像来训练
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter);
            network.fit(trainIter);
        }


        //评价模型
        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // 取出第一条数据进行预测
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        // 保存模型
        if (save) {
            log.info("Save model....");
            ModelSerializer.writeModel(network, "model.bin", true);
        }
        log.info("****************Example finished********************");
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    //构建LeNet
    public MultiLayerNetwork lenetModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false)
                .activation(Activation.RELU) // 用RELU激活
                .learningRate(1e-2) // 学习速率
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static void main(String[] args) throws Exception {
        new AnimalsClassification().run(args);
    }

}
