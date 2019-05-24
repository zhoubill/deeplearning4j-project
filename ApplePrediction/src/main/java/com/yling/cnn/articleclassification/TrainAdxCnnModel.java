package com.yling.cnn.articleclassification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 基于CNN的文本分类
 * @author zhouzhou
 *
 */
public class TrainAdxCnnModel {

	static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		String WORD_VECTORS_PATH = "adx/word2vec.model";
		// 基础配置
		int batchSize = 10;
		int vectorSize = 100; // 词典向量的维度,这边是100
		int nEpochs = 3; // 迭代代数
		int truncateReviewsToLength = 256; // 词长大于256则抛弃
		int cnnLayerFeatureMaps = 100; // 卷积神经网络特征图标 / channels / CNN每层layer的深度
		PoolingType globalPoolingType = PoolingType.MAX;
		Random rng = new Random(100); // 随机抽样
		// 设置网络配置->我们有多个卷积层，每个带宽3,4,5的滤波器
		ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().weightInit(WeightInit.RELU)
				.activation(Activation.LEAKYRELU).updater(Updater.ADAM).convolutionMode(ConvolutionMode.Same) 
				// This is important so we can 'stack' the results later
				.regularization(true).l2(0.0001).learningRate(0.01).graphBuilder().addInputs("input")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),"input")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder().kernelSize(4, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),"input")
				.addLayer("cnn5",
						new ConvolutionLayer.Builder().kernelSize(5, vectorSize).stride(1, vectorSize).nIn(1)
								.nOut(cnnLayerFeatureMaps).build(),"input")
				.addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5") // Perform depth concatenation
				.addLayer("globalPool", new GlobalPoolingLayer.Builder().poolingType(globalPoolingType).build(),
						"merge")
				.addLayer("out",
						new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
								.activation(Activation.SOFTMAX).nIn(3 * cnnLayerFeatureMaps).nOut(2) 
								// 2 classes: positive or negative
								.build(),
						"globalPool")
				.setOutputs("out").build();
		ComputationGraph net = new ComputationGraph(config);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		// 加载向量字典并获取训练集合测试集的DataSetIterators
		System.out.println("Loading word vectors and creating DataSetIterators");
		WordVectors wordVectors = WordVectorSerializer
				.fromPair(WordVectorSerializer.loadTxt(new File(WORD_VECTORS_PATH)));

		DataSetIterator trainIter = getDataSetIterator(true, wordVectors, batchSize, truncateReviewsToLength, rng);
		DataSetIterator testIter = getDataSetIterator(false, wordVectors, batchSize, truncateReviewsToLength, rng);
		System.out.println("Starting training");
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainIter);
			trainIter.reset();
			// 进行网络演化(进化)获得网络判定参数
			Evaluation evaluation = net.evaluate(testIter);
			testIter.reset();
			System.out.println(evaluation.stats());
		}
		// 训练之后:加载一个句子并输出预测
		String contentsFirstPas = "我的 手机 是 手机号码";

		INDArray featuresFirstNegative = ((CnnSentenceDataSetIterator) testIter).loadSingleSentence(contentsFirstPas);
		INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
		List<String> labels = testIter.getLabels();
		System.out.println("\n\nPredictions for first negative review:");
		for (int i = 0; i < labels.size(); i++) {
			System.out.println("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
		}

	}

	private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
			int maxSentenceLength, Random rng) {
		String path = isTraining ? "adx/rnnsenec.txt" : "adx/rnnsenectest.txt";
		LabeledSentenceProvider sentenceProvider = new LabeledSentence(path, rng);

		return new CnnSentenceDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
				.minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength).useNormalizedWordVectors(false)
				.build();
	}

}
