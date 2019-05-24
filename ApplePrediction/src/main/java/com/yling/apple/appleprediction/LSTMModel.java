package com.yling.apple.appleprediction;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMModel {
	//数据输入的维度设置（就是有几种属性的字段就设置多大的数字）
	private static final int IN_NUM = 6;
	private static final int OUT_NUM = 1;
	//设置需要训练的迭代次数
	private static final int Epochs = 40;
	//设置模型属性配置中的损失函数迭代次数
	private static final int iterations = 1;
	//设置神经网络中第一层隐藏神经元数量
	private static final int lstmLayer1Size = 50;
	private static final int lstmLayer2Size = 100;

	public static MultiLayerNetwork getNetModel(int nIn, int nOut) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(iterations)
				//设置学习速率，就是每次梯度下降的步长
				.learningRate(0.1).rmsDecay(0.5).seed(12345).regularization(true)
				.l2(0.001) //设置惩罚系数（L2正则化）提高模型的泛化能力
				.weightInit(WeightInit.XAVIER).updater(Updater.RMSPROP).list()
				.layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size).activation(Activation.TANH).build())
				.layer(1,new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size).activation(Activation.TANH)
								.build())
				.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
						.nIn(lstmLayer2Size).nOut(nOut).build())
				.pretrain(false)
				.backprop(true)
//				.backpropType(BackpropType.TruncatedBPTT)
//				.tBPTTForwardLength(60)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		//设置监听函数，这里设置的每训练1步就打印训练的分数值
		net.setListeners(new ScoreIterationListener(1));

		return net;
	}

	public static void train(MultiLayerNetwork net, StockDataIterator iterator) throws IOException {
		//这里执行迭代训练
		for (int i = 0; i < Epochs; i++) {
			DataSet dataSet = null;
			while (iterator.hasNext()) {
				dataSet = iterator.next();
				net.fit(dataSet);
			}
			iterator.reset();
			System.out.println();
			System.out.println("=================>完成第" + i + "次完整训练");
			net.rnnClearPreviousState();
		}
		
		File locationToSave = new File("src/main/resources/ApplePriceLSTM_".concat(String.valueOf("meanprice")).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);
	}

	private static INDArray getInitArray(StockDataIterator iter) {
		double[] maxNums = iter.getMaxArr();
		INDArray initArray = Nd4j.zeros(1, 6, 1);
		initArray.putScalar(new int[] { 0, 0, 0 }, 3433.85 / maxNums[0]);
		initArray.putScalar(new int[] { 0, 1, 0 }, 3445.41 / maxNums[1]);
		initArray.putScalar(new int[] { 0, 2, 0 }, 3327.81 / maxNums[2]);
		initArray.putScalar(new int[] { 0, 3, 0 }, 3470.37 / maxNums[3]);
		initArray.putScalar(new int[] { 0, 4, 0 }, 304197903.0 / maxNums[4]);
		initArray.putScalar(new int[] { 0, 5, 0 }, 3.8750365e+11 / maxNums[5]);
		return initArray;
	}

	public static void main(String[] args) throws IOException {
		//如果装了anonda就会报这样的错误，移除mkl_rt.dll文件即可
//		System.loadLibrary("mkl_rt");
		String inputFile = LSTMModel.class.getClassLoader().getResource("sh000001.csv").getPath();
		int batchSize = 1;
		int exampleLength = 30;
		//初始化深度神经网络
		StockDataIterator iterator = new StockDataIterator();
		iterator.loadData(inputFile, batchSize, exampleLength);
		File locationToSave = new File("src/main/resources/ApplePriceLSTM_".concat(String.valueOf("meanprice")).concat(".zip"));
		MultiLayerNetwork net = getNetModel(IN_NUM, OUT_NUM);
		train(net, iterator);
		INDArray initArray = getInitArray(iterator);
		net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		System.out.println("预测结果：");
		for (int j = 0; j < 20; j++) {
			INDArray output = net.rnnTimeStep(initArray);
			System.out.print(output.getDouble(0) * iterator.getMaxArr()[1] + " ");
		}
	}

}
