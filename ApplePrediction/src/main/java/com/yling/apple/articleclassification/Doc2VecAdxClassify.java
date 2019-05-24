package com.yling.apple.articleclassification;

import java.util.List;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;


public class Doc2VecAdxClassify {

	private String path = "adx/rnnsenec.txt";
	ParagraphVectors paragraphVectors;
	LabelAwareIterator iterator;
	TokenizerFactory tokenizerFactory;
	public static void main(String[] args) {
		Doc2VecAdxClassify doc2vec = new Doc2VecAdxClassify();
		doc2vec.makeParagraphVectors();
		// 预测分类
		System.out.println(doc2vec.paragraphVectors
				.predict("小麦 喂猪 小麦 饲料 玉米 大牛 同志 效率 适口 用于 很好 取代 生猪 养殖 喂养 喂猪 较低 热能 进行 乳猪 想用 使用 "));
		
		System.out.println(doc2vec.paragraphVectors
				.predict("白天 产仔 母猪 母猪 产仔 发情 白天 配种 护理 夜间 烯醇 前列 生子 注射 时间 分娩 上午 子宫 次日 排出 恶露 胎衣 工作量 发生 加速 安静 劳动 难产 减少 顺利 缩短 兽医 及时 药物 副作用 不能 方便 接产 常会 促进 诊治 断奶 往往 造成 怎样才能 能力 子猪 预防 对接 膜炎 提高 死亡 天数 繁殖 天天 调整 利于 规律 改变传统 排卵 复原 成本低 肌肉注射 颈部 微克 诱导 以前 授精 早上 安排 临产 下午 养猪户 妊娠 以往 多数 试一试 情况下"));
		
		System.out.println(doc2vec.paragraphVectors
				.predict("奶牛 品种 品种 兼用 三河 中国 西门塔尔 奶牛 丹麦 瑞士 赛牛 短角 草原 娟姗 夏牛 红牛 斯坦 新疆 科尔沁 分为  "));
		
		System.out.println(doc2vec.paragraphVectors
				.predict("南江 黄羊 羔羊 饲养 母羊 小羊羔 进行 妈妈 圈舍 即可 每天 日子 运动 奶水 注射 注意 饲养 饲喂 母羊 管理 只羊 晒太阳 木板 单独 出入口 体质 寒冷 冬季 促使 成长 干草 铺设 拿开 母子 防疫 增强 越冬 羊群 运动场 怀孕 增加 阳光明媚 地面温度 方法 窗户 薄膜 傍晚 刮大风 密封 出生 辅助 加大 以后 跟着 预防 发生 足够 吃到 肺病 帮助 吃饱 母乳 达到 精饲料 充足 生活 分开 体重 粗饲料 刚出生 保证 天气 堵上 晴朗 防止 刻意 赶到 阿福丁 需要 有效 放到 进入 育成 寄生虫 侵害 配方 公羊 饲料 阶段"));
		
		System.out.println(doc2vec.paragraphVectors
				.predict("母牛 食盐 提高 发生 左右 需水 增加 主要 不足 水比 奶牛 发酵 保持 能力 标准 供给 饲料 奶 混合 干草 之一 生产的 "));
		
		MeansBuilder meansBuilder = new MeansBuilder(
				(InMemoryLookupTable<VocabWord>) doc2vec.paragraphVectors
						.getLookupTable(),
				doc2vec.tokenizerFactory);
		LabelSeeker seeker = new LabelSeeker(doc2vec.iterator.getLabelsSource()
				.getLabels(),
				(InMemoryLookupTable<VocabWord>) doc2vec.paragraphVectors
						.getLookupTable());
		LabelledDocument document = new LabelledDocument();
		document.setContent("小麦 喂猪 小麦 饲料 玉米 大牛 同志 效率 适口 用于 很好 取代 生猪 养殖 喂养 喂猪 较低 热能 进行 乳猪 想用 使用 ");
		document.addLabel("1");
		INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
		List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
		for (Pair<String, Double> score : scores) {
			System.out.println(" " + score.getFirst() + ": "+ score.getSecond());
		}
	}
	public void makeParagraphVectors() {
		String inputFile = Doc2VecAdxClassify.class.getClassLoader().getResource(path).getPath();
		System.out.println("path is :" + inputFile);
		iterator = new TxtLabelAwareIterator(inputFile);
//		System.out.println(iterator.getLabelsSource().getLabels());
		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		paragraphVectors = new ParagraphVectors.Builder()
		.learningRate(0.025)
		.minLearningRate(0.001)
		.batchSize(1000)
		.epochs(25)
		.iterate(iterator)
		.trainWordVectors(true)	
		.tokenizerFactory(tokenizerFactory)
		.build();
		// Start model training
		paragraphVectors.fit();
		
	}
}

