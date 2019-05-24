package com.yling.nlp.word2vec;

import java.nio.charset.Charset;

public class FitExample {

	public static void main(String[] args) {
		Word2VecUtils.newWord2Vec()
				.addAllTextFile("/Users/zhouzhou/Downloads/【TXT-006】百科全书/", file -> file.getName().endsWith(".txt"))
				.charset(Charset.forName("GB2312")).saveAt("/Users/zhouzhou/temp/result", true).build();
	}

}
