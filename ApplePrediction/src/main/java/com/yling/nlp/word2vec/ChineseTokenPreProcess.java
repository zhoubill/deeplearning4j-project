package com.yling.nlp.word2vec;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

public final  class ChineseTokenPreProcess implements TokenPreProcess {

	 @Override
	  public String preProcess(String token) {
	    token = token.trim();
	    if (token.isEmpty()) return null;
	    return token;
	  }

}
