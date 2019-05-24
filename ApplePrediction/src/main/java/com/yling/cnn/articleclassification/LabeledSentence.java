package com.yling.cnn.articleclassification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.util.RandomUtils;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.collection.CompactHeapStringList;

public class LabeledSentence implements LabeledSentenceProvider {

	private int totalCount;
	private Map<String, List<String>> filesByLabel;
	private List<String> normList;
	private List<String> negList;
	private final List<String> sentenslist;
	private final int[] labelIndexes;
	private final Random rng;
	private final int[] order;
	private final List<String> allLabels;
	private int cursor = 0;

	public LabeledSentence(String path) {
		this(path, new Random());
	}

	public LabeledSentence(String path, Random rng) {
		totalCount = 0;
		filesByLabel = new HashMap<String, List<String>>();
		normList = new ArrayList<String>();
		negList = new ArrayList<>();
		BufferedReader buffered = null;
		try {
			buffered = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
			String line = buffered.readLine();
			while (line != null) {
				String[] lines = line.split("\t");
				String label = lines[0];
				String contennt = lines[1];
				if ("1".equalsIgnoreCase(label)) {
					normList.add(contennt);
				} else if ("0".equalsIgnoreCase(label)) {
					negList.add(contennt);
				}
				totalCount++;
				line = buffered.readLine();
			}
			buffered.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("totalCount is:" + totalCount);
		filesByLabel.put("1", normList);
		filesByLabel.put("0", negList);
		this.rng = rng;
		if (rng == null) {
			order = null;
		} else {
			order = new int[totalCount];
			for (int i = 0; i < totalCount; i++) {
				order[i] = i;
			}
			RandomUtils.shuffleInPlace(order, rng);
		}
		allLabels = new ArrayList<>(filesByLabel.keySet());
		Collections.sort(allLabels);
		Map<String, Integer> labelsToIdx = new HashMap<>();
		for (int i = 0; i < allLabels.size(); i++) {
			labelsToIdx.put(allLabels.get(i), i);
		}
		sentenslist = new CompactHeapStringList();
		labelIndexes = new int[totalCount];
		int position = 0;
		for (Map.Entry<String, List<String>> entry : filesByLabel.entrySet()) {
			int labelIdx = labelsToIdx.get(entry.getKey());
			for (String f : entry.getValue()) {
				sentenslist.add(f);
				labelIndexes[position] = labelIdx;
				position++;
			}
		}
	}

	@Override
	public boolean hasNext() {
		return cursor < totalCount;
	}

	@Override
	public Pair<String, String> nextSentence() {
		int idx;
		if (rng == null) {
			idx = cursor++;
		} else {
			idx = order[cursor++];
		}
		;
		String label = allLabels.get(labelIndexes[idx]);
		String sentence;
		sentence = sentenslist.get(idx);
		return new Pair<>(sentence, label);
	}

	@Override
	public void reset() {
		cursor = 0;
		if (rng != null) {
			RandomUtils.shuffleInPlace(order, rng);
		}
	}

	@Override
	public int totalNumSentences() {
		return totalCount;
	}

	@Override
	public List<String> allLabels() {
		return allLabels;
	}

	@Override
	public int numLabelClasses() {
		return allLabels.size();
	}
}
