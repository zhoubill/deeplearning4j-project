package com.yling.apple.articleclassification;

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
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.nd4j.linalg.collection.CompactHeapStringList;

public class TxtLabelAwareIterator implements LabelAwareIterator {

	private int totalCount;
	private Map<String, List<String>> filesByLabel;
	private List<String> cowList;
	private List<String> dugList;
	private List<String> pigList;
	private List<String> sheepList;
	private final List<String> sentenslist;
	private final int[] labelIndexes;
	private final Random rng;
	private final int[] order;
	private final List<String> allLabels;
	private LabelsSource source;
	private int cursor = 0;

	public TxtLabelAwareIterator(String path) {
		this(path, new Random());
	}

	public TxtLabelAwareIterator(String path, Random rng) {
		totalCount = 0;
		filesByLabel = new HashMap<String, List<String>>();
		cowList = new ArrayList<String>();
		dugList = new ArrayList<>();
		pigList = new ArrayList<>();
		sheepList = new ArrayList<>();
		BufferedReader buffered = null;
		try {
			buffered = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
			String line = buffered.readLine();
			while (line != null) {
				String[] lines = line.split(",");			
				String label = lines[0];
				String contennt = lines[1];
				if ("10".equalsIgnoreCase(label)) {
					cowList.add(contennt);
				} else if ("0".equalsIgnoreCase(label)) {
					dugList.add(contennt);
				} else if("1".equalsIgnoreCase(label)) {
					pigList.add(contennt);
				} else if("2".equalsIgnoreCase(label)) {
					sheepList.add(contennt);
				}
				totalCount++;
				line = buffered.readLine();
			}
			buffered.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		filesByLabel.put("10", cowList);
		filesByLabel.put("0", dugList);
		filesByLabel.put("1", pigList);
		filesByLabel.put("2", sheepList);
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
		source = new LabelsSource(allLabels);
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
	public LabelledDocument next() {
		return nextDocument();
	}

	@Override
	public boolean hasNextDocument() {
		return hasNextDocument();
	}

	@Override
	public LabelledDocument nextDocument() {
		LabelledDocument document = new LabelledDocument();
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
		document.setContent(sentence);
		document.addLabel(label);
		return document;
	}

	@Override
	public void reset() {
		cursor = 0;
		if (rng != null) {
			RandomUtils.shuffleInPlace(order, rng);
		}
	}

	@Override
	public LabelsSource getLabelsSource() {
		return source;
	}

	@Override
	public void shutdown() {
	}
}
