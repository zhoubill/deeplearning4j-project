package com.yling.apple.appleprediction;

public class AppleDailyData {
	
	 //平均价
	 private double meanPrice;
	 //最高价
	 private double highPrice;
	 //最低价
	 private double lowPrice;
	 //ID
	 private String priceId;
	 //日期
	 private String dateStr;
	 
	public double getMeanPrice() {
		return meanPrice;
	}
	public void setMeanPrice(double meanPrice) {
		this.meanPrice = meanPrice;
	}
	public double getHighPrice() {
		return highPrice;
	}
	public void setHighPrice(double highPrice) {
		this.highPrice = highPrice;
	}
	public double getLowPrice() {
		return lowPrice;
	}
	public void setLowPrice(double lowPrice) {
		this.lowPrice = lowPrice;
	}
	public String getPriceId() {
		return priceId;
	}
	public void setPriceId(String priceId) {
		this.priceId = priceId;
	}
	public String getDateStr() {
		return dateStr;
	}
	public void setDateStr(String dateStr) {
		this.dateStr = dateStr;
	}
}
