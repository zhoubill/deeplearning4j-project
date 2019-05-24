package com.yling.apple.appleprediction;

public class DailyData {
	//开盘价
    private double openPrice;
    //收盘价
    private double closeprice;
    //最高价
    private double maxPrice;
    //最低价
    private double minPrice;
    //成交量
    private double turnover;
    //成交额
    private double volume;
 
    public double getTurnover() {
 
        return turnover;
    }
 
    public double getVolume() {
        return volume;
    }
 
    public DailyData(){
 
    }
 
    public double getOpenPrice() {
        return openPrice;
    }
 
    public double getCloseprice() {
        return closeprice;
    }
 
    public double getMaxPrice() {
        return maxPrice;
    }
 
    public double getMinPrice() {
        return minPrice;
    }
 
    public void setOpenPrice(double openPrice) {
        this.openPrice = openPrice;
    }
 
    public void setCloseprice(double closeprice) {
        this.closeprice = closeprice;
    }
 
    public void setMaxPrice(double maxPrice) {
        this.maxPrice = maxPrice;
    }
 
    public void setMinPrice(double minPrice) {
        this.minPrice = minPrice;
    }
 
    public void setTurnover(double turnover) {
        this.turnover = turnover;
    }
 
    public void setVolume(double volume) {
        this.volume = volume;
    }
 
    @Override
    public String toString(){
        StringBuilder builder = new StringBuilder();
        builder.append("开盘价="+this.openPrice+", ");
        builder.append("收盘价="+this.closeprice+", ");
        builder.append("最高价="+this.maxPrice+", ");
        builder.append("最低价="+this.minPrice+", ");
        builder.append("成交量="+this.turnover+", ");
        builder.append("成交额="+this.volume);
        return builder.toString();
    }
}
