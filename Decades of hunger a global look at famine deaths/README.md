# Decades of Hunger: A Global Look at Famine Deaths

## Description

This notebook provides an in-depth analysis of global famine data, examining trends in famine deaths over time, their principal causes, and the relationship with GDP per capita. It includes various visualizations to help understand the patterns and impacts of famines across different regions and decades.

## Dataset

The analysis is based on a dataset of famines, which includes information on the entity (country or region), year, GDP per capita, deaths from famines, and the principal cause. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/hassanobaya/famines-by-level-of-gdp-per-capita-at-the-time).

## Analyses

- **Data Loading and Cleaning**: Ingests the CSV data, extracts country names, imputes missing GDP values, and derives decades from years.
- **Data Overview**: Displays information about the dataset, including column types, sample rows, descriptive statistics, and checks for nulls and duplicates.
- **Top 10 Countries by Total Deaths**: Visualizes the countries with the highest cumulative famine deaths using a horizontal bar chart.
- **Famine Deaths Over Time**: Shows the trend of total famine deaths per decade with a line plot.
- **Principal Causes of Famines**: Uses a stacked bar chart to display the distribution of famine deaths by cause over decades.
- **GDP per Capita vs. Famine Deaths**: Scatter plot to explore the relationship between GDP per capita and famine deaths, including a regression line.
- **Anomaly Detection**: Identifies outliers in famine deaths using the Isolation Forest algorithm.
- **Comparison by Cause**: Small-multiple line plots to compare famine deaths over time for each principal cause.
- **Choropleth Map**: Interactive world map summarizing total famine deaths per country.

## Key Findings

- **Top Impacted Countries**: China, India, and Russia have experienced the highest cumulative famine deaths.
- **Trends Over Time**: Famine deaths peaked in the 1940s and have generally declined since, with some causes showing resurgence in recent decades.
- **Causes of Famines**: Government policies and armed conflicts have been significant contributors to famine deaths, especially in the mid-20th century.
- **GDP and Famines**: There is a noticeable inverse relationship between GDP per capita and famine deaths, though with some exceptions.
- **Geographical Distribution**: Asia and Africa have been the most affected continents, while developed regions show minimal famine impacts.

## Usage

This notebook can be run cell by cell in a Jupyter environment. It requires the following Python libraries:

- pandas
- numpy
- matplotlib
- plotly
- seaborn
- scikit-learn

If running on Kaggle, all dependencies are pre-installed, and the dataset is readily available.
