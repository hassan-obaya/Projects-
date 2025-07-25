# Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# Load CSV Path
path = "/kaggle/input/famines-by-level-of-gdp-per-capita-at-the-time/famines-by-level-of-gdp-per-capita-at-the-time.csv"

# Function: Load & Clean Dataset
def load_and_clean(path):
    """
    Load the famine dataset and perform basic cleaning.
    """
    df = pd.read_csv(path)
    df['Country'] = df['Entity'].str.replace(r'\s+\d.*$', '', regex=True)
    df['GDP per capita'] = df['GDP per capita'].fillna(df['GDP per capita'].mean())
    df['Decade'] = (df['Year'] // 10 * 10).astype(int)
    return df

# Load & Prepare Data
df = load_and_clean(path)

# Function: Display Overview
def display_overview(df):
    """
    Show dataset summary, structure, and null/duplicate diagnostics.
    """
    print("Data Overview")
    print(df.info(), "\n")
    display(df.head().T)
    display(df.describe(include='all').T)
    print("Nulls:\n", df.isnull().sum(), "\nDuplicates:", df.duplicated().sum())

# Explore Dataset
display_overview(df)

# Set Global Plot Styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Function: Top N Countries by Famine Deaths
def bar_top_countries(df, n=10):
    """
    Horizontal bar chart of countries with highest famine deaths.
    """
    top = df.groupby('Country')['Deaths from famines'].sum().nlargest(n)
    sns.barplot(x=top.values, y=top.index, palette='magma')
    for i, v in enumerate(top.values):
        plt.text(v + top.values.max() * 0.01, i, f"{v:,}", va='center')
    plt.title(f"Top {n} Countries by Famine Deaths")
    plt.xlabel("Total Deaths")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Visualize Top 10 Famine-Impacted Countries
bar_top_countries(df, n=10)

# Function: Stacked Bar by Decade & Country
def plot_top_countries_stacked(data, top_n=6):
    """
    Stacked bar chart by decade for top countries + 'Other'.
    """
    top_countries = data.groupby('Country')['Deaths from famines'].sum().nlargest(top_n).index.tolist()
    data['Group'] = data['Country'].where(data['Country'].isin(top_countries), 'Other')
    agg = data.groupby(['Decade', 'Group'])['Deaths from famines'].sum().unstack(fill_value=0).loc[:, top_countries + ['Other']]
    
    decades = agg.index
    x = np.arange(len(decades))
    bottom = np.zeros(len(decades))
    colors = sns.color_palette("tab10", n_colors=len(agg.columns))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, country in enumerate(agg.columns):
        vals = agg[country].values / 1e6
        ax.bar(x, vals, bottom=bottom/1e6, color=colors[i], label=country, edgecolor='white', linewidth=0.7)
        bottom += agg[country].values

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in decades], rotation=45)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Deaths (millions)")
    ax.set_title(f"Famine Deaths by Decade: Top {top_n} Countries + Other", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}M"))
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    sns.despine(left=True, bottom=True)
    
    totals = bottom / 1e6
    for idx, total in enumerate(totals):
        ax.text(idx, total + 0.4, f"{total:.1f}M", ha='center', va='bottom', fontweight='bold', fontsize=7)

    ax.legend(title="Country", bbox_to_anchor=(1.02, 0.5), loc='center left', ncol=1, frameon=False, title_fontsize=12, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# Visualize Stacked Bar by Decade
plot_top_countries_stacked(df, top_n=6)

# Function: Scatter + Regression Line
def scatter_regression(df):
    """
    Plot GDP vs Famine Deaths with regression fit.
    """
    X = df[['GDP per capita']]
    y = df['Deaths from famines']
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    plt.scatter(df['GDP per capita'], y, alpha=0.6, edgecolor='k')
    plt.plot(df['GDP per capita'], y_pred, color='crimson', lw=2)
    plt.title("GDP per Capita vs Famine Deaths")
    plt.xlabel("GDP per Capita (USD)")
    plt.ylabel("Deaths from Famines")
    plt.tight_layout()
    plt.show()

# GDP vs Famine Deaths
scatter_regression(df)

# Function: Stacked Area by Cause
def stacked_area_causes(df):
    """
    Stacked area chart of deaths by cause over decades.
    """
    agg = df.groupby(['Decade', 'Principal cause'])['Deaths from famines'].sum().unstack(fill_value=0)
    decades = agg.index
    causes = agg.columns
    colors = sns.color_palette("tab10", len(causes))
    plt.stackplot(decades, agg.T, labels=causes, colors=colors, alpha=0.8)
    plt.legend(title="Cause", loc='upper left')
    plt.title("Famine Deaths by Cause (Stacked Area)")
    plt.xlabel("Decade")
    plt.ylabel("Deaths")
    plt.xticks(decades, [f"{d}s" for d in decades], rotation=45)
    plt.tight_layout()
    plt.show()

# Area Plot by Cause
stacked_area_causes(df)

# Function: Small Multiples for Each Cause
def small_multiples_causes(df):
    """
    Line plot per cause across decades in subplots.
    """
    df['Decade'] = (df['Year'] // 10 * 10).astype(int)
    agg = df.groupby(['Decade', 'Principal cause'])['Deaths from famines'].sum().unstack(fill_value=0)
    decades = agg.index.tolist()
    causes = agg.columns.tolist()
    fig, axes = plt.subplots(len(causes), 1, figsize=(10, 2 * len(causes)), sharex=True)
    colors = sns.color_palette("tab10", n_colors=len(causes))
    for ax, cause, color in zip(axes, causes, colors):
        ax.plot(decades, agg[cause], marker='o', color=color, linewidth=2)
        ax.set_ylabel(cause, fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.text(decades[-1], agg[cause].iloc[-1], f"{int(agg[cause].iloc[-1]):,}", va='center', ha='left', fontsize=9, color=color)
    axes[-1].set_xticks(decades)
    axes[-1].set_xticklabels([f"{d}’s" for d in decades], rotation=45)
    axes[-1].set_xlabel("Decade", fontsize=12)
    fig.suptitle("Famine Deaths Over Time — by Principal Cause", y=0.92, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Subplot Trends by Cause
small_multiples_causes(df)

# Function: Choropleth Map of Famine Deaths
def choropleth_map(df):
    """
    Interactive world map of famine deaths by country.
    """
    deaths = df.groupby('Country')['Deaths from famines'].sum().reset_index()
    fig = px.choropleth(
        deaths, 
        locations='Country', 
        locationmode='country names',
        color='Deaths from famines',
        title='Famine Deaths by Country',
    )
    fig.show()

# Display Choropleth Map
choropleth_map(df)
