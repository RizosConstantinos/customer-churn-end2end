import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)

def numeric_correlation(df):
    """
    Plot correlation matrix for numeric features, excluding 'customerID'.
    """
    # Select numeric columns, excluding 'customerID'
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['customerID'], errors='ignore')
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()

def plot_churn_distribution(df):
    """
    Plot Customer Churn distribution with counts and percentages.
    Bars are colored red for churned customers, green for retained.
    """
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100

    # Prepare DataFrame for proper hue usage in seaborn
    plot_df = pd.DataFrame({
        'Churn': churn_counts.index,
        'Count': churn_counts.values
    })

    colors = {"No": "green", "Yes": "red"}

    # Barplot with hue
    sns.barplot(
        data=plot_df, x='Churn', y='Count', hue='Churn', dodge=False,
        palette=colors, legend=False
    )

    # Annotate bars with counts and percentages
    for i, (count, perc) in enumerate(zip(churn_counts.values, churn_percent.values)):
        plt.text(i, count + 5, f"{count} ({perc:.1f}%)", ha='center', fontweight='bold')

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.ylim(0, churn_counts.max() * 1.2)
    plt.show()

def feature_distribution_by_churn(df):
    """
    Plot stacked bar charts of categorical features by Churn (as percentages)
    and display percentages on the bars. Skip customerID.
    """
    cat_features = df.select_dtypes(include=['object']).drop(columns=['customerID','TotalCharges','Churn','gender','PhoneService','MultipleLines','StreamingMovies','DeviceProtection','OnlineSecurity','Partner','StreamingTV','OnlineBackup','Dependents'], errors='ignore')

    for col in cat_features:
        # Compute percentages
        churn_ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100  # percentages
        churn_ct = churn_ct[['No', 'Yes']] if 'No' in churn_ct.columns else churn_ct

        # Plot
        ax = churn_ct.plot(
            kind='bar',
            stacked=True,
            colormap='coolwarm',
            figsize=(7,5)
        )
        plt.title(f"{col} Distribution by Churn")
        plt.ylabel("Percentage")
        plt.xlabel(col)

        # Add percentage labels on each bar
        for i, idx in enumerate(churn_ct.index):
            bottom = 0
            for j, val in enumerate(churn_ct.columns):
                perc = churn_ct.loc[idx, val]
                if perc > 0:
                    ax.text(
                        i, 
                        bottom + perc / 2,  # middle of segment
                        f"{perc:.0f}%", 
                        ha='center', 
                        va='center',
                        color='white', 
                        fontsize=10,
                        fontweight='bold'
                    )
                    bottom += perc

        plt.xticks(rotation=0)
        plt.legend(title='Churn')
        plt.tight_layout()
        plt.show()


def check_missing_values(df: pd.DataFrame, placeholders=None):
    """Check for missing or placeholder values."""
    if placeholders is None:
        placeholders = ["NA", "N/A", "Unknown"]
    missing = (df.isnull() | (df == " ") | df.isin(placeholders)).sum()
    return missing
    
def boxplot_by_churn(df, feature, target="Churn"):
    """
    Plot boxplot of a numerical feature grouped by churn.
    """

    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df,
        x=target,
        y=feature,
        hue=target,
        palette={"Yes": "#e74c3c", "No": "#2ecc71"},
        legend=False
    )

    plt.title(f"{feature} distribution by {target}")
    plt.xlabel(target)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()