from pyspark.sql import SparkSession
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import avg
import geopandas as gpd
import folium

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("WHO ART Coverage Analysis") \
    .getOrCreate()

# Step 2: Load the CSV data
csv_path = "C:/Users/user/Downloads/OneDrive/Desktop/Pyspark/data.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at path: {csv_path}")
else:
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Step 3: Show the first few rows
    df.show(5)
    df.printSchema()

    # Step 4: Count the number of rows
    print(f"Total Records: {df.count()}")

    # Step 5: Filter and select relevant columns
    df_filtered = df.select("Location", "Period", "FactValueNumeric")
    df_filtered.show(10)

    # Step 6: Calculate average ART coverage by location
    df_avg = df_filtered.groupBy("Location").agg(avg("FactValueNumeric").alias("Avg_ART_Coverage"))
    df_avg.show(10)

    # Step 7: Save the processed data
    processed_data_path = "C:/Users/user/Downloads/OneDrive/Desktop/Pyspark/data/processed_art_coverage.csv"
    df_avg.write.mode("overwrite").csv(processed_data_path, header=True)

    # Step 8: Load the processed data
    df_avg = spark.read.csv(processed_data_path, header=True, inferSchema=True)

    # Step 9: Show sample data
    df_avg.show(10)

    # Step 10: Convert to Pandas DataFrame
    df_pandas = df_avg.toPandas()

    # Step 11: Convert Avg_ART_Coverage column to numeric
    df_pandas["Avg_ART_Coverage"] = pd.to_numeric(df_pandas["Avg_ART_Coverage"], errors="coerce")

    # Step 12: Drop rows with NaN values
    df_pandas = df_pandas.dropna()

    # Step 13: Sort values
    df_pandas = df_pandas.sort_values(by="Avg_ART_Coverage", ascending=False)

    # Step 14: Plot the data
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df_pandas["Location"][:15], y=df_pandas["Avg_ART_Coverage"][:15], palette="viridis", hue=df_pandas["Location"][:15], dodge=False)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Country")
    plt.ylabel("Avg ART Coverage (%)")
    plt.title("Top 15 Countries by ART Coverage")
    plt.legend([],[], frameon=False)  # Hide the legend
    plt.show()

    # Additional Plot: Distribution of ART Coverage
    plt.figure(figsize=(10, 5))
    sns.histplot(df_pandas["Avg_ART_Coverage"], bins=20, kde=True, color="blue")
    plt.xlabel("Avg ART Coverage (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of ART Coverage")
    plt.show()

    # Scatter Plot: ART Coverage vs. Period (2015-2024)
    df_filtered_pandas = df_filtered.toPandas()
    df_filtered_pandas["Period"] = pd.to_datetime(df_filtered_pandas["Period"], format="%Y")
    df_filtered_pandas = df_filtered_pandas[(df_filtered_pandas["Period"].dt.year >= 2015) & (df_filtered_pandas["Period"].dt.year <= 2024)]
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="Period", y="FactValueNumeric", data=df_filtered_pandas)
    plt.xticks(rotation=45)
    plt.xlabel("Year")
    plt.ylabel("ART Coverage (%)")
    plt.title("ART Coverage vs. Period (2015-2024)")
    plt.show()

    # Convert PySpark DataFrame to Pandas
    df_pandas = df_filtered.select("Period", "Location", "FactValueNumeric").toPandas()

    # Convert 'Period' column to datetime
    df_pandas["Period"] = pd.to_datetime(df_pandas["Period"], format="%Y")

    # Compute average ART coverage per country
    country_avg = df_pandas.groupby("Location")["FactValueNumeric"].mean()

    # Define a threshold for high ART coverage (e.g., above 50%)
    high_prevalence_countries = country_avg[country_avg > 50].index.tolist()

    # Filter the dataset for only high-prevalence countries
    df_high_prevalence = df_pandas[df_pandas["Location"].isin(high_prevalence_countries)]

    # Group by Year and Location for plotting
    df_trend = df_high_prevalence.groupby(["Period", "Location"]).mean().reset_index()

    # Plot ART Coverage Trends in High-Prevalence Countries
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_trend, x="Period", y="FactValueNumeric", hue="Location", marker="o")
    plt.xlabel("Year")
    plt.ylabel("ART Coverage (%)")
    plt.title("ART Coverage Trends in High-Prevalence Countries")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # Load world map from local directory
    world = gpd.read_file("C:/Users/user/Downloads/OneDrive/Desktop/Pyspark/ne_110m_admin_0_countries.shp")

    # Print the columns of the world GeoDataFrame
    print(world.columns)

    # Convert ART coverage data to Pandas
    df_geo = df_filtered.select("Location", "FactValueNumeric").toPandas()

    # Merge with world map using the correct column name
    world = world.merge(df_geo, left_on="ADMIN", right_on="Location", how="left")

    # Create a Folium Map
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Add choropleth layer
    folium.Choropleth(
        geo_data=world,
        name="choropleth",
        data=world,
        columns=["Location", "FactValueNumeric"],
        key_on="feature.properties.ADMIN",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="ART Coverage (%)",
    ).add_to(m)

    # Display map
    m.save("art_coverage_map.html")
    print("Map saved as art_coverage_map.html")

    # Host address to show visualizations
    print("Visualizations are ready. Please check the generated plots and the map saved as 'art_coverage_map.html'.")