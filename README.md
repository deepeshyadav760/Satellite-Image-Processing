# 🌿 Carbon Sequestration in the Sundarbans Mangrove Forest and Chilika Lake Ecosystems

This repository contains the code, data processing pipeline, and analysis behind the research study titled:

> **Carbon Sequestration in the Sundarbans Mangrove Forest and Chilika Lake Ecosystems**  
> **Authors**: Deepesh Yadav, Venkat  
> **Institution**: Atria University  
> **Date**: March 14, 2025

---

## 🧠 Project Overview

This research investigates the **carbon sequestration capacity** of two ecologically important Indian ecosystems:

- **Sundarbans Mangrove Forest** *(West Bengal)*
- **Chilika Lake** *(Odisha)*

By leveraging **remote sensing datasets** (Landsat 8 and SRTM) and **machine learning models** (Random Forest and SARIMA), we aim to quantify, analyze, and forecast carbon sequestration patterns from **2018 to 2024**.

---

## 🔬 Key Objectives

- Estimate carbon stocks and CO₂ absorption capacity of the ecosystems
- Study spatial and seasonal variation in carbon sequestration
- Build predictive models for future projections
- Propose actionable policy recommendations for conservation

---

## 📊 Key Findings

### 🌳 Sundarbans Mangrove Forest
- **2024 Carbon Stock**: 24.37 million tonnes
- **Annual CO₂ Absorption**: 545,293.77 tonnes
- **Seasonal Peak**: Post-Monsoon
- **2029 Projection**: -1.16% reduction in both area and carbon stock

### 💧 Chilika Lake
- **Average Sequestration Rate**: 5.69 tons CO₂/ha/year
- **Annual Total CO₂ Sequestration**: ~258 million tons
- **High-Capacity Zones**: 24% of lake area
- **Seasonal Peak**: Summer
  

## 📦 Tech Stack

- **Programming**: Python  
- **Libraries**: `scikit-learn`, `statsmodels`, `GeoPandas`, `Rasterio`, `GDAL`  
- **Remote Sensing Platforms**: Google Earth Engine API  
- **Satellite Data**: Landsat 8 (OLI/TIRS), SRTM Elevation Data  
- **Vegetation Indices**:
  - **NDVI** (Normalized Difference Vegetation Index)
  - **NDWI** (Normalized Difference Water Index)
  - **MNDVI** (Modified NDVI for mangroves)
  - **EVI** (Enhanced Vegetation Index)
    

## 🧪 ML Model Performance

| Model         | MAE (tons)     | RMSE (tons)    | R² Score |
|---------------|----------------|----------------|----------|
| Random Forest | 30,056,795.64  | 5,177,802.94   | 0.2456   |
| SARIMA        | 17,531,306.20  | 34,661,381.87  | 0.6619   |

SARIMA outperformed Random Forest due to its ability to capture **seasonal trends**.


## 🦠 COVID-19 Impact

During the 2021 lockdown, the Sundarbans experienced a **notable decline in CO₂ absorption**, dropping from ~547,000 tonnes in 2020 to ~260,000 tonnes in 2021 due to **reduced atmospheric CO₂** availability. This highlights the sensitivity of carbon sinks to global activity patterns.


## 🧭 Policy Recommendations

### Sundarbans Mangroves
- ✅ Implement **community-based conservation** programs
- 🌐 Foster **India-Bangladesh transboundary collaboration**
- 💰 Enable **carbon credit incentives** for ecosystem preservation

### Chilika Lake
- 🌱 **Restore seagrass beds** by 25% by 2030
- 🔁 **Optimize freshwater inflows** and salinity levels
- 🚫 **Reduce agricultural runoff** through watershed buffer zones
  
## 📌 Future Work
- 🛰️ Integrate field measurements to enhance accuracy
- 🌪️ Analyze ecosystem resilience under **extreme weather events**
- 🔍 Develop high-resolution models under **climate change scenarios**
- 💬 Study **socioeconomic impacts** on carbon sequestration dynamics

## 📚 References
- Alongi, D. M. (2014). *Carbon cycling and storage in mangrove forests*. Annual Review of Marine Science, 6, 195-219.  
- Donato, D. C., et al. (2011). *Mangroves among the most carbon-rich forests in the tropics*. Nature Geoscience, 4(5), 293-297.  
- Simard, M., et al. (2019). *Mangrove canopy height globally related to precipitation, temperature and cyclone frequency*. Nature Geoscience, 12(1), 40-45.  
- Tang, W., et al. (2018). *Big geospatial data analytics for global mangrove biomass and carbon estimation*. Sustainability, 10(2), 472.  
- IPCC. (2022). *Climate Change 2022: Impacts, Adaptation and Vulnerability*. Sixth Assessment Report.

## 🤝 Acknowledgements
We thank **Atria University** for institutional support and mentorship, and express our gratitude to satellite data providers — **USGS Landsat** and **NASA SRTM** — for their open-access geospatial datasets.

> 🌏 *This project is a step toward climate resilience by harnessing data science and ecological intelligence to protect India’s blue carbon ecosystems.*
