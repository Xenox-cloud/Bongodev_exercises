# Earthquake Decision Support Dashboard

This project is an interactive dashboard for **earthquake monitoring and decision support**, built using **Streamlit**. It processes earthquake datasets, detects aftershock clusters, visualizes risk trends, and provides decision-making metrics for geologists and disaster management teams.

---

## Features

- **Upload CSV Dataset**: Supports CSV files with `time, place, latitude, longitude, depth, magnitude`.
- **Executive Summary**: Displays total events, critical events, last 24h activity, average and maximum magnitude.
- **Alerts**: Highlights high-risk earthquakes (M ≥ 4.5) in the last 24 hours.
- **Trends & Charts**:
  - Risk distribution
  - Hourly event counts
  - Daily event trends
  - Magnitude over time
  - Energy Released Over Time
- **Hazard Mapping**: Interactive map of earthquake locations with magnitude visualization.
- **Aftershock Clusters**: Detects events within 24h and 50 km of previous events; shows map and cluster summary.
- **Decision Support**: Recommended actions based on recent seismic activity.
- **Filters**: Filter by time window, magnitude, depth, risk level, and tectonic type.
- **Data Export**: Download filtered dataset as CSV.

---

## Dataset Format

The CSV contains:

| Column     | Description                       |
|------------|-----------------------------------|
| `time`     | Event timestamp (`YYYY-MM-DD HH:MM:SS.sss`) |
| `place`    | Location description               |
| `latitude` | Latitude coordinate                |
| `longitude`| Longitude coordinate               |
| `depth`    | Depth in km                        |
| `magnitude`| Magnitude of the earthquake        |

---

## How to Run

1. Install dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn

