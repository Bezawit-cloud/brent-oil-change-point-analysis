# 🛢️ Brent Oil Change Point Dashboard

**Interactive Dashboard for Brent Oil Price Analysis**  

This project is a full-stack data visualization dashboard designed to help stakeholders explore how various events affect Brent oil prices. It combines a **Flask backend** for data APIs with a **React frontend** for interactive charts and event highlights.  

---

## 🚀 Key Features

- **Historical Price Trends** – View Brent oil prices over time.  
- **Event Highlights** – Visualize key events (green dots) and detected price change points (red dots).  
- **Interactive Filters** – Select date ranges to drill down into specific periods.  
- **Responsive Design** – Works seamlessly on desktop, tablet, and mobile.  
- **Data Insights** – Explore correlations between events, political decisions, and oil price volatility.  

---

## 🏗️ Tech Stack

- **Backend:** Python, Flask, Pandas, Flask-CORS  
- **Frontend:** React.js, Recharts  
- **Data:** CSV datasets with historical prices, change points, and event correlations  

---

## 📂 Project Structure

dashboard/
│
├── backend/
│ ├── app.py # Flask API server
│ ├── BrentOilPrices.csv # Historical price data
│ ├── change_points.csv # Detected change points
│ └── events.csv # Event correlation data
│
├── frontend/
│ ├── src/
│ │ ├── App.js # React dashboard UI
│ │ └── index.js
│ └── package.json
│
└── README.md

---

## ⚙️ Setup Instructions

### 1. Backend

1. Navigate to the backend folder:  
   ```bash
   cd dashboard/backend
conda create -n oil-dashboard python=3.10
conda activate oil-dashboard
pip install flask pandas flask-cors
python app.py
Backend APIs available at:

http://127.0.0.1:5000/api/historical

http://127.0.0.1:5000/api/change_points

http://127.0.0.1:5000/api/event_correlations
2. Frontend

Navigate to the frontend folder:

cd dashboard/frontend


Install npm packages:

npm install


Start the React app:

npm start


Open your browser at http://localhost:3000 to view the dashboard.

📊 Dashboard Usage

Date Filters: Pick a start and/or end date to view specific periods.

Visual Cues:

Red dots = Detected change points in oil price

Green dots = Events affecting prices

Interactive Insights: Hover over chart points for exact values.
## 💡 Why This Project Matters

This dashboard provides actionable insights for energy analysts, economists, and policy-makers. It combines real-world data with interactive visualization to make trends and correlations clear, helping stakeholders understand the impact of events on Brent oil prices.

## 🧰 Skills Demonstrated

- Full-stack development (Flask + React)  
- Data visualization and interactive charts with Recharts  
- Data processing and analysis using Pandas  
- API development and JSON data handling  
- Responsive web design and UX/UI principles


