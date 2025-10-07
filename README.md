# Lead Scoring & Prioritization Tool (Caprae Capital Pre-Work)

## 🎯 **The Problem We Solve**

The challenge with lead generation tools is that they often deliver a high *volume* of data without a clear way to determine which leads are worth a sales team's time. Our primary goal is to turn raw data into **actionable intelligence.**

This tool is a high-impact enhancement for any lead generation platform, allowing sales teams to **focus exclusively on high-potential targets** that align with Caprae Capital’s investment strategy.

## ✨ **The Solution: Intelligent Lead Prioritization**

This tool assigns two crucial metrics to every lead:

### 1. **Lead Score (0-100):** *How valuable is the company?*
This score determines the overall quality and potential of a lead based on criteria that matter to the business.

* **Customizable Weights:** Sales leaders can dynamically adjust the importance of key factors like **Company Size**, **Estimated Revenue**, **Engagement Signals** (e.g., recent funding), and **Contact Title Relevance** (e.g., CEO vs. Manager) to instantly align the prioritization with the current sales strategy.
* **Result:** Ensures the sales team is chasing the companies most likely to result in a deal.

### 2. **Confidence Score (0-100):** *How actionable is the lead?*
This metric assesses the reliability and completeness of the contact information.

* **Focus on Completeness:** It measures the presence of key contact points (Email, Phone, LinkedIn URL).
* **Result:** Reduces wasted time by ensuring the team only contacts leads with verifiable and complete data, acting as a built-in data quality and validation feature (bonus point feature).

---

## 💻 **How to Use the Tool (Quick Start)**

This tool is designed to be robust and easy to deploy in any environment.

### **Option 1: Interactive Dashboard (Recommended for Sales Leaders)**
For real-time control and visualization, the tool runs as an interactive web application, allowing immediate adjustment of scoring weights.

1.  **Prerequisites:** Python and necessary libraries (listed below).
2.  **Run:** `streamlit run leadscore.py`

### **Option 2: Console Mode (For Testing and Automated Systems)**
If a web interface isn't needed, the core logic runs independently, providing a simple, text-based output and saving results.

* **Output:** Generates a prioritized `.csv` file of leads and a summary HTML file with charts (saved to `/mnt/data/`).
* **Run:** `python leadscore.py`

### **Setup (For IT/Development Teams)**
The tool requires standard Python libraries:

```bash
pip install streamlit pandas numpy plotly
