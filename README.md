# Waze User Retention Analytics 🚘

**Waze User Churn Prevention Dashboard**  
This project enables data-driven insights for understanding and preventing user churn, focusing on optimizing user retention through behavioral pattern analysis.

### [Live Dashboard](https://haproxy-traffic-splitter/views/WazeUserRetentionData/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

![Dashboard Overview](waze-retention-dashboard.png)

---

## Executive Summary & Key Findings 📄
![Executive Summary](waze-executive-summary.png)

---

### Strategic Insights: User Behavior Patterns 🔎

Our analysis revealed key patterns in user churn behavior:

1. **Usage Intensity (82% Retention vs. 18% Churn)**: ❤️‍🔥
   - **Churned Users**: Higher intensity (~3 more drives/month)
   - **Retained Users**: More consistent usage patterns
   - **Strategic Focus**: Balance between engagement and sustainability

2. **Activity Concentration (Key Metrics)**: ↕️
   - **Drive Patterns**: Churned users average 698km per driving day
   - **Time Distribution**: Retained users show 2x more active days
   - **Resource Focus**: Target high-intensity users with specialized features

3. **Platform Distribution**: ↔️
   - 64.48% iPhone users
   - 35.52% Android users
   - **Platform Impact**: No significant difference in churn rates

### Business Impact 💥
- **Goal**: Reduce user churn rate (currently 18%)
- **Potential Impact**: Target high-risk user segments
- **Resource Allocation Model**: Focus on user experience optimization

---

## Statistical Analysis: Device Impact on User Engagement 📊

### A/B Testing for Business Decision-Making 
![Device Usage Analysis](waze-statistical-analysis.png)

We conducted hypothesis testing to determine if device type affects user engagement:

1. **Research Question**: 🔍
   - Is there a statistically significant difference in ride frequency between iPhone and Android users?
   - **Null Hypothesis**: No difference in mean rides between platforms
   - **Alternative Hypothesis**: Significant difference exists in mean rides

2. **Statistical Insights**: 📈
   - Two-sample t-test performed (p-value = 0.1434)
   - Failed to reject null hypothesis at α = 0.05
   - **Key Finding**: Device type does not significantly impact ride frequency

3. **Business Implications**: 💡
   - Platform-agnostic user experience successfully maintained
   - Resource allocation should focus on usage patterns rather than device-specific solutions
   - Consistent cross-platform performance validates current development approach

### Implementation Strategy 📋
- Develop retention strategies focusing on usage frequency rather than device type
- Maintain cross-platform consistency in future feature development
- Weight device type appropriately in churn prediction models

---

## Phase 2: Advanced Regression Analysis 🔬

### Project Overview
Following our initial exploratory analysis, we conducted a comprehensive binomial logistic regression model to predict user churn with greater precision.

### Model Performance 📊
- **Dataset Size**: 14,299 users
- **Model Accuracy**: 82.55%
- **Precision**: 54.44%
- **Recall**: 9.66%

### Key Insights from Regression Analysis 🧠

1. **Engagement Intensity**
   - Each additional day a user opens the app reduces churn risk by ~10%
   - **Strategic Recommendation**: Develop features encouraging daily app engagement (traffic updates, gas prices, road alerts)

2. **User Segmentation**
   - Professional Drivers: 7.6% churn rate
   - Non-Professional Drivers: 19.9% churn rate
   - **Strategic Recommendation**: Create segment-specific retention tactics

3. **Proactive Retention Strategies**
   - Develop an early warning system to identify at-risk users
   - Focus on enhancing features that demonstrate value during shorter, routine drives

### Next Steps 🚀
- Improve model predictive capability
- Conduct qualitative user research
- Develop personalized re-engagement strategies
- Explore advanced machine learning techniques to enhance churn prediction

---

## Project Documentation 📄

### Business Intelligence Documents 📑
- [Strategy Document](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-strategy-doc.pdf) (PDF)
- [Project & Stakeholder Requirements](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-project-requirements.pdf) (PDF)
- [EDA Results](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-data-summary.pdf) (PDF)
- [Statistical Analysis Report](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-statistical-report.pdf) (PDF)
- [Dashboard Mockup](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-dashboard-mockup.png) (Image)

### Data Analysis Process 📶

 **Data Files** 📂
- [Waze User Activity Data](https://github.com/mslawsky/waze-user-analytics/raw/main/waze_dataset.csv)
- [Churn Analysis Results](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-data-summary1.png)
- [Platform Usage Patterns](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-data-summary2.png)
- [Combined Analysis Results](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-data-summary2.png)

---

## Dashboard Development 📊

1. **Data Integration & Cleaning** 💾
   - Standardized user activity metrics
   - Validated data completeness (700 records with missing labels addressed)
   - Normalized driving metrics
   - Cross-referenced device data

2. **Metric Development** 📈
   - User Activity Patterns
   - Drive Intensity Metrics
     * Kilometers per drive
     * Drives per active day
     * Total activity days
   - Platform Usage Statistics
   - Churn Probability Indicators

3. **Visualization Strategy** 🖼️
   - User behavior pattern tracking
   - Cross-platform comparison
   - Temporal usage analysis
   - Churn risk indicators

4. **Statistical Analysis** 📉
   - Hypothesis test formulation
   - Descriptive statistics computation
   - Two-sample t-testing methodology
   - Statistical significance interpretation

---

### Implementation Recommendations 📋

1. **Immediate Actions** ✅
   - Develop targeted retention strategies for high-intensity users
   - Implement early warning system for churn risk patterns
   - Create specialized features for super-users (potential long-haul drivers)

2. **Resource Optimization** ➕
   - Platform-specific engagement programs
   - Usage pattern-based feature development 
   - Enhanced user experience for power users

---

## Contact ✉️

For inquiries about this analysis:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© Melissa Slawsky 2025. All Rights Reserved.  
This repository contains proprietary analysis.

**Published Project URL**: [Waze User Retention Dashboard](https://haproxy-traffic-splitter/views/WazeUserRetentionData/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

### Additional Technical Documentation 📄

1. **Python Analysis Notebooks**
   - [Initial Data Exploration](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-project-lab.pdf) (PDF)
   - [User Behavior Analysis](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-project-lab.py) (PY)
   - [Statistical Hypothesis Testing](https://github.com/mslawsky/waze-user-analytics/raw/main/waze-hypothesis-test.py) (PY)
   - [Churn Prediction Model](notebook-link) (Completed)

2. **Data Dictionary**
   ```python
   variables = {
       'label': 'User retention status (churned/retained)',
       'sessions': 'Number of app sessions',
       'drives': 'Number of drives recorded',
       'total_sessions': 'Aggregate session count',
       # Existing and new variables added
   }
