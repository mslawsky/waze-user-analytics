# Waze User Retention Analytics

**Waze User Churn Prevention Dashboard**  
This project enables data-driven insights for understanding and preventing user churn, focusing on optimizing user retention through behavioral pattern analysis.

### [Live Dashboard](https://haproxy-traffic-splitter/views/WazeUserRetentionData/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

![Dashboard Overview](waze-retention-dashboard.png)

---

## Executive Summary & Key Findings
![Executive Summary](waze-executive-summary.png)

---

### Strategic Insights: User Behavior Patterns

Our analysis revealed key patterns in user churn behavior:

1. **Usage Intensity (82% Retention vs. 18% Churn)**:
   - **Churned Users**: Higher intensity (~3 more drives/month)
   - **Retained Users**: More consistent usage patterns
   - **Strategic Focus**: Balance between engagement and sustainability

2. **Activity Concentration (Key Metrics)**:
   - **Drive Patterns**: Churned users average 698km per driving day
   - **Time Distribution**: Retained users show 2x more active days
   - **Resource Focus**: Target high-intensity users with specialized features

3. **Platform Distribution**:
   - 64.48% iPhone users
   - 35.52% Android users
   - **Platform Impact**: No significant difference in churn rates

### Business Impact
- **Goal**: Reduce user churn rate (currently 18%)
- **Potential Impact**: Target high-risk user segments
- **Resource Allocation Model**: Focus on user experience optimization

---

## Project Documentation

### Business Intelligence Documents
- [Strategy Document](https://github.com/mslawsky/waze-user-analytics/blob/main/waze-strategy-doc.pdf) (PDF)
- [Stakeholder Requirements](link) (PDF)
- [Project Requirements](link) (PDF)
- [EDA Results](link) (PDF)
- [Dashboard Mockup](link) (Image)

### Data Analysis Process

📂 **Data Files**
- [Waze User Activity Data](https://github.com/mslawsky/blob/main/waze_dataset.csv)
- [Churn Analysis Results](link)
- [Platform Usage Patterns](link)
- [Combined Analysis Results](link)

---

## Dashboard Development

1. **Data Integration & Cleaning**
  - Standardized user activity metrics
  - Validated data completeness (700 records with missing labels addressed)
  - Normalized driving metrics
  - Cross-referenced device data

2. **Metric Development**
  - User Activity Patterns
  - Drive Intensity Metrics
    * Kilometers per drive
    * Drives per active day
    * Total activity days
  - Platform Usage Statistics
  - Churn Probability Indicators

3. **Visualization Strategy**
  - User behavior pattern tracking
  - Cross-platform comparison
  - Temporal usage analysis
  - Churn risk indicators

### Implementation Recommendations

1. **Immediate Actions**
  - Develop targeted retention strategies for high-intensity users
  - Implement early warning system for churn risk patterns
  - Create specialized features for super-users (potential long-haul drivers)

2. **Resource Optimization**
  - Platform-specific engagement programs
  - Usage pattern-based feature development 
  - Enhanced user experience for power users

---

## Contact

For inquiries about this analysis:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© Melissa Slawsky 2025. All Rights Reserved.  
This repository contains proprietary analysis.

**Published Project URL**: [Waze User Retention Dashboard]([dashboard-link](https://haproxy-traffic-splitter/views/WazeUserRetentionData/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link))

---

### Additional Technical Documentation

1. **Python Analysis Notebooks**
  - [Initial Data Exploration](notebook-link)
  - [User Behavior Analysis](notebook-link)
  - [Churn Prediction Model](notebook-link)

2. **Data Dictionary**
  ```python
  variables = {
      'label': 'User retention status (churned/retained)',
      'sessions': 'Number of app sessions',
      'drives': 'Number of drives recorded',
      'total_sessions': 'Aggregate session count',
      # Add other variables
  }
