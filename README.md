# Mental Health Prediction from Social Media Responses

This project predicts a user's mental health impact level based on their social media behavior using a deep learning model (BERT + BiLSTM).

## Project Structure
- `train_and_evaluate.py` — Train and evaluate the model on real-world survey data.
- `project.py` — Launch a GUI (Tkinter app) for self-assessment.
- `saved_models/` — Stores the trained model.
- `assets/` — Stores the background image for the GUI.
- `report/` — Contains the final project report in LaTeX format.

## Setup Instructions

1. **Install Dependencies**  
Make sure you have Python 3.8+ installed.  
Then install required libraries:
```bash
pip install -r requirements.txt

# 📊 Social Media and Mental Health Dataset – README

## 📘 Introduction

The **Social Media and Mental Health** dataset provides a valuable resource for understanding the relationship between **social media usage** and **mental health outcomes** across a diverse population. As social media becomes increasingly ingrained in daily life, concerns about its potential negative impact on mental well-being have grown. This dataset is designed to help researchers, data scientists, and psychologists analyze behavioral patterns and assess how online engagement may contribute to mental health challenges such as anxiety, depression, or distraction.

---

## 📂 Dataset Source

- **Author**: [Souvik Ahmed](https://www.kaggle.com/souvikahmed071)
- **Kaggle Link**: [Click here to access](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health)

---

## 📄 Dataset Description

The dataset is collected via **self-reported questionnaires** and contains responses from individuals about their **social media usage habits**, **emotional well-being**, and **mental health awareness**. It contains **20+ features**, which can be grouped into three main categories:

### 1. **Demographic Attributes**
- `1_what_is_your_age`
- `2_gender`
- `3_relationship_status`
- `4_occupation_status`
- `5_what_type_of_organizations_are_you_affiliated_with`

### 2. **Social Media Usage Patterns**
- `6_do_you_use_social_media`
- `7_what_social_media_platforms_do_you_commonly_use`
- `8_what_is_the_average_time_you_spend_on_social_media_every_day`
- `9_how_often_do_you_find_yourself_using_social_media_without_a_specific_purpose`
- `10_how_often_do_you_get_distracted_by_social_media_when_you_are_busy_doing_something`
- `11_do_you_feel_restless_if_you_havent_used_social_media_in_a_while`

### 3. **Mental Health Indicators**
- `12_on_a_scale_of_1_to_5_how_easily_distracted_are_you`
- `13_on_a_scale_of_1_to_5_how_much_are_you_bothered_by_worries`
- `14_do_you_find_it_difficult_to_concentrate_on_things`
- `15_on_a_scale_of_15_how_often_do_you_compare_yourself_to_other_successful_people_through_the_use_of_social_media`
- `16_following_the_previous_question_how_do_you_feel_about_these_comparisons_generally_speaking`
- `17_how_often_do_you_look_to_seek_validation_from_features_of_social_media`
- `18_how_often_do_you_feel_depressed_or_down`
- `19_on_a_scale_of_1_to_5_how_frequently_does_your_interest_in_daily_activities_fluctuate`
- `20_on_a_scale_of_1_to_5_how_often_do_you_face_issues_regarding_sleep`

---

## 🧠 Potential Applications

### 🔍 Exploratory Data Analysis
- Understand usage differences by age or gender.
- Correlate time spent with feelings of anxiety or concentration loss.

### 🧪 Predictive Modeling
- Predict emotional states using classification models.
- Identify user clusters with similar behavioral patterns.

### 📉 Sentiment & Text Analysis
- Apply NLP on open-ended responses for deeper insights.

---

## ✅ Ethical Considerations

- Avoid misrepresentation or overgeneralization.
- Respect data privacy and use only for research or academic purposes.
- Be cautious when drawing mental health conclusions.

---

## 🔧 Format & Size

- **Format**: CSV
- **Rows**: ~500
- **Columns**: 20+
- **Contains missing values**: Yes (especially in text responses)

---

## 🔗 Citation

```
Souvik Ahmed. (2022). Social Media and Mental Health [Dataset]. Kaggle. https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health
```

---

## 📞 Contact

For inquiries, visit the [Kaggle Dataset Page](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health).
