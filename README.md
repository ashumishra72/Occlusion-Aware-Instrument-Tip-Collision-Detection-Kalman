# Occlusion-Aware-Instrument-Tip-Collision-Detection-Kalman

The project detects instrument tip collisions in a pattern cutting training video. It includes a trained YOLO11n segmentation detector, human collision annotations, feature extraction, classifier evaluation, Kalman trajectories, event extraction, diagnostic figures, and research reports. The available recording supplies 3,917 sampled frames, including 558 positive frames and 64 collision annotation rows.

The historical phase compared XGBoost, LightGBM, CatBoost, Random Forest, SVM, LSTM, GRU, and TCN using 43 features. Random Forest achieved approximately 0.86 ROC AUC, while XGBoost achieved approximately 0.53 frame F1. However, short temporal groups and threshold selection using pooled evaluation labels can make performance appear optimistic.

The revised study reuses the video, detector observations, and annotations. It introduces independent Kalman filters, uncertainty and missingness features, short trajectory histories, event preserving groups, and validation based selection. Therefore, historical scores provide context; only comparisons made under the same revised protocol support conclusions about the tested changes.

The revised experiment compares original features, Kalman features, and Kalman features with two seconds of history, each using Random Forest and XGBoost. Kalman tracking predicts missing positions temporarily and records uncertainty. It does not confirm physical contact or establish position accuracy during an unobserved interval.

With frame focused selection, adding trajectory history raises Random Forest frame F1 from 0.319 to 0.348. However, event detection does not automatically improve: neighboring positive frames can merge several annotated contacts into one predicted event.

Event focused tuning changes the threshold and merge gap while preserving the outer classifier scores. For Kalman plus history with Random Forest, event F1 increases from 0.342 to 0.354, but detected events fall from 26 to 23 as false events fall from 62 to 43. The original feature Random Forest remains stronger at 0.392 event F1. More independently annotated videos are needed before claiming reliable generalization. These conclusions describe this recording, not all procedures.
