🎬 Can we predict a blockbuster before it hits theaters?

We built a machine learning model that predicts whether a movie will earn $100M+ using the TMDB 5000 dataset, TMDB API data, and critic scores we scraped ourselves.

🔧 What we did:
• Cleaned messy JSON columns & missing budgets
• Engineered features: director track record, star power, sequel flag, release season, genres
• Trained Logistic Regression, Random Forest & XGBoost
• Used a time-based split (train < 2010, test ≥ 2010) to mimic real-world prediction

📊 What we found:
• Best model: [MODEL] → ROC-AUC [X], F1 [Y]
• Critic scores [improved / barely changed] ROC-AUC from [X] to [Y]
• Top drivers: [budget, sequel, director history]
• ⚠️ Including post-release features (votes, popularity) pushed accuracy to ~100% — pure data leakage. A great reminder that a "perfect" model usually means a mistake.

💡 Biggest lesson: how you split and which features you allow matters more than which algorithm you pick.

📂 Code & full results: [GitHub link]

Team: [@Person1] [@Person2] [@Person3] [@Person4] [@Person5]

#MachineLearning #DataScience #Python #XGBoost #DataAnalytics #Portfolio
