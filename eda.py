"""
Phase 4: Exploratory Data Analysis (EDA)
Reads data/cleaned_dataset.csv (created in Phase 3) and generates
charts saved into a charts/ folder, plus prints summary stats.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

# Create a folder to save chart images
os.makedirs("charts", exist_ok=True)

# 1. Load cleaned dataset from Phase 3
df = pd.read_csv("data/cleaned_dataset.csv")
df["clean_text"] = df["clean_text"].fillna("")  # safety net in case of any leftover NaNs

# 2. Class balance: how many fake vs real articles
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Fake (0)", "Real (1)"])
plt.title("Fake vs Real News Count")
plt.ylabel("Number of Articles")
plt.savefig("charts/class_balance.png")
plt.show()

# 3. Article length distribution (word count) by class
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="word_count", hue="label", bins=50, kde=True)
plt.title("Article Length Distribution (Fake vs Real)")
plt.xlabel("Word Count")
plt.xlim(0, 1000)  # zoom in, most articles are under 1000 words
plt.savefig("charts/word_count_distribution.png")
plt.show()

print("\nAverage word count - Fake:", df[df["label"] == 0]["word_count"].mean().round(1))
print("Average word count - Real:", df[df["label"] == 1]["word_count"].mean().round(1))

# 4. Word cloud for FAKE news
fake_text = " ".join(df[df["label"] == 0]["clean_text"])
fake_wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(fake_text)
plt.figure(figsize=(10, 5))
plt.imshow(fake_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words - FAKE News")
plt.savefig("charts/wordcloud_fake.png")
plt.show()

# 5. Word cloud for REAL news
real_text = " ".join(df[df["label"] == 1]["clean_text"])
real_wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(real_text)
plt.figure(figsize=(10, 5))
plt.imshow(real_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words - REAL News")
plt.savefig("charts/wordcloud_real.png")
plt.show()

# 6. Top 20 most frequent words overall (bar chart)
all_words = " ".join(df["clean_text"]).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)
words, counts = zip(*top_words)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 20 Most Frequent Words (Overall)")
plt.xlabel("Frequency")
plt.savefig("charts/top_words.png")
plt.show()

print("\nAll charts saved to the charts/ folder.")
print("EDA complete.")