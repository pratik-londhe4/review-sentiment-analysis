from tkinter import Tk, Label, Entry, Button, messagebox
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import time
from selenium.common.exceptions import NoSuchElementException
from tkinter import Tk, Label, Entry, Button, messagebox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def fetch_reviews_amazon(url, label_text):
    # Set up Firefox WebDriver
    options = Options()
    options.headless = True  # Run Firefox in headless mode
    service = Service("geckodriver")  # Path to geckodriver executable
    driver = webdriver.Firefox(service=service, options=options)
    driver.get(url)
    time.sleep(2)
    see_all_reviews_link = driver.find_element(
        By.XPATH, "//a[@data-hook='see-all-reviews-link-foot']"
    )
    see_all_reviews_link.click()
    print("fetching reviews")

    reviews = []

    review_count = 0  # Initialize the review count

    # Fetch all review elements
    while len(reviews) < 100:
        # Fetch all review elements on the current page
        review_elements = driver.find_elements(
            By.CSS_SELECTOR, "[data-hook='review-body']"
        )

        # Extract the text of each review element
        for review_element in review_elements:
            review_text = review_element.text.strip()
            reviews.append(review_text)
            review_count += 1  # Increment the review count
            label_text.config(text=f"Reviews fetched: {review_count}")
            window.update_idletasks()  # Update the label immediately

            # Break the loop if the maximum number of reviews has been reached
            if len(reviews) >= 100:
                break

        # Check if there is a next page button
        try:
            next_page_button = driver.find_element(By.LINK_TEXT, "Next page")
            # Click on the next page button if it exists
            next_page_button.click()
            time.sleep(2)
        except NoSuchElementException:
            break

    # Close the WebDriver
    driver.quit()

    df = pd.DataFrame({"review": reviews})

    return df


def analyze_reviews():
    # Get the entered product link from the entry widget
    product_url = entry_link.get()

    label_text = Label(window, text="fetching reviews")
    label_text.grid(row=3, column=1, padx=10, pady=5)
    df = fetch_reviews_amazon(product_url, label_text)

    with open("classifier.pickle", "rb") as file:
        classifier = pickle.load(file)

    with open("vectorizer.pickle", "rb") as file:
        vectorizer = pickle.load(file)

    def predict_sentiment(text):
        # Transform the text using the vectorizer
        X = vectorizer.transform([text])

        # Predict the sentiment using the classifier
        sentiment = classifier.predict(X)

        return sentiment[0]

    df["review"] = df["review"].astype(str)
    df["predicted_sentiment"] = df["review"].apply(predict_sentiment)

    # Print the DataFrame with predicted sentiments

    sentiment_counts = df["predicted_sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Analysis")

    # Add count labels on top of each bar
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count, str(count), ha="center", va="bottom")

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, padx=10, pady=5)

    plt.close(fig)


window = Tk()
window.geometry("700x600")
window.title("Amazon Product Reviews Analysis")

# Create a label and entry widget for the product link
label_link = Label(window, text="Enter Amazon Product Link:")
label_link.grid(row=0, column=0, padx=10, pady=5)
entry_link = Entry(window, width=50)
entry_link.grid(row=0, column=1, padx=10, pady=5)

# Create a button to trigger the analysis
button_analyze = Button(window, text="Analyze Reviews", command=analyze_reviews)
button_analyze.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

# Run the GUI event loop
window.mainloop()
