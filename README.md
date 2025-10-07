# ml-poc-classification

A light-weight proof of concept (PoC) for text classification using a fine-tuned BERT model. The project demonstrates hate speech detection by classifying raw input text into pre-defined labels via a simple web interface.

---

## 📋 Project Overview

Minimal Machine Learning pipeline for BERT (`bert-base-uncased`)-based text classification. Includes training, evaluating, and a very small Flask-based web application for real-time inference.

### Core Features

- Load CSV data and train a BERT-based classifier
- Evaluate classification performance
- Run a Flask server for real-time text classification
- Predicts the **type of hate speech** based on user input

This PoC is developed as a technical foundation for a future larger project that will include OCR integration, database support, and real-time communication via WebSockets.

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/NathanGrs00/ml-poc-classification.git
cd ml-poc-classification
```
### 2️⃣ Install Dependencies

Use `pip` to install all the required packages:

```bash
pip install -r requirements.txt
```

> 💡 Python 3.8+ is recommended.

### 3️⃣ Launch the Web Server

Start the Flask app:

```bash
python main.py
```

Click the localhost link flask generates. It will show a simple HTML page with an input field where you can type or paste text to classify.

---

## 🚀 Usage

1. Go to `http://localhost:5000`
2. Enter any text (e.g., a tweets or comments)
3. Click the "Submit" button
4. The model returns a label with the type of hate speech (if any)

---

## 🗺️ Roadmap

Feedback from peer students included the following changes to be implemented in the Proof of Concept:
- Model is not reliable enough.
  - [ ] Extend dataset with more data.
  - [ ] Balance the percentage of hateful and neutral comments.
  - [x] Verify the current labeled data in the dataset.
- UI looks too simple, result is hard to read.
  - [x] Change the labels to a more formatted version. Instead of just displaying 'violence' display a little more information.
  - [x] Give each result a colored label

This PoC is a stepping stone towards a larger application. Future plans are:

- 🧾 Adding OCR to scan document/text image input
- 🗄️ Database connection for saving classification output and user feedback.
- 🔌 WebSockets for real-time updates

---

## 📣 Feedback & Contribution

I love feedback and contributions!

Report bugs or request features via [GitHub Issues](https://github.com/NathanGrs00/ml-poc-classification/issues).

Fork the repository and make a Pull Request to contribute.

Feel free to propose improvements!

---

## ✅ License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for more information.

---

## 🤝 Contributing

Contributions are always welcome!

To contribute:

1. **Fork** the repository
2. **Create a new branch** for your feature or bugfix
3. **Commit** your changes with clear messages
4. **Push** to your fork
5. **Open a Pull Request** describing your changes

For major changes, consider opening an issue first to discuss your proposal.


