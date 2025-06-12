# 🚦 Automatic Memo System for Traffic Offenses

An AI-powered system that generates formal memos for traffic violations like helmetless riding, overspeeding, and other offenses. Designed for use by law enforcement, this tool automates memo creation from raw text or audio evidence.

---

## 📌 Features

- 📄 Auto-generate memos from incident notes or voice reports
- 🧠 Intelligent detection of traffic violations (helmet, speed, etc.)
- 🏷️ Extracts key data: offender details, timestamp, and location
- 🗂️ Supports export in Markdown and PDF formats
- 🧩 Modular architecture for easy integration and extension

---

## 🏗️ Folder Structure

automatic_memo_system/├── data/ # Input image/video data

├── offense_detection/ # Violation classification logic

├── memo_generator/ # AI-powered memo creation

├── templates/ # Markdown templates for memos

├── output/ # Generated memos

├── tests/ # Unit tests

├── main.py # Main executable script

└── requirements.txt # Python dependencies
