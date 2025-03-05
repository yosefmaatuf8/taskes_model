# Project Setup Guide

##  Project Installation Guide

This guide will help you set up the project on **Windows**, **Linux**, and **macOS**.

---

##  Prerequisites

Before installing the project, ensure you have:

1. **Python 3.8+** installed  
   - Check with:  
     ```sh
     python --version
     ```
     or  
     ```sh
     python3 --version
     ```

2. **pip** (Python package manager) installed  
   - Check with:  
     ```sh
     pip --version
     ```

3. **Git** installed (for cloning the repository)  
   - Check with:  
     ```sh
     git --version
     ```

---

##  Installation Steps

### 1️⃣ Clone the Repository

```sh
git clone <your-repo-url>
cd <your-project-folder>
```

---

### 2️⃣ Create a Virtual Environment

#### 💻 Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

#### 🐧 Linux/macOS:
```sh
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies & Setup Project

Run the following command to install the project in **editable mode** (for development):

```sh
pip install -e .
```

This will:
- Install all dependencies from `setup.py` and `requirements.txt`
- Set up the project in editable mode, so any changes take effect immediately.

---

### 4️⃣ Create and Configure `.env` File

Copy the example environment file and modify it with your credentials:

```sh
cp .env_example .env
```
Then, open `.env` and update the required variables.

---

### 5️⃣ Run the Project

Now, you can start using the project:

```sh
python src/main_manager.py
```

---

## ⚙️ Additional Commands

### ✅ Update Dependencies:
```sh
pip install -r requirements.txt
```

### ✅ Run Tests:
```sh
pytest tests/
```

### ✅ Deactivate Virtual Environment:
```sh
deactivate
```

---

##  Troubleshooting

### "Command Not Found" Errors:
- Ensure Python and pip are installed correctly.
- If using Windows, try `python3` or `py` instead of `python`.

### Virtual Environment Not Activating:
- Windows: Try `venv\Scripts\activate.bat` (CMD) or `venv\Scripts\Activate.ps1` (PowerShell).
- macOS/Linux: Use `source venv/bin/activate`.

---

##  License
This project is licensed under **MIT License**.

---

##  Support
For issues, open a GitHub issue or contact the project maintainers.