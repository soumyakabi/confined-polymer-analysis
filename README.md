# Confined Polymer Analysis (EXACT Implementation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

A professional Streamlit web application for analyzing polymer chain distributions in cylindrical confinement using Monte Carlo simulations and analytical solutions.

## 🎯 Overview

This application implements **exact computational methods** for studying confined polymer chains using:

- **FJC (Freely Jointed Chain)** - The simplest polymer model with independent segments
- **SAW (Self-Avoiding Walk)** - Advanced model that prevents chain self-intersection
- **WLC (Wormlike Chain)** - Sophisticated model incorporating chain stiffness

The app analyzes two types of distributions:
- **P(x)** - Longitudinal distribution along the confinement axis
- **P(y)** - Transverse distribution perpendicular to the confinement axis

**Key Features:**
- Interactive parameter adjustment
- Real-time Monte Carlo simulations (up to 200,000 walkers)
- Bootstrap error estimation (800+ samples)
- Analytical solution comparison (Fourier series, Gaussian approximations)
- Professional PDF export of results
- Docker containerization for easy deployment

---

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
  - [Windows 10/11 Installation](#windows-1011-installation)
  - [Linux Installation](#linux-installation)
  - [macOS Installation](#macos-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)
- [Support](#-support)

---

## 💻 System Requirements

### Minimum Requirements
- **CPU:** 2 cores (4+ cores recommended for faster simulations)
- **RAM:** 4 GB (8+ GB recommended)
- **Storage:** 5 GB free space
- **Internet:** Required for initial setup

### Operating Systems
- Windows 10/11 (Pro, Enterprise, or Education - version 1903 or later)
- Linux (Ubuntu 20.04+, Debian 10+, Fedora 33+, or similar)
- macOS 10.15+ (Intel or Apple Silicon)

---

## 🚀 Installation Guide

### Windows 10/11 Installation

Follow these steps carefully to set up the application on Windows:

#### Step 1: Enable Virtualization

1. **Check if virtualization is enabled:**
   - Open **Task Manager** (Ctrl + Shift + Esc)
   - Go to **Performance** tab → **CPU**
   - Check if "Virtualization" shows **Enabled**
   
2. **If disabled, enable it in BIOS:**
   - Restart your computer
   - Press **F2**, **F10**, **Del**, or **Esc** during boot (key depends on manufacturer)
   - Find **Virtualization Technology** (Intel VT-x or AMD-V)
   - Set to **Enabled**
   - Save and exit BIOS

#### Step 2: Enable WSL 2 (Windows Subsystem for Linux)

1. **Open PowerShell as Administrator:**
   - Press **Windows key**
   - Type "PowerShell"
   - Right-click **Windows PowerShell**
   - Select **Run as administrator**

2. **Enable required Windows features:**
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **Restart your computer**

4. **Set WSL 2 as default version:**
   ```powershell
   wsl --set-default-version 2
   ```

5. **Update WSL kernel (if prompted):**
   ```powershell
   wsl --update
   ```

#### Step 3: Install Ubuntu Linux Distribution

1. **Install Ubuntu via command line:**
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Launch Ubuntu:**
   - Search for "Ubuntu" in Start Menu
   - First launch will take 2-5 minutes to set up
   - Create a **Linux username** and **password** when prompted
   - Remember these credentials!

3. **Verify WSL version:**
   ```powershell
   wsl -l -v
   ```
   - You should see `Ubuntu-22.04` with `VERSION` showing `2`

4. **Update Ubuntu packages:**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

#### Step 4: Enable Hyper-V and Containers (Windows Pro/Enterprise only)

1. **Open PowerShell as Administrator**

2. **Check current status:**
   ```powershell
   Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V, Containers
   ```

3. **Enable features:**
   ```powershell
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   Enable-WindowsOptionalFeature -Online -FeatureName Containers -All
   ```

4. **Restart your computer when prompted**

#### Step 5: Install Docker Desktop

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click **Download for Windows**

2. **Run the installer:**
   - Double-click the downloaded `.exe` file
   - Check **"Use WSL 2 instead of Hyper-V"** option
   - Complete installation
   - Restart computer if prompted

3. **Start Docker Desktop:**
   - Launch from Start Menu
   - Wait for Docker to fully start (whale icon in system tray)

4. **Configure WSL Integration:**
   - Open Docker Desktop
   - Go to **Settings** (gear icon)
   - Navigate to **Resources → WSL Integration**
   - Enable integration with **Ubuntu-22.04**
   - Click **Apply & Restart**

5. **Verify Docker installation:**
   ```powershell
   docker --version
   docker compose version
   ```

#### Step 6: Download and Run the Application

1. **Download this repository:**
   - Click the green **Code** button on GitHub
   - Select **Download ZIP**
   - Extract to `C:\PolymerAnalysis` (or your preferred location)

2. **Open PowerShell in the project directory:**
   - Navigate to the extracted folder
   - Hold **Shift** and **Right-click** in the folder
   - Select **Open PowerShell window here**

3. **Start the application:**
   ```powershell
   docker compose up
   ```
   - First run takes 3-5 minutes to download and build
   - Subsequent runs take only seconds

4. **Access the application:**
   - Open your web browser
   - Visit: **http://localhost:8501**

5. **Stop the application:**
   - Press **Ctrl + C** in PowerShell
   - Or run: `docker compose down`

---

### Linux Installation

#### Ubuntu/Debian-based Systems

1. **Update system packages:**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install Docker:**
   ```bash
   # Install prerequisites
   sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
   
   # Add Docker GPG key
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   
   # Add Docker repository
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   
   # Install Docker
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```

3. **Add your user to Docker group:**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

4. **Verify Docker installation:**
   ```bash
   docker --version
   docker compose version
   ```

5. **Download and run the application:**
   ```bash
   # Clone or download repository
   git clone https://github.com/YOUR_USERNAME/confined-polymer-analysis.git
   cd confined-polymer-analysis
   
   # Start application
   docker compose up
   ```

6. **Access at:** http://localhost:8501

#### Fedora/RHEL-based Systems

1. **Install Docker:**
   ```bash
   sudo dnf -y install dnf-plugins-core
   sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
   sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. **Add user to Docker group:**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **Follow steps 5-6 from Ubuntu installation above**

---

### macOS Installation

1. **Download Docker Desktop for Mac:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download for **Mac with Intel chip** or **Mac with Apple chip**

2. **Install Docker Desktop:**
   - Open the downloaded `.dmg` file
   - Drag Docker icon to Applications folder
   - Launch Docker from Applications
   - Grant necessary permissions when prompted

3. **Verify Docker:**
   ```bash
   docker --version
   docker compose version
   ```

4. **Download and run application:**
   ```bash
   # Clone or download repository
   git clone https://github.com/YOUR_USERNAME/confined-polymer-analysis.git
   cd confined-polymer-analysis
   
   # Start application
   docker compose up
   ```

5. **Access at:** http://localhost:8501

---

## ⚡ Quick Start

Once Docker is installed and the application is downloaded:

1. **Navigate to project directory** in terminal/PowerShell
2. **Start application:**
   ```bash
   docker compose up
   ```
3. **Open browser:** http://localhost:8501
4. **Configure parameters** in the left sidebar
5. **Click "COMPUTE"** button
6. **View results** and export as PDF/CSV

---

## 📖 Usage Guide

### Interface Overview

The application has two main sections:

**Sidebar (Left)** - Parameter Configuration:
- Distribution Type: P(x) or P(y)
- Geometry Parameters: Chain length, confinement dimensions
- Model Selection: FJC, SAW, WLC
- Analytical Solutions: Fourier, Gaussian, etc.
- Bootstrap Options

**Main Area (Right)** - Results:
- Interactive plots
- Statistical comparison tables
- Export buttons (PDF, CSV, PNG)

### Step-by-Step Analysis

1. **Select Distribution Type:**
   - **P(x)** for longitudinal analysis (along cylinder axis)
   - **P(y)** for transverse analysis (perpendicular to axis)

2. **Set Geometry Parameters:**
   - **Kuhn Length (a):** Segment length (typically 0.01-0.5 µm)
   - **Confinement Length (L):** Cylinder length for P(x)
   - **Cylinder Radius (R):** Confinement radius

3. **Configure Chain Parameters:**
   - **Chain Length (N):** Number of segments (5-100)
   - **Tethering Point:** Where chain is anchored

4. **Select Models:**
   - Check one or more: FJC, SAW, WLC
   - For WLC: Set persistence length (lₚ)

5. **Choose Analytical Solutions:**
   - Select comparison methods
   - Fourier series, Gaussian fits, etc.

6. **Run Simulation:**
   - Click "🚀 COMPUTE"
   - Wait for completion (10 seconds to 2 minutes depending on complexity)

7. **Analyze Results:**
   - Compare simulation vs analytical curves
   - Check statistical metrics (mean, std dev, KS test)
   - Review bootstrap error estimates

8. **Export Results:**
   - PDF: Professional report with all figures and statistics
   - CSV: Raw data for further analysis
   - PNG: High-resolution plots

---

## 🔬 Technical Details

### Monte Carlo Simulations

- **FJC:** 150,000 independent walkers
- **SAW:** Adaptive sampling with EXACT algorithm (5,000-20,000 walkers)
- **WLC:** 50,000 walkers with persistence length constraints

### Statistical Analysis

- **Bootstrap Resampling:** 800 samples for error estimation
- **Kernel Density Estimation:** Gaussian kernels with adaptive bandwidth
- **Kolmogorov-Smirnov Test:** Distribution comparison metric

### Analytical Solutions

- **Fourier P(x):** 600-term series expansion
- **Fourier P(y):** Bessel function series
- **Gaussian Approximation:** Standard and truncated variants
- **Image Method:** Reflection boundary conditions

### Performance

- **Computation Time:** 10 seconds - 2 minutes
- **Memory Usage:** 2-4 GB RAM
- **CPU Utilization:** Multi-core optimized with Numba JIT compilation

---

## 🔧 Troubleshooting

### Windows Issues

**Issue:** "Virtualization not enabled"
- **Solution:** Enable VT-x/AMD-V in BIOS settings (see Step 1)

**Issue:** "WSL 2 installation failed"
- **Solution:** Update Windows to latest version, then retry WSL installation

**Issue:** "Docker Desktop not starting"
- **Solution:** 
  1. Restart computer
  2. Check WSL 2 is running: `wsl -l -v`
  3. Update WSL: `wsl --update`
  4. Reinstall Docker Desktop

**Issue:** "Port 8501 already in use"
- **Solution:** Stop other applications using port 8501, or edit `docker-compose.yml` to use different port

### Linux Issues

**Issue:** "Permission denied" when running Docker
- **Solution:** Add user to docker group: `sudo usermod -aG docker $USER && newgrp docker`

**Issue:** "Cannot connect to Docker daemon"
- **Solution:** Start Docker service: `sudo systemctl start docker`

### Application Issues

**Issue:** Sidebar widgets duplicated
- **Solution:** Clear browser cache and refresh page

**Issue:** Computation takes too long
- **Solution:** Reduce chain length (N) or number of models selected

**Issue:** Out of memory error
- **Solution:** Close other applications, or increase Docker memory limit in settings

---

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{confined_polymer_analysis_2025,
  author = {Your Name},
  title = {Confined Polymer Analysis: EXACT Implementation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/confined-polymer-analysis},
  doi = {10.5281/zenodo.XXXXX},
  version = {1.0.0}
}
```

**Associated Paper:**
```
[Your Paper Title]
Authors: [Your Name et al.]
Journal: [Journal Name]
DOI: [Paper DOI]
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🤝 Support

### Getting Help

- **Issues:** Report bugs or request features on [GitHub Issues](https://github.com/YOUR_USERNAME/confined-polymer-analysis/issues)
- **Discussions:** Ask questions in [GitHub Discussions](https://github.com/YOUR_USERNAME/confined-polymer-analysis/discussions)
- **Email:** your.email@university.edu

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

### Acknowledgments

- Developed for research in polymer physics and soft matter
- Built with Streamlit, NumPy, SciPy, and Numba
- Docker containerization for reproducibility

---

## 📊 Project Structure

```
confined-polymer-analysis/
├── polymer-analysis-app.py   # Main Streamlit application
├── pdf_export_module.py      # PDF report generation
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose orchestration
├── run.sh                     # Linux/Mac launch script
├── run_fixed.bat              # Windows launch script
├── Repair-DockerWSL.ps1       # Windows troubleshooting script
├── README.md                  # This file
├── LICENSE                    # MIT License
└── CITATION.cff               # Citation metadata
```

---

## 🎓 Educational Use

This application is designed for:
- **Researchers:** Computational polymer physics studies
- **Students:** Learning Monte Carlo methods and polymer theory
- **Educators:** Teaching material for soft matter courses

---

**Version:** 1.0.0  
**Last Updated:** November 16, 2025  
**Maintainer:** [Your Name]  
**Status:** Active Development  

---

For the latest updates and releases, visit: [GitHub Repository](https://github.com/soumyakabi/confined-polymer-analysis)
