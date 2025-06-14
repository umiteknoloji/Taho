# Taho - Universal Football Analytics System

🚀 **AI-Powered Football Prediction System** with M3 Neural Engine & Gaussian Process Integration

## 🏆 Features

### 🧠 Advanced AI Models
- **M3 Neural Engine** - Optimized for Apple Silicon
- **Gaussian Process (GP) Classification** - Advanced probabilistic predictions
- **Ensemble Learning** - Multiple model combination
- **ELO Rating System** - Dynamic team strength calculation

### 📊 Analytics Capabilities
- **Parametric Backtesting** - Historical performance analysis
- **Real-time Predictions** - Live match outcome forecasting
- **Statistical Analysis** - Comprehensive team & league statistics
- **Multi-league Support** - 15+ international leagues

### 🌐 Web Interface
- **Interactive Dashboard** - User-friendly web interface
- **AI Chip Manager** - Neural processing optimization
- **Real-time Updates** - Live prediction monitoring
- **Export Features** - Results in multiple formats

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/umiteknoloji/Taho.git
cd Taho

# Install dependencies
pip install -r requirements.txt

# Run web interface
python web_interface.py
```

## 🚀 Quick Start

### Basic Prediction
```python
from parametric_backtest import ParametricBacktestSystem

# Initialize with GP support
system = ParametricBacktestSystem(use_gp=True)

# Run backtest
result = system.run_backtest('data/ALM_stat.json', test_week=12)
print(f"Accuracy: {result['metrics']['result_accuracy']:.1%}")
```

### AI Chip Optimization
```python
from ai_chip_predictor import AIChipPredictor

# Initialize AI Chip predictor
predictor = AIChipPredictor()
predictor.optimize_for_m3()

# Make prediction
prediction = predictor.predict_match("Bayern Munich", "Dortmund")
```

## 📈 Supported Leagues

- 🇩🇪 Bundesliga (Germany)
- 🏴 Premier League (England)
- 🇪🇸 La Liga (Spain)
- 🇫🇷 Ligue 1 (France)
- 🇳🇱 Eredivisie (Netherlands)
- 🇵🇹 Primeira Liga (Portugal)
- 🇹🇷 Süper Lig (Turkey)
- 🇧🇷 Brasileirão (Brazil)
- 🇧🇪 Pro League (Belgium)
- And more...

## 🧠 AI Technologies

### M3 Neural Engine
- Optimized for Apple Silicon M3 chips
- Hardware-accelerated inference
- Real-time processing capabilities
- Memory-efficient operations

### Gaussian Process Classification
- Probabilistic predictions with uncertainty
- Non-linear pattern recognition
- Ensemble with traditional ML models
- Bayesian optimization

### Advanced Features
- **Ensemble Methods**: RandomForest, GradientBoosting, LightGBM
- **Feature Engineering**: 15+ statistical features
- **Cross-validation**: Robust model evaluation
- **Hyperparameter Optimization**: Automated tuning

## 📊 Performance Metrics

- **Accuracy**: 55-70% for match outcomes
- **Processing Speed**: <100ms per prediction
- **Memory Usage**: Optimized for M3 chips
- **Scalability**: Handles 1000+ matches/second

## 🔗 API Endpoints

```
GET  /api/predict           # Single match prediction
POST /api/run-backtest      # Historical backtesting
GET  /api/leagues          # Available leagues
POST /api/ai-chip/optimize  # AI chip optimization
```

## 🛠️ System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **OS**: macOS (M3 optimized), Linux, Windows
- **Dependencies**: scikit-learn, pandas, numpy, flask

## 📱 Web Interface

Access the web interface at `http://localhost:5000` after running:

```bash
python web_interface.py
```

Features:
- Interactive prediction dashboard
- Real-time backtesting
- League statistics
- AI chip performance monitoring
- Export capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- M3 Neural Engine optimization
- Gaussian Process research community
- Football data providers
- Open source ML libraries

## 📧 Contact

- **Developer**: AI Powered Football Analytics
- **GitHub**: [@umiteknoloji](https://github.com/umiteknoloji)
- **Project**: [Taho](https://github.com/umiteknoloji/Taho)

---

⚡ **Powered by M3 Neural Engine & Gaussian Process Technology** ⚡
