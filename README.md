<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />

# Pazham     🍌
A machine learning model that predicts multiple features of a banana based on its physical characteristics:
1. Number of seeds
2. Curvature (in degrees)


## Basic Details
### Team Name: (AB)²


### Team Members
- Team Lead: Atul Biju - Adi Shankara Institute of Engineering and Technology
- Member 2: Amal Babu - Adi Shankara Institute of Engineering and Technology


## Overview

This project uses a Random Forest Regressor to predict multiple banana characteristics based on various physical features. The model achieves good accuracy (R² scores > 0.80) on synthetic data and can be retrained with real-world data.


## Features

### Input Features
The model takes the following measurements as input:
- Length (centimeters)
- Width (centimeters)
- Weight (grams)
- Ripeness level (scale 1-5)
- Color (1=green, 2=yellow, 3=brown)

### Predictions
The model predicts:
1. Number of seeds
2. Curvature (degrees), reported to two decimal places

Curvature is measured by the vision pass rather than the trained model, whose curvature
head returns a near-constant regardless of the image.

### Unnecessary Measurements

Every upload also returns seven further readings, none of which anyone needs:

| Reading | Output |
| --- | --- |
| Ripeness detector | Ripeness as an exact percentage |
| Spot counter | Brown spots visible now, plus a three-day forecast |
| Shape classification | Straight / Slightly curved / Curved / Extremely curved / Suspicious |
| Weight vs. length analyzer | Density in g/cm3, and whether the banana would float |
| End detection | Which end is the top, then asks whether you are sure |
| Bruise detector | Bounding boxes drawn around bruises you can already see |
| Rotation counter | Degrees turned on a draggable platform, to no end |

Spot count and bruise boxes are measured in pixels by `webapp/spot_detect.py`: the peel is
separated from the background, pixels meaningfully darker than it are marked, and those are
grouped into connected regions. Small regions are speckles, large ones are bruises. This runs
in about 12 ms and needs no API. Blemishes that span half the frame or run off its edge are
discarded, so a cluttered photo reports nothing rather than boxing the tablecloth.

Shape class, volume, density and the spot forecast are arithmetic derived in
`webapp/model_utils.py`. Ripeness, curvature, the top end and the raw dimensions are read
from the photo by the vision pass, which shares a single API call with the seed-count
sanity-check so the page costs two requests rather than nine. Without a `GROQ_API_KEY`
those readings degrade to "could not be measured" rather than failing the request.

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - scikit-learn

## Usage

The model is implemented in a Jupyter notebook (`model.ipynb`). To use it:

1. Open `model.ipynb` in Jupyter or VS Code
2. Run all cells to train the model
3. Use the `predict_seeds()` function with your banana measurements

Example usage:
```python
predictions = predict_banana_features(
    length=16,    # cm
    width=3.2,    # cm
    weight=130,   # g
    ripeness=4,   # scale 1-5
    color=2       # yellow
)

print(f"Predicted seeds: {predictions['seeds']}")
print(f"Predicted curvature: {predictions['curvature']}°")
```

## Model Performance

Current model metrics on synthetic data:
- Mean Squared Error: 0.20
- R² Score: 0.80

Note: These metrics are based on synthetic training data. Performance may vary with real-world data.

# Screenshots
![3](https://github.com/user-attachments/assets/c5d8c609-2515-45b2-8423-5dd5a94a8deb)
Displays training progress and loss metrics during model training.   <br><br><br>

![1](https://github.com/user-attachments/assets/3fe1baf1-7142-44cd-9164-443e93fe948a)
Select or capture a banana image to predict seed count and curvature.  <br><br><br>


![2](https://github.com/user-attachments/assets/489dd5e4-a178-4045-bcee-4403d28a2f5d)
Displays predicted seed count and curvature angle after analysis.   <br><br>



## Future Improvements

- Replace synthetic data with real banana measurements
- Add image processing to automatically extract features
- Implement cross-validation
- Add visualization of feature importance

## License

[MIT License](LICENSE)

Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)
