# Number Beauty: The Aesthetics of Lottery Numbers

Number Beauty is a Python project designed to analyze the beauty criteria of lottery numbers, particularly focusing on Spanish lottery ticket sales. Inspired by societal preferences for certain numerical patterns, this project explores the characteristics that make a number "beautiful" or "ugly" and examines their impact on lottery sales.

## Background

The project is inspired by the curious superstitions surrounding lottery ticket sales. Historically, "ugly" numbers have been returned unsold, even when they later won substantial prizes. This repository investigates the beauty criteria and public preferences that lead to these perceptions of beauty or aversion in numbers.

## Beauty Criteria of Lottery Numbers

The project defines several key criteria for "beautiful" and "ugly" numbers based on historical sales data:

1. **Lucky 13**: Numbers ending in 13 are highly sought after, selling at rates of up to 98.77%. When starting with 13, sales are slightly lower but still significant (87.83%).

2. **The Unlucky Zeros**: Left-leading zeros are seen as useless, while right-ending zeros are avoided for being too "perfect" and round.

3. **Repetition Aversion**: Repeated digits are usually disliked, but symmetrical perfection, like "00000" or "99999," is highly prized and sells well, with nearly 98% of such tickets sold.

## Repository Structure

- **notebooks/**: Contains the exploratory data analysis (EDA) notebook that investigates beauty criteria through descriptive statistics and visualizations.
- **resources/**: Includes images and other resources referenced in the app.
- **requirements.txt**: Lists the required libraries for replicating the analysis.
- **src/**: Houses Python scripts for analyzing and visualizing lottery numbers based on beauty criteria.
- **.streamlit/**: Includes streamlit app configuration specs. 

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/almudenamcastro/number_beauty.git
    cd number_beauty
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Analyzing Lottery Numbers

To explore the dataset and run the beauty analysis, open the EDA notebook:

```bash
jupyter notebook notebooks/1-%20eda.ipynb
```

Inside, you'll find a breakdown of beauty criteria applied to the dataset, with visualizations that explore patterns such as the preference for numbers ending in "13," aversions to zeros, and patterns of digit repetition.

## Contributing

1. Fork this repo and create your branch (`git checkout -b feature-branch`).
2. Commit your changes (`git commit -m 'Add a new feature'`).
3. Push to the branch (`git push origin feature-branch`).
4. Open a pull request.