# Description Genius 🦉
[Features](#features) • [Getting Started](#getting-started) • [Disclaimer](#disclaimer)

Easily generate captivating product descriptions using product features and any additional sources of information that you may have e.g. product reviews, usage instructions etc.

## Features
Description Genius brings the latest advances in Generative AI in an easy to use package.

The interactive UI lets you upload a CSV file containing a list of your products alongside their features. An example CSV file could look like this:

| Id | Color | Material | Product Group | Sizes | Price | Care Guide |
|---|---|---|---|---|---|---|
| 1 | Black | Leather | Pants | s/m/l | $250 | Hand wash |
| 2 | Purple | Cotton | Sweater | m | $99 | Machine washable at 45C |


## Getting Started
Description Genius comes with an easy to use UI built on top of [Streamlit](https://streamlit.io/).

To get started:
1) Clone this repository
2) Change to the project directory and install the required libraries using `pip install -r requirements.txt`
3) Launch the app using `streamlit run app.py`


This should open up a browser window with the Description Genius UI. Simply upload a CSV file containing your product features alongside a prompt for the language model. Also provide a few examples of how your output should look like. Optionally, you can include additional sources of information such as styling guidelines, material information etc.

Once done, click on `Generate`. Description Genius will now generate your descriptions and output them in a table directly in the UI. You can now make any edits to the generated text and finally download it by clicking the `Download data as CSV` button.

## Disclaimer

**This is not an officially supported Google product.**

*Copyright 2023 Google LLC. This solution, including any related sample code or data, is made available on an “as is,” “as available,” and “with all faults” basis, solely for illustrative purposes, and without warranty or representation of any kind. This solution is experimental, unsupported and provided solely for your convenience. Your use of it is subject to your agreements with Google, as applicable, and may constitute a beta feature as defined under those agreements. To the extent that you make any data available to Google in connection with your use of the solution, you represent and warrant that you have all necessary and appropriate rights, consents and permissions to permit Google to use and process that data. By using any portion of this solution, you acknowledge, assume and accept all risks, known and unknown, associated with its usage, including with respect to your deployment of any portion of this solution in your systems, or usage in connection with your business, if at all.*
