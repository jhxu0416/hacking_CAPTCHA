# Hacking CAPTCHA
Fun project hacking [Really Simple CAPTCHA](https://wordpress.org/plugins/really-simple-captcha/#description)

Initial idea and training sets: [reference_0](http://python.jobbole.com/89004/)
[reference_1](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/)

## Data set

Data set is in s3: (s3://captcha-training-img/wordpress.zip)

## Methodology I (Neural Network)

### Extracting single character from given CAPTCHA

Given CAPTCHA, use openCV to find contour of each charachter, and save them into training dir. I'm using naive method to separate multiple charachters in one contour based on width-height ratio. Please let me know it you have a better idea about separating them.

[Detail](/src/extract_character.ipynb)

### Train the neural network for single character

[Detail](/src/train_model.ipynb)

<img src='src/captcha_model.png'><br>

### Use the model to predict CAPTCHA

[Detail](/src/solve_captcha.ipynb)

## Methodology II (Eigenvector)

Under construction...